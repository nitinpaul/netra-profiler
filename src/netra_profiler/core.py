import time
import warnings
from typing import Any

import polars as pl

from netra_profiler import engine

from . import __version__
from .alerts import DiagnosticEngine

CORRELATION_SAMPLE_SIZE = 100_000


class Profiler:
    """
    The main entry point for Netra-Profiler

    This class manages the lifecycle of a profiling session:
        1. Ingestion (DataFrame -> LazyFrame)
        2. Plan Construction (Delegated to engine)
        3. Execution (Multi-Pass Strategy)
    """

    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        """
        Initializes the profiler.

        Args:
            df: A Polars DataFrame (eager) or LazyFrame.
        """
        # If a DataFrame is passed, it is converted to LazyFrame to
        # ensure all downstream operations use the query optimizer.
        if isinstance(df, pl.DataFrame):
            self._df = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            self._df = df
        else:
            raise TypeError(f"Unsupported type: {type(df)}. Must be pl.DataFrame or pl.LazyFrame")

        # OPTIMIZATION: Preprocess Complex Types (Structs/Lists)
        # This "flattens" the data view for the engine, enabling
        # support for nested JSON/Parquet without engine changes.
        self._df = engine.preprocess_complex_types(self._df)

    def profile(self, bins: int = 20, top_k: int = 10) -> dict[str, Any]:
        """
        Executes the profiling plan and returns the results.

        This uses a Multi-Pass strategy to ensure stability:
        1. Scalars (Min, Max, Mean)
        2. Histograms (Distributions)
        3. Top-K (Frequent Items)
        4. Correlations (Pearson & Spearman)
        5. Alerts (Diagnostics)

        Args:
            bins: Number of bins for histograms (default: 20)
            top_k: Number of most frequent items to return for text columns (default: 10)

        Returns:
            A dictionary containing the combined statistics and
            a '_meta' key with execution details and warnings.
        """
        start_time = time.time()
        warnings_list: list[str] = []

        # PASS 1: Scalar Statistics (Foundation)
        profile = self._run_scalar_pass()

        # Initialize Metadata
        profile["_meta"] = {
            "timestamp": time.time(),
            "version": __version__,
            "warnings": warnings_list,  # Reference to the local list we are appending to
            "correlation_method": None,
        }

        # PASS 2: Histograms
        self._run_histogram_pass(profile, bins, warnings_list)

        # PASS 3: Top-K Values
        self._run_top_k_pass(profile, top_k, warnings_list)

        # PASS 4: Correlations
        self._run_correlation_pass(profile, warnings_list)

        # PASS 5: Alerts (Diagnostics)
        self._run_alerts_pass(profile, warnings_list)

        # Finalize Metadata
        profile["_meta"]["engine_time"] = round(time.time() - start_time, 4)

        return profile

    def _run_scalar_pass(self) -> dict[str, Any]:
        """Executes Pass 1: Scalar Statistics."""
        try:
            scalar_plan = engine.build_scalar_plan(self._df)
            scalar_df = scalar_plan.collect(engine="streaming")
            profile = scalar_df.rows(named=True)[0]
        except Exception as e:
            # If Pass 1 fails, we can't really do anything.
            raise RuntimeError(f"Critical Error: {e}") from e

        return profile

    def _run_histogram_pass(
        self, profile: dict[str, Any], bins: int, warnings_list: list[str]
    ) -> None:
        """Executes Pass 2: Histograms"""
        try:
            # 1. Build all plans
            histogram_plans = engine.build_histogram_plans(self._df)

            if histogram_plans:
                # 2. PARALLEL EXECUTION
                # collect_all() runs all lazy plans in parallel threads.
                # This saturates I/O and CPU much better than a Python loop.
                eager_dfs = pl.collect_all(histogram_plans)

                # 3. Process results in memory (Fast)
                for eager_df in eager_dfs:
                    if eager_df.height > 0:
                        column_name = eager_df.columns[0]
                        try:
                            # Eager hist() ensures struct output
                            # It returns a DataFrame with columns:
                            # 'break_point', 'category', 'count'
                            histogram_df = eager_df[column_name].hist(bin_count=bins)
                            profile[f"{column_name}_histogram"] = histogram_df.to_dicts()
                        except Exception as e:
                            warnings_list.append(
                                f"Histogram generation failed for column '{column_name}': {e}"
                            )

        except Exception as e:
            warnings_list.append(f"Histogram generation failed: {e}")

    def _run_top_k_pass(
        self, profile: dict[str, Any], top_k: int, warnings_list: list[str]
    ) -> None:
        """Executes Pass 3: Top-K Values"""
        try:
            top_k_plans = engine.build_top_k_plan(self._df, k=top_k)

            for plan in top_k_plans:
                # We collect each column individually to keep memory usage low
                df = plan.collect()

                if df.height > 0:
                    column_name = df["column_name"][0]
                    values = df.select("value", "count").to_dicts()
                    profile[f"{column_name}_top_k"] = values

        except Exception as e:
            warnings_list.append(f"Top-K generation failed: {e}")

    def _run_correlation_pass(self, profile: dict[str, Any], warnings_list: list[str]) -> None:
        """Executes Pass 4: Correlations"""
        try:
            correlation_plan = engine.build_correlation_plan(self._df)

            # Check if we actually have columns to correlate
            # We use collect_schema() to check cheaply
            if len(correlation_plan.collect_schema()) > 1:
                # Adaptive sampling logic

                # We reuse the row count from Pass 1 (cost = 0)
                row_count = profile.get("table_row_count", 0)

                # Fetch Data (Sampled or Full)
                if row_count > CORRELATION_SAMPLE_SIZE:
                    # Case 1: Big Data -> Sample
                    # We cast to Float64 to handle potential integer overflow
                    correlation_df = (
                        correlation_plan.select(pl.all().cast(pl.Float64))
                        .collect()
                        .sample(n=CORRELATION_SAMPLE_SIZE, with_replacement=False, shuffle=True)
                    )
                    method_used = f"sampled (n={CORRELATION_SAMPLE_SIZE})"
                else:
                    # Case 2: Small Data -> Exact
                    correlation_df = correlation_plan.select(pl.all().cast(pl.Float64)).collect()
                    method_used = "exact"

                # We drop Nulls prevents NaN propagation in the matrix
                correlation_df = correlation_df.drop_nulls()

                # Compute Matrices
                if correlation_df.height > 0 and correlation_df.width > 1:
                    profile["correlations"] = {}

                    # We expect RuntimeWarnings (Divide by Zero) when correlating constant columns.
                    # This is normal behavior for dirty data, so we suppress the log noise locally.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)

                        # Compute PEARSON (Standard .corr())
                        try:
                            pearson_matrix = correlation_df.corr()
                            columns = pearson_matrix.columns

                            # Format: Add a 'column' column so we know which row is which variable
                            # Output structure: [{'column': 'age', 'age': 1.0, 'income': 0.8}, ...]
                            pearson_matrix = pearson_matrix.with_columns(
                                pl.Series(name="column", values=columns)
                            )
                            # Reorder so that "column" appears first and
                            # Clean NaNs (for JSON safety)
                            pearson_matrix = pearson_matrix.select(
                                pl.col("column"), pl.exclude("column").fill_nan(None)
                            )

                            profile["correlations"]["pearson"] = pearson_matrix.to_dicts()
                        except Exception as e:
                            warnings_list.append(f"Correlation (pearson) failed: {e}")

                        # Compute SPEARMAN (Rank data -> .corr())
                        try:
                            # Spearman is just Pearson on the ranks
                            spearman_matrix = correlation_df.select(pl.all().rank()).corr()

                            columns = spearman_matrix.columns

                            spearman_matrix = spearman_matrix.with_columns(
                                pl.Series(name="column", values=columns)
                            )

                            spearman_matrix = spearman_matrix.select(
                                pl.col("column"), pl.exclude("column").fill_nan(None)
                            )

                            profile["correlations"]["spearman"] = spearman_matrix.to_dicts()
                        except Exception as e:
                            warnings_list.append(f"Correlation (spearman) failed: {e}")

                # Metadata Injection
                profile["_meta"]["correlation_method"] = method_used

        except Exception as e:
            warnings_list.append(f"Correlation generation failed: {e}")

    def _run_alerts_pass(self, profile: dict[str, Any], warnings_list: list[str]) -> None:
        """Executes Pass 5: Diagnostic Alerts"""
        try:
            engine = DiagnosticEngine(profile)
            alerts = engine.run()

            # Serialize alerts to dicts for JSON output
            profile["alerts"] = [
                {
                    "column": a.column,
                    "type": a.alert_type,
                    "level": a.level.value,  # Convert Enum to string
                    "message": a.message,
                    "value": a.value,
                }
                for a in alerts
            ]
        except Exception as e:
            warnings_list.append(f"Alert generation failed: {e}")
