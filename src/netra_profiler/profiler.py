import datetime
import time
import warnings
from typing import Any, cast

import polars as pl

from netra_profiler import __version__, engine
from netra_profiler.diagnostics import DiagnosticEngine
from netra_profiler.types import NetraProfile, is_numeric, is_string_type

CORRELATION_SAMPLE_SIZE = 100_000


class Profiler:
    """
    The main entry point for Netra Profiler

    This class manages the lifecycle of a profiling session:
        1. Ingestion (DataFrame -> LazyFrame)
        2. Plan Construction (Delegated to engine)
        3. Execution (Multi-Pass Strategy)
    """

    def __init__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        dataset_name: str = "Unknown",
        dataset_format: str = "Unknown",
    ):
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

        self.dataset_name = dataset_name
        self.dataset_format = dataset_format

        # Preprocess Complex Types (Structs/Lists)
        # This "flattens" the data view for the engine, enabling
        # support for nested JSON/Parquet without engine changes.
        self._df = engine.preprocess_complex_types(self._df)

    def run(self, bins: int = 20, top_k: int = 10) -> NetraProfile:
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
        profiling_start_time = time.time()
        profiler_warnings: list[str] = []

        # PASS 1: Scalar Statistics (Foundation)
        profile_data = self._run_scalar_pass()

        # PASS 2: Histograms
        self._run_histogram_pass(profile_data, bins, profiler_warnings)

        # PASS 3: Top-K Values
        self._run_top_k_pass(profile_data, top_k, profiler_warnings)

        # PASS 4: Correlations
        self._run_correlation_pass(profile_data, profiler_warnings)

        # Build the structured Profile object based on our Profile Object Schema
        profile = self._build_profile_object(profile_data, profiling_start_time, profiler_warnings)

        # PASS 5: Diagnostics
        self._run_diagnostics_pass(profile, profiler_warnings)

        return profile

    def _run_scalar_pass(self) -> dict[str, Any]:
        """Executes Pass 1: Scalar Statistics (Streaming Mode)"""
        scalar_plan = engine.build_scalar_plan(self._df)
        scalar_df = scalar_plan.collect(engine="streaming")
        return scalar_df.rows(named=True)[0]

    def _run_histogram_pass(
        self, profile_data: dict[str, Any], bins: int, profiler_warnings: list[str]
    ) -> None:
        """Executes Pass 2: Histograms"""

        # 1. Build all plans
        histogram_plans = engine.build_histogram_plans(self._df)

        if histogram_plans:
            # 2. PARALLEL EXECUTION
            # collect_all() runs all lazy plans in parallel threads.
            # This saturates I/O and CPU much better than a Python loop.
            histogram_eager_dfs = pl.collect_all(histogram_plans)

            # 3. Process results in memory (Fast)
            for histogram_eager_df in histogram_eager_dfs:
                if histogram_eager_df.height > 0:
                    column_name = histogram_eager_df.columns[0]
                    try:
                        # We re-use the min-max values computed in Pass 1
                        # to calculate the bin edges
                        min_value = profile_data.get(f"{column_name}_min")
                        max_value = profile_data.get(f"{column_name}_max")

                        if (
                            min_value is not None
                            and max_value is not None
                            and min_value < max_value
                        ):
                            step = (max_value - min_value) / bins
                            # Calculate edges terminating at max_value to avoid float drift
                            edges = [min_value + (step * i) for i in range(bins)]
                            edges.append(max_value)

                            # Eager hist() ensures struct output
                            # It returns a DataFrame with columns:
                            # 'break_point', 'category', 'count'
                            # We rename 'category' to 'bin' for semantic clarity
                            histogram_df = (
                                histogram_eager_df[column_name]
                                .hist(bins=edges)
                                .rename({"category": "bin"})
                            )
                        else:
                            # Fallback for constant columns (min == max)
                            histogram_df = (
                                histogram_eager_df[column_name]
                                .hist(bin_count=bins)
                                .rename({"category": "bin"})
                            )

                        profile_data[f"{column_name}_histogram"] = histogram_df.to_dicts()
                    except Exception as e:
                        profiler_warnings.append(
                            f"Histogram generation failed for column '{column_name}': {e}"
                        )

    def _run_top_k_pass(
        self, profile_data: dict[str, Any], top_k: int, profiler_warnings: list[str]
    ) -> None:
        """Executes Pass 3: Top-K Values"""

        top_k_plans = engine.build_top_k_plan(self._df, k=top_k)

        if top_k_plans:
            # We execute all columns in parallel using the streaming engine
            top_k_dfs = pl.collect_all(top_k_plans, engine="streaming")

            for top_k_df in top_k_dfs:
                if top_k_df.height > 0:
                    column_name = top_k_df["column_name"][0]
                    top_k_values = top_k_df.select("value", "count").to_dicts()
                    profile_data[f"{column_name}_top_k"] = top_k_values

    def _run_correlation_pass(
        self, profile_data: dict[str, Any], profiler_warnings: list[str]
    ) -> None:
        """Executes Pass 4: Correlations"""

        correlation_plan = engine.build_correlation_plan(self._df)

        # Check if we actually have columns to correlate
        # We use collect_schema() to check cheaply
        if len(correlation_plan.collect_schema()) > 1:
            # Adaptive sampling logic

            # We reuse the row count from Pass 1 (cost = 0)
            row_count = profile_data.get("table_row_count", 0)

            # Fetch Data (Sampled or Full)
            if row_count > CORRELATION_SAMPLE_SIZE:
                # Case 1: Big Data -> Sample
                # We use Systematic Sampling
                step_size = max(1, row_count // CORRELATION_SAMPLE_SIZE)

                correlation_df = (
                    # We cast to Float64 to handle potential integer overflow
                    correlation_plan.select(pl.all().cast(pl.Float64))
                    # gather_every() only uses every step_size row in the dataset
                    # which is more robust than using head or tail and closest to
                    # random sampling in streaming mode. High disk I/O, low RAM usage
                    .gather_every(step_size)
                    .collect()
                )
                correlations_sampling_method = f"systematic_sample (~{CORRELATION_SAMPLE_SIZE})"
            else:
                # Case 2: Small Data -> Exact
                correlation_df = correlation_plan.select(pl.all().cast(pl.Float64)).collect()
                correlations_sampling_method = "exact"

            # We drop Nulls to prevent NaN propagation in the matrix
            correlation_df = correlation_df.drop_nulls()

            # Compute Matrices
            if correlation_df.height > 0 and correlation_df.width > 1:
                profile_data["correlations"] = {
                    "pearson": [],
                    "spearman": [],
                    "sampling_method": correlations_sampling_method,
                }

                # We expect RuntimeWarnings (Divide by Zero) when correlating constant columns.
                # This is normal behavior for dirty data, so we suppress the log noise locally.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)

                    # Compute PEARSON (Standard .corr())
                    try:
                        # Format: Add a 'column' column so we know which row is which
                        # Output structure: [{'column': 'age', 'age': 1.0, 'income': 0.8}, ...]
                        pearson_matrix = correlation_df.corr().with_columns(
                            pl.Series(name="column", values=correlation_df.columns)
                        )
                        # Reorder so that "column" appears first and
                        # Clean NaNs (for JSON safety)
                        pearson_matrix = pearson_matrix.select(
                            pl.col("column"), pl.exclude("column").fill_nan(None)
                        )

                        profile_data["correlations"]["pearson"] = self._extract_correlation_pairs(
                            pearson_matrix
                        )
                    except Exception as e:
                        profiler_warnings.append(f"Correlations (pearson) calculation failed: {e}")

                    # Compute SPEARMAN (Ranks -> .corr())
                    try:
                        # Spearman is just Pearson on the ranks
                        spearman_matrix = (
                            correlation_df.select(pl.all().rank())
                            .corr()
                            .with_columns(pl.Series(name="column", values=correlation_df.columns))
                        )

                        spearman_matrix = spearman_matrix.select(
                            pl.col("column"), pl.exclude("column").fill_nan(None)
                        )

                        profile_data["correlations"]["spearman"] = self._extract_correlation_pairs(
                            spearman_matrix
                        )
                    except Exception as e:
                        profiler_warnings.append(f"Correlations (spearman) calculation failed: {e}")

                # Metadata Injection
                profile_data["correlations"]["sampling_method"] = correlations_sampling_method

    def _extract_correlation_pairs(self, matrix_df: pl.DataFrame) -> list[dict[str, Any]]:
        """
        De-duplicates a symmetric correlation matrix into an Edge-List format.
        Drops self-correlations (A-A) and duplicates (A-B, B-A).
        """
        matrix_dicts = matrix_df.to_dicts()
        pairs = []
        seen = set()

        for row in matrix_dicts:
            column_a = row["column"]
            for column_b, score in row.items():
                if column_b == "column" or score is None:
                    continue
                if column_a == column_b:  # Drop diagonal (self-correlation = 1.0)
                    continue

                # Tuple sorting ensures A-B and B-A generate the same signature
                pair_signature = tuple(sorted([column_a, column_b]))

                if pair_signature not in seen:
                    seen.add(pair_signature)
                    pairs.append({"column_a": column_a, "column_b": column_b, "score": score})

        # Sort by correlation strength (absolute value) descending
        pairs.sort(key=lambda x: abs(x["score"]), reverse=True)
        return pairs

    def _build_profile_object(
        self,
        profile_data: dict[str, Any],
        profiling_start_time: float,
        profiler_warnings: list[str],
    ) -> NetraProfile:
        """Maps the engine's flat dictionary output into the Enterprise Data Contract schema."""
        row_count = profile_data.get("table_row_count", 0)
        profiling_end_time = time.time()

        profile = {
            "dataset": {
                "name": self.dataset_name,
                "format": self.dataset_format,
                "row_count": row_count,
            },
            "columns": {},
            "correlations": profile_data.get(
                "correlations", {"pearson": [], "spearman": [], "sampling_method": None}
            ),
            "alerts": [],  # Will be populated by Pass 5
            "_meta": {
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "execution_start_epoch": profiling_start_time,
                "execution_end_epoch": profiling_end_time,
                "engine_time_seconds": round(profiling_end_time - profiling_start_time, 4),
                "profiler_version": __version__,
                "warnings": profiler_warnings,
            },
        }

        # Extract and alphabetize column names
        column_names = sorted(
            {key.removesuffix("_null_count") for key in profile_data if key.endswith("_null_count")}
        )

        for column_name in column_names:
            # Type Stability: Guarantee empty lists instead of missing keys
            column_profile = {
                "data_type": profile_data.get(f"{column_name}_data_type"),
                "null_count": profile_data.get(f"{column_name}_null_count", 0),
                "n_unique": profile_data.get(f"{column_name}_n_unique", 0),
                "histogram": profile_data.get(f"{column_name}_histogram", []),
                "top_k": profile_data.get(f"{column_name}_top_k", []),
            }

            data_type_string = column_profile.get("data_type", "")

            # Inject numeric stats safely
            if is_numeric(data_type_string):
                numeric_metrics = [
                    "min",
                    "max",
                    "mean",
                    "zero_count",
                    "std",
                    "skew",
                    "kurtosis",
                    "p25",
                    "p50",
                    "p75",
                ]
                for metric in numeric_metrics:
                    column_profile[metric] = profile_data.get(f"{column_name}_{metric}")

            # Inject string stats safely
            elif is_string_type(data_type_string):
                string_metrics = ["min_length", "max_length", "mean_length", "min", "max"]
                for metric in string_metrics:
                    column_profile[metric] = profile_data.get(f"{column_name}_{metric}")

            profile["columns"][column_name] = column_profile

        return cast(NetraProfile, profile)

    def _run_diagnostics_pass(self, profile: NetraProfile, profiler_warnings: list[str]) -> None:
        """Executes Pass 5: Diagnostics"""

        diagnostic_engine = DiagnosticEngine(profile)
        alerts = diagnostic_engine.run()

        # Serialize alerts to dicts for JSON output
        profile["alerts"] = [
            {
                "column_name": alert.column_name,
                "type": alert.type,
                "level": alert.level.value,  # Convert Enum to string
                "message": alert.message,
                "value": alert.value,
            }
            for alert in alerts
        ]
