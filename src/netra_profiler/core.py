import time
from typing import Any

import polars as pl

from netra_profiler import engine

from . import __version__


class Profiler:
    """
    The main entry point for Netra-Profiler

    This class manages the lifecycle of a profiling session:
        1. Ingestion (DataFrame -> LazyFrame)
        2. Plan Construction (Delegated to engine)
        3. Execution (Three-Pass Strategy)
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

    def profile(self, bins: int = 20, top_k: int = 10) -> dict[str, Any]:
        """
        Executes the profiling plan and returns the results.

        This uses a "Three-Pass" strategy to ensure stability:
        1. Scalars (Min, Max, Mean) - Fast, robust.
        2. Histograms (Distributions) - Can fail without crashing the report.
        3. Top-K (Frequent Items) - Memory intensive, isolated.

        Args:
            bins: Number of bins for histograms (default: 20)
            top_k: Number of most frequent items to return for text columns (default: 10)

        Returns:
            A dictionary containing the combined statistics and
            a '_meta' key with execution details and warnings.
        """
        start_time = time.time()
        warnings: list[str] = []

        # PASS 1: Scalar Statistics
        try:
            scalar_plan = engine.build_scalar_plan(self._df)
            scalar_df = scalar_plan.collect(engine="streaming")
            profile = scalar_df.rows(named=True)[0]
        except Exception as e:
            # If Pass 1 fails, we can't really do anything.
            raise RuntimeError(f"Critical Error: {e}") from e

        # PASS 2: Histograms
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
                            warnings.append(
                                f"Histogram generation failed for column '{column_name}': {e}"
                            )

        except Exception as e:
            warnings.append(f"Histogram generation failed: {e}")

        # PASS 3: Top-K Values
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
            warnings.append(f"Top-K generation failed: {e}")

        # Metadata Injection
        execution_time = time.time() - start_time

        profile["_meta"] = {
            "timestamp": time.time(),
            "execution_time": round(execution_time, 4),
            "warnings": warnings,
            "version": __version__,
        }

        return profile
