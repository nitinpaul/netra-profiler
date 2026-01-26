from typing import Any

import polars as pl

from netra_profiler.engine import build_query_plan


class Profiler:
    """
    The main entry point for Netra-Profiler

    This class manages the lifecycle of a profiling session:
        1. Ingestion (DataFrame -> LazyFrame)
        2. Plan Construction (Delegated to engine)
        3. Execution (Streaming)
    """

    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        """
        Initializes the profiler.

        Args:
            df: A Polars DataFrame (eager) or LazyFrame.
                If a DataFrame is passed, it is converted to LazyFrame to
                ensure all downstream operations use the query optimizer.
        """

        if isinstance(df, pl.DataFrame):
            self._df = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            self._df = df
        else:
            raise TypeError(f"Unsupported type: {type(df)}. Must be pl.DataFrame or pl.LazyFrame")

    def profile(self) -> dict[str, Any]:
        """
        Executes the profiling plan and returns the results.

        Returns:
            A dictionary containing the profile statistics.
        """

        # 1. Build the Query Plan
        # We ask the engine to construct the massive expression graph.
        query_plan = build_query_plan(self._df)

        # 2. Execute with Streaming
        # We add "type: ignore" to silence the false positive from strict type checkers.
        # The argument is valid at runtime.
        result_df = query_plan.collect(engine="streaming")  # type: ignore

        # 3. Serialize Output
        # The result is always a 1-row DataFrame.
        # We extract that single row as a standard Python dictionary.
        return result_df.rows(named=True)[0]
