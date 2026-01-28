import polars as pl


def build_scalar_plan(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    PASS 1: Scalar Statistics (Fast, Streaming-Friendly).
    Computes single-value stats: Mean, Min, Max, Quantiles, Nulls.

    Args:
        lf: The input LazyFrame

    Returns:
        A LazyFrame which yields a 1-row DataFrame containing
        all computed scalar statistics when collected.
    """

    expressions: list[pl.Expr] = []

    # 1. Global Computations (Table level)
    # We prefix these with 'table_' to keep the namespace clean.
    expressions.append(pl.len().alias("table_row_count"))

    # 2. Column-Level Computations
    # We iterate over the schema to decide what stats to compute for which type.
    schema = lf.collect_schema()
    for column_name, data_type in schema.items():
        # Universal Stats (All Columns)
        expressions.extend(
            [
                pl.col(column_name).null_count().alias(f"{column_name}_null_count"),
                # Exact `n_unique` is expensive (requires sorting/hashing).
                # approx_n_unique uses HyperLogLog++, which is O(1) memory and streaming-friendly.
                pl.col(column_name).approx_n_unique().alias(f"{column_name}_n_unique"),
            ]
        )

        # Numeric Stats (Integers & Floats)
        if data_type in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.Float32,
            pl.Float64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            expressions.extend(
                [
                    # Basic
                    pl.col(column_name).mean().alias(f"{column_name}_mean"),
                    pl.col(column_name).min().alias(f"{column_name}_min"),
                    pl.col(column_name).max().alias(f"{column_name}_max"),
                    # Distribution Stats
                    pl.col(column_name).std().alias(f"{column_name}_std"),
                    pl.col(column_name).skew().alias(f"{column_name}_skew"),
                    pl.col(column_name).kurtosis().alias(f"{column_name}_kurtosis"),
                    # Quantiles (Percentiles)
                    pl.col(column_name).quantile(0.25).alias(f"{column_name}_p25"),
                    pl.col(column_name).median().alias(f"{column_name}_p50"),
                    pl.col(column_name).quantile(0.75).alias(f"{column_name}_p75"),
                ]
            )

        # String/Categorical Stats
        elif data_type in (pl.String, pl.Categorical):
            # We create a string-expression for the column (DRY)
            # If it's categorical, we cast to String to access .str namespace methods
            column_as_string = pl.col(column_name).cast(pl.String)

            expressions.extend(
                [
                    # 1. Lexicographical stats (First/Last alphabetical value)
                    column_as_string.min().alias(f"{column_name}_min"),
                    column_as_string.max().alias(f"{column_name}_max"),
                    # 2. Length stats
                    column_as_string.str.len_chars().mean().alias(f"{column_name}_len_mean"),
                    column_as_string.str.len_chars().min().alias(f"{column_name}_len_min"),
                    column_as_string.str.len_chars().max().alias(f"{column_name}_len_max"),
                ]
            )

    # 3. Construct the Query Plan
    return lf.select(expressions)


def build_histogram_plans(lf: pl.LazyFrame) -> list[pl.LazyFrame]:
    """
    PASS 2: Histogram Data Fetcher.
    Creates a list of execution plans to fetch raw numeric column data.

    Args:
        lf: The input LazyFrame.

    Returns:
        A list of LazyFrames, where each plan selects a single numeric column.
    """

    plans: list[pl.LazyFrame] = []
    schema = lf.collect_schema()

    for column_name, data_type in schema.items():
        if data_type in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.Float32,
            pl.Float64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            # We fetch the raw column data (cast to Float64) instead of calculating
            # the histogram in the Lazy engine. This allows the core orchestrator
            # to execute .hist() in Eager mode, which guarantees that Polars returns
            # the full bin metadata (breakpoints/categories) necessary for plotting,
            # rather than the optimized list of counts often returned by the Lazy engine.
            plan = lf.select(pl.col(column_name).cast(pl.Float64))
            plans.append(plan)

    return plans


def build_top_k_plan(lf: pl.LazyFrame, k: int = 10) -> list[pl.LazyFrame]:
    """
    PASS 3: Top-K Values (Most Frequent Items).
    Computes the top 10 most frequent values for all string/categorical columns.

    Args:
        lf: The input LazyFrame

    Returns:
        A list of LazyFrames. Each LazyFrame corresponds to a specific column
        and yields a small table with 'value' and 'count' columns.
    """
    plans: list[pl.LazyFrame] = []
    schema = lf.collect_schema()

    for column_name, data_type in schema.items():
        if data_type in (pl.String, pl.Categorical):
            # Logic: GroupBy -> Count -> Sort -> Head(10)
            # We explicitly cast to String to handle Categoricals safely
            plan = (
                lf.select(pl.col(column_name).cast(pl.String).alias("value"))
                .group_by("value")
                .len()  # counts the group size
                .sort("len", descending=True)
                .head(k)
                .select(
                    pl.lit(column_name).alias("column_name"),
                    pl.col("value"),
                    pl.col("len").alias("count"),
                )
            )
            plans.append(plan)

    return plans
