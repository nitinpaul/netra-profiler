import polars as pl


def build_query_plan(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Builds a single, massive Polars expression graph to compute
    summary statistics for all the columns in a single pass.

    Args:
        lf: The input LazyFrame

    Returns:
        A LazyFrame which yields a 1-row DataFrame containing
        all computed statistics when collected.
    """

    expressions: list[pl.Expr] = []

    # 1. Global Computations (Table level)
    # We prefix these with 'table_' to keep the namespace clean.
    expressions.append(pl.len().alias("table_row_count"))

    # 2. Column-Level Computations
    # We iterate over the schema to decide what stats to compute for which type.
    schema = lf.collect_schema()
    for column_name, data_type in schema.items():
        # --- Universal Stats (All Columns) ---
        expressions.extend(
            [
                pl.col(column_name).null_count().alias(f"{column_name}_null_count"),
                # Exact `n_unique` is expensive (requires sorting/hashing).
                # approx_n_unique uses HyperLogLog++, which is O(1) memory and streaming-friendly.
                pl.col(column_name).approx_n_unique().alias(f"{column_name}_n_unique"),
            ]
        )

        # --- Numeric Stats (Integers & Floats) ---
        if data_type.is_numeric():
            expressions.extend(
                [
                    pl.col(column_name).mean().alias(f"{column_name}_mean"),
                    pl.col(column_name).min().alias(f"{column_name}_min"),
                    pl.col(column_name).max().alias(f"{column_name}_max"),
                    pl.col(column_name).std().alias(f"{column_name}_std"),
                ]
            )

        # --- String/Categorical Stats ---
        elif data_type in (pl.String, pl.Categorical):
            pass  # TODO

    # 3. Construct the Query Plan
    return lf.select(expressions)
