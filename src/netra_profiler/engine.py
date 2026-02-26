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
    # We iterate over the schema to decide what stats to compute for each type.
    schema = lf.collect_schema()
    for column_name, data_type in schema.items():
        # Universal Stats (All Columns)
        expressions.extend(
            [
                pl.col(column_name).null_count().alias(f"{column_name}_null_count"),
                pl.col(column_name).n_unique().alias(f"{column_name}_n_unique"),
            ]
        )

        # Numeric Stats (Integers & Floats)
        if data_type.is_numeric():
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
        elif data_type in (pl.String, pl.Categorical, pl.Enum):
            # We create a string-expression for the column (DRY)
            column_string = pl.col(column_name).cast(pl.String)
            expressions.extend(
                [
                    # 1. Lexicographical stats (First/Last alphabetical value)
                    column_string.min().alias(f"{column_name}_min"),
                    column_string.max().alias(f"{column_name}_max"),
                    # 2. Length stats
                    column_string.str.len_chars().mean().alias(f"{column_name}_mean_length"),
                    column_string.str.len_chars().min().alias(f"{column_name}_min_length"),
                    column_string.str.len_chars().max().alias(f"{column_name}_max_length"),
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
        if data_type.is_numeric():
            # We fetch the raw column data (cast to Float64) instead of calculating
            # the histogram in the Lazy engine. This allows the core orchestrator
            # to execute .hist() in Eager mode, which guarantees that Polars returns
            # the full bin metadata (breakpoints/categories) necessary for plotting,
            # rather than the optimized list of counts returned by the Lazy engine.
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
        if data_type in (pl.String, pl.Categorical, pl.Enum):
            # Logic: GroupBy -> Count -> Sort -> Head(k)
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


def build_correlation_plan(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    PASS 4: Correlation Data Fetcher.

    Identifies and selects all numeric columns.
    This plan is intended to be collected (potentially with sampling)
    by the core module to compute the correlation matrix.
    """
    # 1. Identify Numeric Columns
    # We use the schema without scanning data
    schema = lf.collect_schema()
    numeric_columns = []

    for column_name, data_type in schema.items():
        if data_type.is_numeric():
            numeric_columns.append(column_name)

    # 2. Return a plan selecting only these columns
    if not numeric_columns:
        # Return an empty plan if no numeric columns exist
        return pl.LazyFrame({})

    return lf.select(numeric_columns)


def preprocess_complex_types(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transforms complex types (Structs, Lists, Arrays) into
    profilable scalar columns.

    Strategy:
    1. Structs: Flattened into 'parent_child' columns.
    2. Lists/Arrays: Converted to '_len' integer columns.
    """
    schema = lf.collect_schema()

    # We will build a list of expressions to select/transform
    expressions = []

    for column_name, data_type in schema.items():
        # 1. Handle Structs (Flattening)
        if isinstance(data_type, pl.Struct):
            # We explicitly alias fields to prevent naming collisions
            # e.g. "user" -> "user_name", "user_age"
            struct_fields = data_type.fields
            for field in struct_fields:
                expressions.append(
                    pl.col(column_name)
                    .struct.field(field.name)
                    .alias(f"{column_name}_{field.name}")
                )

        # 2. Handle Lists & Arrays (Length Stats)
        elif isinstance(data_type, (pl.List, pl.Array)):
            # We replace the original column with length stats
            expressions.append(pl.col(column_name).list.len().alias(f"{column_name}_len"))

        # 3. Pass through everything else (Scalars)
        else:
            expressions.append(pl.col(column_name))

    return lf.select(expressions)
