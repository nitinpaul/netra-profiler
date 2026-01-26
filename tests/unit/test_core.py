import polars as pl
import pytest

from netra_profiler.core import Profiler


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """
    Creates a synthetic DataFrame with mixed types
    and edge cases(nulls, duplicates, negative numbers).
    """
    return pl.DataFrame(
        {
            "age": [25, 30, 35, None, 25],  # Numeric with nulls & duplicates
            "salary": [50000.0, 60000.0, 75000.0, 50000.0, None],  # Float with nulls
            "city": ["Groningen", "Thrissur", "Delhi", None, "Groningen"],  # String
        }
    )


def test_profiler_basic_stats(sample_df: pl.DataFrame):
    """
    Verifies that the profiler returns the correct statistical values.
    """
    # Initialize and run
    profiler = Profiler(sample_df)
    profile = profiler.profile()

    # Debug: Print the report if the test fails so we can see what happened
    print(f"\nGenerared Report: {profile}")

    # 1. Check Global Stats
    assert profile["table_row_count"] == sample_df.height

    # 2. Check Numeric Stats (Age)
    # Note: Polars mean([25, 30, 35, 25]) = 28.75. Nulls are ignored.
    assert profile["age_min"] == sample_df["age"].min()
    assert profile["age_max"] == sample_df["age"].max()
    assert profile["age_null_count"] == sample_df["age"].null_count()
    # We use pytest.approx for floating point comparisons to avoid precision errors
    assert profile["age_mean"] == pytest.approx(sample_df["age"].mean())

    # 3. Check String Stats (City)
    assert profile["city_null_count"] == 1
    # "Groningen" appears twice, so approx_n_unique should be 3 (Groningen, Thrissur, Delhi)
    # Note: approx_n_unique is approximate, on tiny data, it might behave strictly or loosely.
    # For small integers/strings, Polars often exact-counts, but we just check it exists for now.
    assert "city_n_unique" in profile
