import polars as pl
import pytest

from netra_profiler import Profiler


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


def test_profiler_basic_stats(sample_df: pl.DataFrame) -> None:
    """
    Verifies that the profiler returns the correct statistical values.
    """
    # Initialize and run
    profiler = Profiler(sample_df)
    profile = profiler.profile()

    # Debug: Print the report if the test fails so we can see what happened
    print(f"\nGenerared Report: {profile}")

    # 1. Verify Global Stats
    assert profile["table_row_count"] == sample_df.height

    # 2. Verify Numeric Stats (Age)
    assert profile["age_min"] == sample_df["age"].min()
    assert profile["age_max"] == sample_df["age"].max()
    assert profile["age_null_count"] == sample_df["age"].null_count()
    # We use pytest.approx for floating point comparisons to avoid precision errors
    assert profile["age_mean"] == pytest.approx(sample_df["age"].mean())
    assert profile["age_std"] == pytest.approx(sample_df["age"].std())
    assert profile["age_p25"] == sample_df["age"].quantile(0.25)
    assert profile["age_p50"] == sample_df["age"].median()
    assert profile["age_p75"] == sample_df["age"].quantile(0.75)

    # 3. Verify String Stats (City)
    assert profile["city_null_count"] == sample_df["city"].null_count()
    # "Groningen" appears twice, so approx_n_unique should be 3 (Groningen, Thrissur, Delhi)
    assert "city_n_unique" in profile

    # "Delhi" should be min alphabetical
    # "Thrissur" should be max alphabetical
    assert profile["city_min"] == "Delhi"
    assert profile["city_max"] == "Thrissur"

    # Lengths:
    # Groningen (9), Thrissur (8), Delhi (5).
    city_lengths = sample_df["city"].str.len_chars()

    assert profile["city_len_min"] == city_lengths.min()  # 5 (Delhi)
    assert profile["city_len_max"] == city_lengths.max()  # 9 (Groningen)
    assert profile["city_len_mean"] == pytest.approx(city_lengths.mean())

    # 4. Verify Distributions
    # Check Top-K Structure (City)
    # Expected: A list of dicts like [{'value': 'Groningen', 'count': 2}, ...]
    assert "city_top_k" in profile
    city_top_k = profile["city_top_k"]
    assert isinstance(city_top_k, list)
    assert len(city_top_k) > 0

    most_frequent_city = city_top_k[0]["value"]
    expected_count = sample_df.filter(pl.col("city") == most_frequent_city).height

    assert city_top_k[0]["count"] == expected_count

    # Check Histogram Structure (Age)
    assert "age_histogram" in profile
    age_histogram = profile["age_histogram"]

    # Verify it is a List of Dictionaries (from Structs)
    assert isinstance(age_histogram, list)
    assert len(age_histogram) > 0

    # Polars Histogram Struct keys: 'breakpoint', 'category', 'count'
    # The 'category' key holds the range string e.g. "(25, 30]"
    first_bin = age_histogram[0]
    assert "breakpoint" in first_bin
    assert "count" in first_bin

    # 5. Verify Metadata
    assert "_meta" in profile
    meta = profile["_meta"]
    assert "execution_time" in meta
    assert meta["execution_time"] > 0
    assert isinstance(meta["warnings"], list)
    # We ensure our clean run has no warnings
    assert len(meta["warnings"]) == 0
