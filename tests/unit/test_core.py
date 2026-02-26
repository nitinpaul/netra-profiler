from typing import Any

import polars as pl
import pytest

from netra_profiler import Profiler


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def profile_result(sample_df: pl.DataFrame) -> dict[str, Any]:
    """
    Runs the profiler ONCE and returns the result.
    This fixture is shared by all tests below.
    """
    profiler = Profiler(sample_df)
    profile = profiler.profile()

    # DEBUG: Print warnings if they exist
    if profile["_meta"]["warnings"]:
        print("\n!!! WARNINGS FOUND !!!")
        for warning in profile["_meta"]["warnings"]:
            print(f"- {warning}")

    return profile


def test_global_stats(profile_result: dict[str, Any], sample_df: pl.DataFrame) -> None:
    """Verifies table-level statistics."""
    assert profile_result["table_row_count"] == sample_df.height


def test_numeric_stats(profile_result: dict[str, Any], sample_df: pl.DataFrame) -> None:
    """Verifies numeric column statistics (Age)."""
    assert profile_result["age_min"] == sample_df["age"].min()
    assert profile_result["age_max"] == sample_df["age"].max()
    assert profile_result["age_null_count"] == sample_df["age"].null_count()

    # We use pytest.approx for floating point comparisons to avoid precision errors
    assert profile_result["age_mean"] == pytest.approx(sample_df["age"].mean())
    assert profile_result["age_std"] == pytest.approx(sample_df["age"].std())

    assert profile_result["age_p25"] == sample_df["age"].quantile(0.25)
    assert profile_result["age_p50"] == sample_df["age"].median()
    assert profile_result["age_p75"] == sample_df["age"].quantile(0.75)


def test_string_stats(profile_result: dict[str, Any], sample_df: pl.DataFrame) -> None:
    """Verifies string column statistics (City)."""
    assert profile_result["city_null_count"] == sample_df["city"].null_count()
    assert "city_n_unique" in profile_result

    # Lexicographical min/max
    # "Delhi" should be min alphabetical
    # "Thrissur" should be max alphabetical
    assert profile_result["city_min"] == "Delhi"
    assert profile_result["city_max"] == "Thrissur"

    # String Lengths
    # Groningen (9), Thrissur (8), Delhi (5)
    city_lengths = sample_df["city"].str.len_chars()
    assert profile_result["city_min_length"] == city_lengths.min()  # 5 (Delhi)
    assert profile_result["city_max_length"] == city_lengths.max()  # 9 (Groningen)
    assert profile_result["city_mean_length"] == pytest.approx(city_lengths.mean())


def test_top_k_stats(profile_result: dict[str, Any], sample_df: pl.DataFrame) -> None:
    """Verifies Top-K frequent item generation."""
    assert "city_top_k" in profile_result
    city_top_k = profile_result["city_top_k"]

    # Check Top-K Structure (City)
    # Expected: A list of dicts like [{'value': 'Groningen', 'count': 2}, ...]
    assert isinstance(city_top_k, list)
    assert len(city_top_k) > 0

    # Verify counts for the most frequent item
    most_frequent_city = city_top_k[0]["value"]
    expected_count = sample_df.filter(pl.col("city") == most_frequent_city).height
    assert city_top_k[0]["count"] == expected_count


def test_histogram_stats(profile_result: dict[str, Any]) -> None:
    """Verifies Histogram generation."""
    assert "age_histogram" in profile_result
    age_histogram = profile_result["age_histogram"]

    assert isinstance(age_histogram, list)
    assert len(age_histogram) > 0

    # Verify Struct keys
    # # Polars Histogram Struct keys: 'breakpoint', 'category', 'count'
    # The 'category' key holds the range string e.g. "(25, 30]"
    first_bin = age_histogram[0]
    assert "breakpoint" in first_bin
    assert "count" in first_bin


def test_correlation_stats(profile_result: dict[str, Any]) -> None:
    """Verifies Correlation Matrix generation."""
    assert "correlations" in profile_result
    correlations = profile_result["correlations"]

    # 1. Pearson
    assert "pearson" in correlations
    pearson_data = correlations["pearson"]
    assert isinstance(pearson_data, list)
    assert len(pearson_data) > 0
    assert "column" in pearson_data[0]

    # Verify self-correlation is 1.0
    age_row = next(row for row in pearson_data if row["column"] == "age")
    assert age_row["age"] == 1.0

    # 2. Spearman
    assert "spearman" in correlations
    assert len(correlations["spearman"]) > 0

    # 3. Method Metadata
    assert profile_result["_meta"]["correlation_method"] == "exact"


def test_metadata(profile_result: dict[str, Any]) -> None:
    """Verifies execution metadata."""
    assert "_meta" in profile_result
    meta = profile_result["_meta"]

    assert "engine_time" in meta
    assert meta["engine_time"] > 0
    assert isinstance(meta["warnings"], list)
    assert len(meta["warnings"]) == 0


def test_alerts(profile_result: dict[str, Any]) -> None:
    """Verifies that the diagnostic engine generates alerts."""
    assert "alerts" in profile_result
    alerts = profile_result["alerts"]
    assert isinstance(alerts, list)
