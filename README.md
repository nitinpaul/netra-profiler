# Netra Profiler

A super-fast, Arrow-native data profiling and quality tool built from the ground up with [Polars](https://github.com/pola-rs/polars). It ships with a beautiful CLI dashboard and an expressive Python API for seamless integration into your data engineering workflows.

## Features

- **Arrow-Native Performance:** Built on Polars foundation to bypass the Python GIL and use zero-copy Apache Arrow memory to profile millions of rows in seconds.
- **100% Mathematical Certainty:** No approximate algorithms (like HyperLogLog). `netra-profiler` computes exact metrics so you can trust your data health metrics implicitly.
- **Comprehensive Profiling:** Automatically extracts scalar statistics (min, max, mean, skew, kurtosis), generates distributions (histograms), identifies Top-K frequent values, and calculates Pearson/Spearman correlation matrices.
- **Built-in Quality Alerts:** The diagnostic engine automatically flags critical data issues like high null percentages, constant columns, and heavy skewness directly in the output.
- **Beautiful Terminal UI:** Includes an information-dense, highly readable CLI dashboard to profile and check your data health directly in the terminal.
- **Complex Type Support:** Automatically flattens nested Structs and computes length statistics for Lists and Arrays, allowing you to profile complex JSON or Parquet files with zero configuration.
- **Python API:** Integrate seamlessly into your data engineering pipelines (Airflow, Jupyter, CI/CD) with a clean, expressive programmatic interface.
- **Export to JSON:** Export the full diagnostic profile directly to JSON (netra run data.csv --json) for CI/CD data quality gates, custom catalogs, or LLM agent context.

## Installation

Install directly from PyPI using your preferred package manager:

```bash
pip install netra-profiler
```

## Quickstart

1. The CLI

The fastest way to profile your data is right from the command line. `netra-profiler` natively supports .csv, .parquet, .json, and .arrow files.

```
netra profile path/to/your/dataset.csv
```
*Tip: Add --json flag to output the raw profile payload instead of the visual dashboard.*

2. The Python API

Integrate netra-profiler directly into your Marimo/Jupyter notebooks, Airflow DAGs, AI Agent contexts or custom data validation pipelines.

```python
import polars as pl
from netra_profiler import Profiler

# 1. Load your data using Polars (Eager or Lazy)
df = pl.read_parquet("sales_data.parquet")

# 2. Initialize the Profiler and run the profiling operation
profiler = Profiler(df)
results = profiler.profile(bins=20, top_k=10)

# 3. Access the profile metrics
print(f"Total Rows Profiled: {results['table_row_count']:,}")
if "revenue_mean" in results:
    print(f"Revenue Mean: ${results['revenue_mean']:.2f}")

# 4. Programmatic Data Quality Gates
# Alerts are pre-computed and categorized by severity (CRITICAL, WARNING, INFO)
alerts = results.get("alerts", [])
critical_issues = [a for a in alerts if a["level"] == "CRITICAL"]

if critical_issues:
    print(f"Pipeline halted: Found {len(critical_issues)} critical data issues!")
    for issue in critical_issues:
        print(f" - [{issue['column']}] {issue['type']}: {issue['message']}")
    raise ValueError("Data quality checks failed.")
```

## Contributing

We welcome contributions! As we scale `netra-profiler` into a comprehensive data observability tool, we are looking for help with test coverage, new file format support, and the upcoming HTML reporting engine.

## License

This software is licensed under the [MIT License](https://github.com/nitinpaul/netra-profiler/blob/main/LICENSE).
