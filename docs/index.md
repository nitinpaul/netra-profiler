# Netra Profiler

**High-performance profiling and data quality tool built with Polars.**

[![PyPI version](https://img.shields.io/pypi/v/netra-profiler.svg)](https://pypi.org/project/netra-profiler/)
[![Python versions](https://img.shields.io/pypi/pyversions/netra-profiler.svg)](https://pypi.org/project/netra-profiler/)
[![Polars Native](https://img.shields.io/badge/Polars-Native-blue?logo=polars)](https://pola.rs/)

Netra Profiler is a next-generation data profiling tool and diagnostic engine built on top of **Polars**. Designed to operate at the speed of your disk I/O, it leverages Polars' Rust-based query optimizer and zero-copy Apache Arrow memory model to process out-of-core enterprise-scale workloads much larger than your RAM size.

The profiler ships with a comprehensive diagnostic engine to detect column-wise data quality issues early in your analysis or modeling workflows, such as high zeros/null count, high cardinality, data skew and more. The tool includes a detailed, zero-configuration CLI for quickly profiling your CSV, JSON, Arrow/IPC or Parquet data files.

---

## Performance

*Preliminary benchmarks run on local hardware. Large-scale benchmarks (100GB+) are in development.*

**Test Environment**

* **CPU:** Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (12 Cores)
* **RAM:** 32 GB
* **OS:** Ubuntu 24.04.4 LTS
* **Storage:** 512GB NVMe SSD

| Shape | Rows | Columns | File Size | Execution Time | Peak RAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 10,000,000 | 10 | ~1.1 GB | *Pending...* | *Pending...* |
| **Deep** | 100,000,000 | 10 | ~11.6 GB | *Pending...* | *Pending...* |
| **Wide** | 10,000,000 | 100 | ~11.6 GB | *Pending...* | *Pending...* |

*Note: Execution time measures the core engine processing time (excluding initial disk I/O scan and schema inference).*

---

## Features

<div class="grid cards" markdown>

-   :material-engine-outline: **Streaming-First Architecture**
    
    The profiler never pulls your entire dataset into main memory. Using systematic sampling and I/O-level slice pushdowns, it calculates statistical distributions and correlation matrices with a bounded memory footprint.

-   :material-file-document-outline: **Enterprise Data Contract**
    
    Downstream tools shouldn't have to guess your output format. `netra-profiler` generates a strictly typed, version-controlled JSON schema (`NetraProfile`) ready for ingestion by your data catalogs, data warehouses, and CI/CD pipelines.

-   :material-alert-decagram-outline: **Built-in Quality Checks**
    
    It doesn't just give you numbers; it gives you insights. The built-in Diagnostic Engine automatically detects schema drift, high nullity, zero-variance columns, and skewed distributions, generating actionable alerts.

-   :material-console: **Beautiful Terminal UI**
    
    Powered by `rich`, the CLI provides a dynamic and detailed dashboard, complete with hardware telemetry, statistical overview, preview of data distribution, and a prioritized Data Health summary.

</div>

---

## Getting Started

Head over to the [quickstart](getting_started/quickstart.md) section to start profiling your data now, with zero configurations.
