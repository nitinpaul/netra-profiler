# Netra-Profiler

**A Polars-based, high-performance data profiler for the AI era.**

> 🚧 **Status: Pre-Alpha / Active Development**
> This project is currently under heavy construction. The API is unstable.

## Mission
Netra-Profiler aims to replace legacy profiling tools by leveraging the streaming, out-of-core capabilities of [Polars](https://pola.rs). It is designed to profile terabyte-scale datasets on a single node without running out of memory.

## Architecture
- **Engine:** Polars LazyFrame (Streaming Mode)
- **Stack:** Python 3.10+, UV, Ruff, Jinja2
- **License:** MIT

## Roadmap
- [ ] Core Statistical Engine (Streaming)
- [ ] HTML Reporting (Plotly/Alpine.js)
- [ ] Rust Plugins for PII & Advanced Stats
