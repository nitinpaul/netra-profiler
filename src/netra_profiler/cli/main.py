import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl
import typer

from netra_profiler import Profiler, __version__
from netra_profiler.cli.console import NetraCLIRenderer, console


def _get_peak_ram_usage_in_mb() -> float:
    """Returns the peak RAM usage (High-Water Mark) of the process in MB."""
    if sys.platform == "win32":
        import psutil  # Deferred Windows-only import # noqa: PLC0415

        process = psutil.Process(os.getpid())
        # Windows gives us the exact Peak Working Set in bytes
        return process.memory_info().peak_wset / (1024 * 1024)
    else:
        import resource  # Deferred Unix-only import # noqa: PLC0415

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            # macOS ru_maxrss is in bytes
            return rusage.ru_maxrss / (1024 * 1024)
        else:
            # Linux ru_maxrss is in kilobytes
            return rusage.ru_maxrss / 1024


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"netra-profiler v{__version__}")
        raise typer.Exit()


def _format_bytes(size: int) -> str:
    """Converts raw bytes to human readable string (e.g. 6.9 GB)."""
    power = 1024.0
    n: float = float(size)

    labels = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    max_index = len(labels) - 1
    count = 0

    while n >= power and count < max_index:
        n /= power
        count += 1

    return f"{n:.2f} {labels[count]}"


def _detect_csv_separator(path: Path) -> str:
    """Peeks at the first chunk of a CSV to auto-detect the separator."""
    try:
        with open(path, encoding="utf-8") as file:
            sample = "".join([file.readline() for _ in range(5)])
            # We restrict the sniffer to common data delimiters to prevent false positives
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
    except Exception:
        # If sniffing fails (e.g., weird encoding or 1-column file), fallback to standard comma
        return ","


def _scan_file(path: Path, full_inference: bool = False) -> tuple[pl.LazyFrame, str]:
    """
    Creates a LazyFrame based on file extension.
    Returns (LazyFrame, file_type_label).
    """
    extension = path.suffix.lower()

    if extension == ".csv":
        # We pass the detected separator explicitly to Polars
        separator = _detect_csv_separator(path)

        infer_schema_length = None if full_inference else 10000
        return pl.scan_csv(
            path, separator=separator, infer_schema_length=infer_schema_length
        ), f"CSV (separator: '{separator}')"
    elif extension == ".parquet":
        return pl.scan_parquet(path), "Parquet"
    elif extension in {".ipc", ".arrow"}:
        return pl.scan_ipc(path), "IPC/Arrow"
    elif extension == ".json":
        try:
            return pl.scan_ndjson(path), "JSON (Newline)"
        except Exception:
            # Fallback for standard JSON array
            return pl.read_json(path).lazy(), "JSON (Standard)"
    else:
        raise ValueError(f"Unsupported extension: {extension}")


app = typer.Typer(
    name="netra",
    help="Netra Profiler: High-performance data profiling and quality tool.",
    add_completion=False,
)


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    pass


@app.command()
def profile(
    file_path: str = typer.Argument(..., help="Path to the dataset (CSV, Parquet, JSON, IPC)."),
    json_output: bool = typer.Option(
        False, "--json", help="Output raw JSON to stdout (silences UI)."
    ),
    bins: int = typer.Option(20, "--bins", min=1, help="Number of histogram bins."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Number of frequent items to show."),
    full_inference: bool = typer.Option(
        False, "--full-inference", help="Force full-file schema inference for messy CSVs."
    ),
) -> None:
    """
    Profile the connected data source and generate the CLI report.
    """
    path = Path(file_path)

    # --- MODE 1: JSON OUTPUT ---
    if json_output:
        try:
            df, _ = _scan_file(path)
            profiler = Profiler(df)
            profile = profiler.run(bins=bins, top_k=top_k)
            print(json.dumps(profile, default=str))
        except Exception as e:
            # Output error as JSON for machine parsing
            print(json.dumps({"error": str(e)}))
            raise typer.Exit(code=1) from None
        return

    # --- MODE 2: CLI OUTPUT ---

    with NetraCLIRenderer() as ui:
        if not path.exists():
            ui.render_data_source_spinner(path.name)
            ui.render_fatal_error(
                step="data_source",
                message="File not found on disk.",
                hint=f"Verify the path: {path}",
            )
            raise typer.Exit(code=1)

        # --- Phase 1: Data Source Connection ---
        try:
            # We pass str(path) so it shows the relative path in the UI
            ui.render_data_source_spinner(path.name)

            # Logic: Scan and Detect
            load_start = time.time()
            df, file_type = _scan_file(path, full_inference=full_inference)
            file_size = os.path.getsize(path)

            schema = df.collect_schema()
            load_time = time.time() - load_start

            datatypes = [str(t) for t in schema.values()]
            datatype_counts = Counter(datatypes)
            schema_info = ", ".join([f"{v} {k}" for k, v in datatype_counts.items()])

            file_info = {"path": str(path), "size": _format_bytes(file_size), "type": file_type}

            ui.render_data_source_panel(
                file_info=file_info,
                schema_info=schema_info,
                columns=len(schema),
                time_taken=load_time,
            )

        except Exception as e:
            # If loading fails, we transition to error state and exit cleanly

            raw_error = str(e)
            # Strip out Polars' internal Python suggestions
            clean_error = raw_error.split("You might want to try:")[0].strip()

            ui.render_fatal_error(
                step="data_source",
                message=f"Connection failed to data source: {path}",
                hint=clean_error,
            )
            raise typer.Exit(code=1) from None

        # --- Phase 2: Data Profiling & Engine Telemetry ---
        try:
            progress = ui.render_engine_status_panel()
            engine_messages = [
                "Resolving lazy execution graph...",
                "Allocating zero-copy memory buffers...",
                "Vectorizing data streams...",
                "Threading execution pipelines...",
                "Collapsing data dimensions...",
                "Calibrating Apache Arrow matrices...",
                "Synthesizing profile topology...",
            ]
            active_message = random.choice(engine_messages)
            progress.add_task(active_message, total=None)

            profiler = Profiler(df, dataset_name=path.name, dataset_format=file_type)
            profile = profiler.run(bins=bins, top_k=top_k)

            engine_time = profile["_meta"]["engine_time_seconds"]
            peak_ram_usage = _get_peak_ram_usage_in_mb()

            throughput = (file_size / engine_time) / (1024**3) if engine_time > 0 else 0.0

            ui.render_engine_telemetry_panel(
                engine_time=engine_time,
                throughput_gb_s=throughput,
                peak_ram_usage=peak_ram_usage,
            )

        except Exception as e:
            raw_error = str(e)
            # Strip out Polars' internal Python suggestions
            clean_error = raw_error.split("You might want to try:")[0].strip()

            # If it's a parsing/type error, give the user the exact solution
            if "parse" in raw_error or "primitive" in raw_error:
                message = "Dataset contains mixed types (Schema Drift)."
                hint = (
                    f"{clean_error}\n\n[brand]Tip:[/] Run with [bold]--full-inference[/bold] "
                    "to force full-file schema inference."
                )
            else:
                message = "Engine panicked during profiling."
                hint = clean_error

            ui.render_fatal_error(
                step="profiling",
                message=message,
                hint=hint,
            )
            raise typer.Exit(code=1) from None

        # --- Phase 3: Results Dashboard ---
        ui.render_profiling_results(profile)


@app.command()
def info() -> None:
    """Prints environment info for debugging."""
    console.print(f"Netra Version: {__version__}")
    console.print(f"Polars Version: {pl.__version__}")
    console.print(f"Python Version: {sys.version.split()[0]}")


if __name__ == "__main__":
    app()
