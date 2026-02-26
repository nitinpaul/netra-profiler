from types import TracebackType
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from netra_profiler import __version__

from .theme import NETRA_THEME

LATENCY_THRESHOLD = 0.01
EXECUTION_TIME_THRESHOLD = 0.01
TOP_K_STRING_LENGTH = 12

console = Console(theme=NETRA_THEME)


class NetraCLIRenderer:
    """
    The UI Engine for Netra Profiler CLI.

    Operates on a 'Stacked Card' layout philosophy within a live terminal display.
    The lifecycle moves through three distinct phases:
    1. Data Source Connection (Ingestion & Schema inference)
    2. Engine Status (Polars lazy execution & Telemetry display)
    3. Results Dashboard (Health Checks & Variable Explorer)
    """

    def __init__(self) -> None:
        # UI State Components
        # We track the active file to dynamically update the main outer border
        self._data_source_name: str | None = None

        # Phase 1: The Data Source Connection Card
        self._data_source_card: Panel | None = None

        # Phase 2: The Engine Status & Telemetry Card
        self._engine_progress_bar: Progress | None = None
        self._engine_status_card: Panel | None = None
        self._engine_telemetry_card: Panel | None = None

        # Phase 3: The Final Results Group (Health + Variables)
        self._profiling_results: Group | None = None

        # The Live Context
        self.live = Live(
            self._build_layout(),
            console=console,
            refresh_per_second=10,
            transient=False,  # Retain the final dashboard on the terminal after exit
        )

    def _build_layout(self) -> Panel:
        """
        Dynamically builds the UI hierarchy.
        Only active (non-None) cards are stacked into the rendering group.
        """
        active_cards: list[RenderableType] = []

        if self._data_source_card:
            active_cards.append(self._data_source_card)

        if self._engine_status_card:
            active_cards.append(self._engine_status_card)
        if self._engine_telemetry_card:
            active_cards.append(self._engine_telemetry_card)

        if self._profiling_results:
            active_cards.append(self._profiling_results)

        # Main Outer Wrapper Title: Show filename if available, else show init state
        panel_title = (
            f"[not dim][brand]{self._data_source_name}[/brand][/]"
            if self._data_source_name
            else "[muted]Initializing...[/muted]"
        )

        # Wrap the active stacked cards in the main border
        return Panel(
            Group(*active_cards),
            title=panel_title,
            subtitle=f"[muted]Netra Profiler v{__version__}[/muted]",
            subtitle_align="right",
            border_style="border.main",
            box=box.ROUNDED,
            padding=(0, 1),  # top and bottom padding = 1
        )

    def _refresh(self) -> None:
        """Pushes the updated layout to the Live display."""
        self.live.update(self._build_layout())

    def __enter__(self) -> "NetraCLIRenderer":
        self.live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.live.stop()

    def render_fatal_error(self, step: str, message: str, hint: str = "") -> None:
        """
        Renders a semantic red error card and halts the visual progression.
        Slots the error into the position of whichever card failed.

        step: "data_source" or "profiling"
        """
        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        grid.add_row(f"[critical]{message}[/critical]")
        if hint:
            grid.add_row(f"└─ [not dim][value]{hint}[/value][/]")

        step_title = step.replace("_", " ").title()
        title_string = f"[not dim][critical]✖[/critical] [value]{step_title}[/value][/]"

        error_panel = Panel(
            grid,
            title=title_string,
            title_align="left",
            border_style="critical",
            box=box.ROUNDED,
            padding=(1, 1),
        )

        # Slot the error into the correct active card
        if step == "data_source":
            self._data_source_card = error_panel
        elif step == "profiling":
            self._engine_telemetry_card = error_panel

        self._refresh()

    # --- Phase 1: Data Source Connection ---

    def render_data_source_spinner(self, data_source_name: str) -> None:
        """Initializes the UI with a spinner while loading the data file."""
        self._data_source_name = data_source_name

        grid = Table.grid(expand=True)
        grid.add_column()

        spinner = Spinner(
            name="arrow3", text=Text("Connecting to data stream...", style="muted"), style="brand"
        )
        # Optical alignment to ensure the spinner is flush with the success checkmark
        aligned_spinner = Padding(spinner, (0, 0, 0, 1))

        grid.add_row(aligned_spinner)

        self._data_source_card = Panel(
            grid,
            title="[not dim][success]⛁[/success] [value]Data Source[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()

    def render_data_source_card(
        self,
        file_info: dict[str, str],
        schema_info: str,
        columns: int,
        time_taken: float,
    ) -> None:
        """Transforms the active loader into a permanent Data Source metadata card."""
        # 1. Latency Formatting
        latency_string = "< 0.01s" if time_taken < LATENCY_THRESHOLD else f"{time_taken:.2f}s"

        # Extract packaged file info
        file_path = file_info.get("path", "Unknown")
        file_size = file_info.get("size", "0 B")
        file_type = file_info.get("type", "Unknown")

        # 2. Vertical Tree Layout
        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        success_msg = (
            " [not dim][success]✔[/success][/] "
            "[not dim][success]Connected to data source:[/success][/] "
            f"[value]{file_path}[/value]"
        )
        grid.add_row(success_msg)
        grid.add_row(f" └─ Latency: [not dim][brand]{latency_string}[/brand][/]")
        grid.add_row(f" └─ Format:  {file_type}")
        grid.add_row(f" └─ Size:    {file_size}")
        grid.add_row(f" └─ Schema:  {columns} Columns ({schema_info})")

        self._data_source_card = Panel(
            grid,
            title="[not dim][success]⛁[/success] [value]Data Source[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()

    # --- Phase 2: Data Profiling & Engine Telemetry ---

    def render_engine_status_card(self) -> Progress:
        """Initializes the indeterminate 'Cylon scanner' progress bar for the engine."""
        self._engine_progress_bar = Progress(
            SpinnerColumn(style="brand"),
            TextColumn("[brand]{task.description}"),
            BarColumn(
                bar_width=None, style="#00FFFF", complete_style="#00FFFF", pulse_style="#FFFF00"
            ),
            expand=True,
        )

        self._engine_status_card = Panel(
            self._engine_progress_bar,
            title="[not dim][#FFDF00]⚙[/#FFDF00] [value]Engine Status[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()
        return self._engine_progress_bar

    def render_engine_telemetry_card(
        self, engine_time: float, throughput_gb_s: float, peak_ram_usage: float
    ) -> None:
        """Morphs the active engine status into a permanent hardware telemetry card."""
        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        engine_time_string = (
            "< 0.01s" if engine_time < EXECUTION_TIME_THRESHOLD else f"{engine_time:.2f}s"
        )

        grid.add_row(f" Engine Time:  [not dim][brand]{engine_time_string}[/brand][/]")
        grid.add_row(f" Data Throughput: [not dim][brand]{throughput_gb_s:.2f} GB/s[/brand][/]")
        grid.add_row(
            f" Peak RAM Usage:  [not dim][brand]{peak_ram_usage:.1f} MB[/brand][/] "
            "[muted](Includes Arrow allocation & compute buffers)[/muted]"
        )

        self._engine_telemetry_card = Panel(
            grid,
            title="[not dim][#FFDF00]⏣[/#FFDF00] [value]Engine Telemetry[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        # We disable the active status card so the telemetry card takes its place
        self._engine_status_card = None
        self._engine_progress_bar = None

        self._refresh()

    # --- Phase 3: Profiling Results ---

    def _render_data_health_card(self, alerts: list[dict[str, Any]], row_count: int) -> Panel:
        """Renders the prioritized list of dataset anomalies, anchored by the row count."""

        grid = Table.grid(padding=(1, 2))
        grid.add_column(justify="center", vertical="top")  # Alert Level Label
        grid.add_column(justify="left")  # Target Column + Diagnostic Message

        row_count_string = f"[muted] Rows Profiled:[/muted] [value]{row_count:,}[/value]"
        summary_text = row_count_string

        if not alerts:
            grid.add_row(
                " [bold black on green][ HEALTHY ][/]",
                "[value]Dataset is healthy. No anomalies detected.[/value]",
            )
        else:
            critical_alerts_count = sum(1 for a in alerts if a["level"] == "CRITICAL")
            warning_alerts_count = sum(1 for a in alerts if a["level"] == "WARNING")

            if critical_alerts_count > 0 or warning_alerts_count > 0:
                issues_count_parts = []
                if critical_alerts_count > 0:
                    issues_count_parts.append(f"[bold #8C0000]{critical_alerts_count} CRITICAL[/]")
                if warning_alerts_count > 0:
                    s = "S" if warning_alerts_count > 1 else ""
                    issues_count_parts.append(f"[bold #997602]{warning_alerts_count} WARNING{s}[/]")

                joined_issues_count = "[muted],[/muted] ".join(issues_count_parts)
                issues_count_string = (
                    f"[muted]Data Issues found: \\[[/muted]{joined_issues_count}[muted]][/muted]"
                )
                summary_text = f"{row_count_string} [muted]|[/muted] {issues_count_string}"

            # Sort alerts by severity: Critical (0) -> Warning (1) -> Info (2)
            alert_level_ranks = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            sorted_alerts = sorted(alerts, key=lambda x: alert_level_ranks.get(x["level"], 3))

            for alert in sorted_alerts:
                alert_level = alert["level"]
                alert_column = alert["column"]
                alert_message = alert["message"]

                if alert_level == "CRITICAL":
                    badge = " [bold black on #8C0000][ CRITICAL ][/]"
                elif alert_level == "WARNING":
                    badge = " [bold black on #997602][ WARNING  ][/]"
                else:
                    badge = " [bold black on #737373][   INFO   ][/]"

                # We stack the column name and the message using a newline
                alert_details = f"[brand]{alert_column}[/brand]\n[muted]└─ {alert_message}[/muted]"

                grid.add_row(badge, alert_details)

        # Assemble the final group (Summary text, blank line, grid)
        panel_content = Group(summary_text, Padding(grid, (2, 0, 0, 0)))

        return Panel(
            panel_content,
            title="[not dim][#FF004D]✚[/] [value]Data Health[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    def _group_column_metrics(
        self, profile: dict[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """
        Scans the flat profile dictionary, extracts base column names,
        and groups them by inferred type (Numeric vs Categorical).
        """
        # Every column processed by the profiler gets a null_count. We use this as the master list.
        columns = set()
        for key in profile:
            if key.endswith("_null_count"):
                columns.add(key.replace("_null_count", ""))

        numerics: dict[str, dict[str, Any]] = {}
        categoricals: dict[str, dict[str, Any]] = {}

        # We sort alphabetically for a deterministic UI layout
        for column in sorted(columns):
            base_metrics = {
                "null_count": profile.get(f"{column}_null_count", 0),
                "n_unique": profile.get(f"{column}_n_unique", 0),
            }

            # Type Inference: If it has a mean, it's definitively Numeric
            if f"{column}_mean" in profile:
                base_metrics.update(
                    {
                        "min": profile.get(f"{column}_min"),
                        "mean": profile.get(f"{column}_mean"),
                        "max": profile.get(f"{column}_max"),
                        "p50": profile.get(f"{column}_p50"),  # Median
                        "histogram": profile.get(f"{column}_histogram", []),
                    }
                )
                numerics[column] = base_metrics
            else:
                base_metrics.update(
                    {
                        "min_length": profile.get(f"{column}_min_length"),
                        "mean_length": profile.get(f"{column}_mean_length"),
                        "max_length": profile.get(f"{column}_max_length"),
                        "top_k": profile.get(f"{column}_top_k", []),
                    }
                )
                categoricals[column] = base_metrics

        return numerics, categoricals

    def _build_sparkline(self, histogram_data: list[dict[str, Any]]) -> str:
        """
        Transforms histogram dictionary list into a Unicode sparkline.
        Maps bin counts to the 8 standard block characters.
        """
        if not histogram_data:
            return ""

        # Extract counts (Polars eager hist returns 'count' in its struct)
        counts = [float(bin_data.get("count", 0)) for bin_data in histogram_data]
        max_count = max(counts) if counts else 0

        if max_count == 0:
            return "[muted]" + (" " * len(counts)) + "[/muted]"

        # 9 characters: Space, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, Full Block
        bars = "  ▂▃▄▅▆▇█"
        sparkline = ""

        for count in counts:
            # Normalize to a 0.0 - 1.0 scale, then map to the 0-8 index of our bars string
            normalized = count / max_count
            bar_index = int(normalized * 8)
            sparkline += bars[bar_index]

        return f"[#7B619C]{sparkline}[/#7B619C]"

    def _build_top_k_string(self, top_k_data: list[dict[str, Any]], row_count: int) -> str:
        """
        Formats the Top-K dictionary list into a dense, readable summary string.
        E.g., "NL (55%), DE (20%)..."
        """
        if not top_k_data or row_count == 0:
            return "[muted]-[/muted]"

        parts = []
        # We only take the top 3 to prevent terminal wrapping
        for item in top_k_data[:3]:
            raw_value = item.get("value")
            count = item.get("count", 0)

            # Handle nulls cleanly
            value_string = "null" if raw_value is None else str(raw_value)

            # We truncate aggressively to protect horizontal real estate
            if len(value_string) > TOP_K_STRING_LENGTH:
                value_string = value_string[:9] + "..."

            value_percentage = (count / row_count) * 100

            # Only show percentages if they are meaningful (> 0%)
            if value_percentage >= 1:
                parts.append(f"{value_string} ({value_percentage:.0f}%)")
            else:
                parts.append(f"{value_string} (<1%)")

        return "[value]" + ", ".join(parts) + "[/value]"

    def _format_number(self, value: Any) -> str:
        """Formats floats to 2 decimal places, leaves ints alone, handles Nones."""
        if value is None:
            return "[muted]-[/muted]"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _format_null_percentage(self, null_count: int, row_count: int) -> str:
        """Calculates and formats the missing values percentage string."""
        null_percentage = (null_count / row_count) * 100 if row_count else 0
        if null_percentage > 0:
            return f"[#C75B5B]{null_percentage:.1f}%[/]"
        return "[value]0%[/]"

    def _render_numeric_table(
        self, numerics: dict[str, dict[str, Any]], row_count: int
    ) -> Table | None:
        """Renders the dedicated table for numeric variables."""
        if not numerics:
            return None

        table = Table(
            box=box.SIMPLE, expand=True, show_edge=False, header_style="muted", border_style="muted"
        )
        table.add_column("Column", style="brand", justify="left")
        table.add_column("Missing", justify="right", style="value")
        table.add_column("Distinct", justify="right", style="value")
        table.add_column("Min", justify="right", style="value")
        table.add_column("Mean", justify="right", style="value")
        table.add_column("Max", justify="right", style="value")
        table.add_column("Distribution", justify="left")

        for column, data in numerics.items():
            null_percentage_string = self._format_null_percentage(
                data.get("null_count", 0), row_count
            )

            distinct_string = f"{data.get('n_unique', 0):,}"

            min_value = self._format_number(data.get("min"))
            mean_value = self._format_number(data.get("mean"))
            max_value = self._format_number(data.get("max"))

            sparkline = self._build_sparkline(data.get("histogram", []))

            table.add_row(
                column,
                null_percentage_string,
                distinct_string,
                min_value,
                mean_value,
                max_value,
                sparkline,
            )

        return table

    def _render_categorical_table(
        self, categoricals: dict[str, dict[str, Any]], row_count: int
    ) -> Table | None:
        """Renders the dedicated table for string, boolean, and categorical variables."""
        if not categoricals:
            return None

        table = Table(
            box=box.SIMPLE, expand=True, show_edge=False, header_style="muted", border_style="muted"
        )
        table.add_column("Column", style="brand", justify="left")
        table.add_column("Missing", justify="right", style="value")
        table.add_column("Distinct", justify="right", style="value")
        table.add_column("Lengths (Min/Avg/Max)", justify="center", style="value")
        table.add_column("Top Values", justify="left")

        for column, data in categoricals.items():
            null_percentage_string = self._format_null_percentage(
                data.get("null_count", 0), row_count
            )

            distinct_str = f"{data.get('n_unique', 0):,}"

            min_length = data.get("min_length")
            mean_length = data.get("mean_length")
            max_length = data.get("max_length")

            if min_length is not None and mean_length is not None and max_length is not None:
                lengths_str = f"{min_length} / {mean_length:.1f} / {max_length}"
            else:
                lengths_str = "-"

            # Generate Top-K String
            top_k_str = self._build_top_k_string(data.get("top_k", []), row_count)

            table.add_row(column, null_percentage_string, distinct_str, lengths_str, top_k_str)

        return table

    def _build_variable_explorer_panel(
        self,
        numerics: dict[str, dict[str, Any]],
        categoricals: dict[str, dict[str, Any]],
        row_count: int,
    ) -> Panel | None:
        """Fuses the numeric and categorical tables into a single Variable Explorer panel."""
        numeric_table = self._render_numeric_table(numerics, row_count)
        categorical_table = self._render_categorical_table(categoricals, row_count)

        if not numeric_table and not categorical_table:
            return None

        renderables: list[RenderableType] = []

        # Sub-header for Numerics
        if numeric_table:
            renderables.append(
                Text.from_markup(
                    " [not dim][accent]#/±[/accent] [value]Numeric Variables[/value][/]"
                )
            )
            renderables.append("")  # Breathing room under the title
            renderables.append(numeric_table)

        # Sub-header for Categoricals
        if categorical_table:
            if numeric_table:
                renderables.append("")
                renderables.append("")  # Extra space to clearly separate the two major sections
            renderables.append(
                Text.from_markup(
                    " [not dim][accent]A/B[/accent] [value]Categorical Variables[/value][/]"
                )
            )
            renderables.append("")  # Breathing room under the title
            renderables.append(categorical_table)

        return Panel(
            Group(*renderables),
            title="[not dim][brand]⊞[/brand] [value]Variable Explorer[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    def render_profiling_results(self, profile: dict[str, Any]) -> None:
        """Assembles the final profiling results from the profile payload."""

        row_count = profile.get("table_row_count", 0)
        alerts = profile.get("alerts", [])
        numerics, categoricals = self._group_column_metrics(profile)

        health_card = self._render_data_health_card(alerts, row_count)
        variable_explorer_card = self._build_variable_explorer_panel(
            numerics, categoricals, row_count
        )

        renderables: list[Any] = [health_card]

        if variable_explorer_card:
            renderables.append(variable_explorer_card)

        self._profiling_results = Group(*renderables)
        self._refresh()
