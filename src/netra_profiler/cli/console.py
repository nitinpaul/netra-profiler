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

NULL_PERCENTAGE_THRESHOLD = 0.5
LATENCY_THRESHOLD = 0.01
EXECUTION_TIME_THRESHOLD = 0.01

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
        self._results_dashboard: Group | None = None

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

        if self._results_dashboard:
            active_cards.append(self._results_dashboard)

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
    def start_data_source_connection(self, data_source_name: str) -> None:
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

    def finish_data_source_connection(
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
    def start_data_profiling(self) -> Progress:
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
            title="[not dim][accent]⚙[/accent] [value]Engine Status[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        self._refresh()
        return self._engine_progress_bar

    def finish_data_profiling(
        self, execution_time: float, throughput_gb_s: float, peak_ram_usage: float
    ) -> None:
        """Morphs the active engine status into a permanent hardware telemetry card."""
        grid = Table.grid(expand=True)
        grid.add_column(style="muted")

        execution_time_string = (
            "< 0.01s" if execution_time < EXECUTION_TIME_THRESHOLD else f"{execution_time:.2f}s"
        )

        grid.add_row(f" Execution Time:  [not dim][brand]{execution_time_string}[/brand][/]")
        grid.add_row(f" Data Throughput: [not dim][brand]{throughput_gb_s:.2f} GB/s[/brand][/]")
        grid.add_row(
            f" Peak RAM Usage:  [not dim][brand]{peak_ram_usage:.1f} MB[/brand][/] "
            "[muted](Includes Arrow allocation & compute buffers)[/muted]"
        )

        self._engine_telemetry_card = Panel(
            grid,
            title="[not dim][accent]⏣[/accent] [value]Engine Telemetry[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )
        # The Swap: Destroy the active status card so the telemetry card takes its place
        self._engine_status_card = None
        self._engine_progress_bar = None

        self._refresh()

    def _render_data_health_card(self, alerts: list[dict[str, Any]]) -> Panel:
        """Renders the prioritized list of dataset anomalies."""

        grid = Table.grid(padding=(1, 2))
        grid.add_column(justify="center", vertical="top")  # Alert Level Label
        grid.add_column(justify="left")  # Target Column + Diagnostic Message

        summary_text = None

        if not alerts:
            grid.add_row(
                "[bold black on #8FBC8F] HEALTHY  [/]",
                "[value]Dataset is healthy. No anomalies detected.[/value]",
            )
        else:
            crit_count = sum(1 for a in alerts if a["level"] == "CRITICAL")
            warn_count = sum(1 for a in alerts if a["level"] == "WARNING")

            if crit_count > 0 or warn_count > 0:
                parts = []
                if crit_count > 0:
                    parts.append(f"[bold #8C0000]{crit_count} CRITICAL[/]")
                if warn_count > 0:
                    s = "S" if warn_count > 1 else ""
                    parts.append(f"[bold #997602]{warn_count} WARNING{s}[/]")

                summary_text = (
                    f"[muted] Data Issues found: \\[[/muted]{', '.join(parts)}[muted]][/muted]"
                )

            # Sort alerts by severity: Critical (0) -> Warning (1) -> Info (2)
            alert_level_ranks = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            sorted_alerts = sorted(alerts, key=lambda x: alert_level_ranks.get(x["level"], 3))

            for alert in sorted_alerts:
                lvl = alert["level"]
                col = alert["column"]
                msg = alert["message"]

                if lvl == "CRITICAL":
                    badge = " [bold black on #8C0000][ CRITICAL ][/]"
                elif lvl == "WARNING":
                    badge = " [bold black on #997602][ WARNING  ][/]"
                else:
                    badge = " [bold black on #737373][   INFO   ][/]"

                # We stack the column name and the message using a newline
                alert_details = f"[brand]{col}[/brand]\n[muted]└─ {msg}[/muted]"

                grid.add_row(badge, alert_details)

        panel_content = Group(summary_text, "", grid) if summary_text else grid

        return Panel(
            panel_content,
            title="[not dim][#FF004D]✚[/] [value]Data Health[/value][/]",
            title_align="left",
            border_style="border.section",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    # --- Phase 3: Results Dashboard ---
    def render_results_dashboard(
        self, profile: dict[str, Any], filename: str, duration: float
    ) -> None:
        """
        Final UI State.
        """
        # 1. Extract the alerts from the profile payload
        alerts = profile.get("alerts", [])

        # 2. Render the components
        data_health_card = self._render_data_health_card(alerts)

        # 3. Stack the final dashboard
        self._results_dashboard = Group(data_health_card)
        self._refresh()
