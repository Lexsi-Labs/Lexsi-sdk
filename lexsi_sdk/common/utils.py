from datetime import date, datetime, time, timedelta, timezone
import re
import warnings
from typing import Callable, Optional, Union
from lexsi_sdk.client.client import APIClient

from lexsi_sdk.common.live_plot import LiveMetricsPlotter, in_notebook
from lexsi_sdk.common.xai_uris import POLL_EVENTS


def parse_float(s):
    """parse float from string, return None if not possible

    :param s: string to parse
    :return: float or None
    """
    try:
        return float(s)
    except ValueError:
        return None


def parse_datetime(s, format="%Y-%m-%d %H:%M:%S"):
    """Parse datetime from string, return None if not possible

    :param s: string to parse
    :param format: format string for datetime parsing
    :return: datetime or None
    """
    try:
        return datetime.strptime(s, format)
    except ValueError:
        return None


def pretty_date(date: str) -> str:
    """return date in format dd-mm-YYYY HH:MM:SS

    :param date: str datetime
    :return: pretty datetime
    """
    try:
        datetime_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        try:
            datetime_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            print("Date format invalid.")

    return datetime_obj.strftime("%d-%m-%Y %H:%M:%S")


def poll_events(
    api_client: APIClient,
    project_name: str,
    event_id: str,
    handle_failed_event: Optional[Callable] = None,
    progress_message: str = "progress",
    plot: bool = True,
):
    """Poll a long-running event stream and render incremental progress.

    Model and system metrics are plotted live (one chart per metric) in a Plotly
    figure when running in a notebook (Jupyter, VS Code or Colab). Outside a
    notebook — or when ``plot`` is False — the metrics fall back to printed
    summaries so CLI runs are not left blind.

    :param api_client: API client with streaming support.
    :param project_name: Project name owning the event.
    :param event_id: Identifier of the event to track.
    :param handle_failed_event: Optional callback to invoke on failure.
    :param progress_message: Label used when printing progress.
    :param plot: When False, always print raw metric summaries instead of
        plotting, even in environments where charts could be rendered.
    :return: None. Raises on failure events.
    """
    last_message = ""
    log_length = 0
    progress = 0
    last_metric_step = None
    last_system_ts = None

    plot_enabled = plot and in_notebook()
    model_plotter = LiveMetricsPlotter("Model metrics", "step") if plot_enabled else None
    system_plotter = (
        LiveMetricsPlotter("System metrics", "timestamp", mode="lines") if plot_enabled else None
    )

    for event in api_client.stream(
        uri=f"{POLL_EVENTS}?project_name={project_name}&event_id={event_id}",
        method="GET",
    ):
        details = event.get("details")

        if not event.get("success"):
            raise Exception(details)
        if details.get("event_logs"):
            print(details.get("event_logs")[log_length:])
            log_length = len(details.get("event_logs"))
        if details.get("message") != last_message:
            last_message = details.get("message")
            print(f"{details.get('message')}")

        model_metrics = details.get("model_metrics") or {}
        system_metrics = details.get("system_metrics") or {}

        if plot_enabled:
            try:
                model_plotter.update(model_metrics)
                system_plotter.update(system_metrics)
            except Exception as exc:
                warnings.warn(
                    f"Live metric plotting disabled, falling back to printed "
                    f"metrics: {exc!r}",
                    stacklevel=2,
                )
                plot_enabled = False

        if not plot_enabled:
            if model_metrics:
                latest_step = max(
                    (points[-1][0] for points in model_metrics.values() if points),
                    default=None,
                )
                if latest_step is not None and latest_step != last_metric_step:
                    last_metric_step = latest_step
                    summary = ", ".join(
                        f"{name}={points[-1][1]:.4g}"
                        for name, points in sorted(model_metrics.items())
                        if points
                    )
                    if summary:
                        print(f"  [model metrics] step {latest_step}: {summary}")

            if system_metrics:
                latest_ts = max(
                    (points[-1][0] for points in system_metrics.values() if points),
                    default=None,
                )
                if latest_ts is not None and latest_ts != last_system_ts:
                    last_system_ts = latest_ts
                    summary = ", ".join(
                        f"{name.replace('system/', '')}={points[-1][1]:.4g}"
                        for name, points in sorted(system_metrics.items())
                        if points
                    )
                    if summary:
                        print(f"  [system metrics]  {summary}")

        if details.get("progress"):
            if details.get("progress") != progress:
                progress = details.get("progress")
                print(f"{progress_message}: {progress}%")
            # display(HTML(f"<progress style='width:100%' value='{progress}' max='100'></progress>"))
        if details.get("status") == "failed":
            if handle_failed_event:
                handle_failed_event()
            raise Exception(details.get("message"))


def _print_metric_history(label: str, metrics: dict, x_label: str) -> None:
    """Print the full step-by-step history for a group of metrics.

    Metrics arrive as ``{name: [[x, y], ...]}`` cumulative series that may be
    logged at different x values (e.g. train metrics every few steps, eval
    metrics only at epoch boundaries). Points are aligned by their x value so
    each printed row lists every metric that has a value at that step/timestamp.

    :param label: Group label used in the line prefix, e.g. ``"model"``.
    :param metrics: mapping of ``{metric_name: [[x, y], ...]}``.
    :param x_label: Name of the x axis printed before each x value, e.g. ``"step"``.
    """
    rows: dict = {}
    for name, points in metrics.items():
        clean = name.replace("system/", "")
        for point in points:
            if point:
                rows.setdefault(point[0], {})[clean] = point[1]

    for x in sorted(rows):
        summary = ", ".join(f"{name}={value:.4g}" for name, value in sorted(rows[x].items()))
        if summary:
            x_str = int(x) if isinstance(x, float) and x.is_integer() else x
            print(f"  [{label} metrics] {x_label} {x_str}: {summary}")


def fetch_event_metrics(
    api_client: APIClient,
    project_name: str,
    event_id: str,
    plot: bool = True,
) -> dict:
    """Fetch a completed event's metrics from the poll stream and plot them.

    Re-uses the ``events/poll`` stream that powers live training metrics. For a
    completed event the stream yields a single message carrying the full
    cumulative metric series, so this drains the stream, extracts the latest
    ``model_metrics``/``system_metrics``, and renders them with the same
    :class:`LiveMetricsPlotter` used during training (one subplot per metric).
    Outside a notebook — or when ``plot`` is False — it prints the final values
    instead.

    :param api_client: API client with streaming support.
    :param project_name: Project name owning the event.
    :param event_id: Identifier of the finetuning event to read metrics from.
    :param plot: When False, always print raw metric summaries instead of
        plotting, even in environments where charts could be rendered.
    :return: mapping with ``model_metrics`` and ``system_metrics`` series.
    """
    model_metrics: dict = {}
    system_metrics: dict = {}

    for event in api_client.stream(
        uri=f"{POLL_EVENTS}?project_name={project_name}&event_id={event_id}",
        method="GET",
    ):
        details = event.get("details")
        if not event.get("success"):
            raise Exception(details)
        # Series are cumulative, so the latest payload supersedes earlier ones.
        if details.get("model_metrics"):
            model_metrics = details["model_metrics"]
        if details.get("system_metrics"):
            system_metrics = details["system_metrics"]
        if details.get("status") in ("completed", "failed"):
            break

    rendered = False
    if plot and in_notebook():
        try:
            if system_metrics:
                LiveMetricsPlotter("System metrics", "timestamp", mode="lines").update(system_metrics)
            if model_metrics:
                LiveMetricsPlotter("Model metrics", "step").update(model_metrics)
            rendered = True
        except Exception as exc:
            warnings.warn(
                f"Metric plotting failed, falling back to printed metrics: {exc!r}",
                stacklevel=2,
            )

    if not rendered:
        if system_metrics:
            _print_metric_history("system", system_metrics, "timestamp")
        if model_metrics:
            _print_metric_history("model", model_metrics, "step")

    if not model_metrics and not system_metrics:
        print("No metrics found for the event")

    return {"model_metrics": model_metrics, "system_metrics": system_metrics}


TIME_RE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2})(?P<tz>Z|[+-]\d{2}:\d{2})?$"
)

def normalize_time(
    value: Optional[Union[str, datetime]],
    base_date: Optional[date] = None
) -> Optional[str]:

    if value is None:
        return None
    base_date = base_date or datetime.utcnow().date()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    if isinstance(value, str):
        match = TIME_RE.match(value)
        if not match:
            raise ValueError("Time must be HH:MM, HH:MM±HH:MM, or HH:MMZ")

        hour = int(match.group("h"))
        minute = int(match.group("m"))
        tz_part = match.group("tz")
        tzinfo = timezone.utc

        if tz_part:
            if tz_part == "Z":
                tzinfo = timezone.utc
            else:
                sign = 1 if tz_part[0] == "+" else -1
                tzh, tzm = map(int, tz_part[1:].split(":"))
                tzinfo = timezone(
                    sign * timedelta(hours=tzh, minutes=tzm)
                )

        dt = datetime.combine(base_date, time(hour, minute, tzinfo=tzinfo))
        return dt.isoformat()

    raise ValueError("Invalid time value")