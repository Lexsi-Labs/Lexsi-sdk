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
):
    """Poll a long-running event stream and render incremental progress.

    Model and system metrics are plotted live (one chart per metric) in a Plotly
    figure when running in a notebook (Jupyter, VS Code or Colab). Outside a
    notebook the metrics fall back to printed summaries so CLI runs are not left
    blind.

    :param api_client: API client with streaming support.
    :param project_name: Project name owning the event.
    :param event_id: Identifier of the event to track.
    :param handle_failed_event: Optional callback to invoke on failure.
    :param progress_message: Label used when printing progress.
    :return: None. Raises on failure events.
    """
    last_message = ""
    log_length = 0
    progress = 0
    last_metric_step = None
    last_system_ts = None

    plot_enabled = in_notebook()
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