from datetime import datetime
from typing import Callable, Optional
from lexsi_sdk.client.client import APIClient
from IPython.display import display, HTML

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
    """Poll long-running event stream and print incremental progress.

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

    for event in api_client.stream(
        uri=f"{POLL_EVENTS}?project_name={project_name}&event_id={event_id}",
        method="GET"
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
        if details.get("progress"):
            if details.get("progress") != progress:
                progress = details.get("progress")
                print(f"{progress_message}: {progress}%")
            # display(HTML(f"<progress style='width:100%' value='{progress}' max='100'></progress>"))
        if details.get("status") == "failed":
            if handle_failed_event:
                handle_failed_event()
            raise Exception(details.get("message"))
