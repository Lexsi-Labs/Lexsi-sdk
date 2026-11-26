from typing import Any, Dict, List, Optional

from lexsi_sdk.client.client import APIClient


def _validate_sample_fields(
    samples: Optional[List[Dict[str, Any]]],
    *,
    input_column: Optional[str] = None,
    target_column: Optional[str] = None,
    actual_output_column: Optional[str] = None,
) -> None:
    """Reject inline sample dicts whose keys are not read during evaluation (F168).

    Samples using ``output`` / ``expected_output`` instead of
    ``actual_output`` / ``target`` used to be stored as ``completed`` with an
    empty headline and every scorer skipped. Fail fast with a rename hint
    rather than letting a silent no-score run through.

    Explicit column mappings opt out: they tell the worker exactly which
    keys to read, so those key names are legitimate in that case.

    :param samples: the inline sample dictionaries to validate.
    :param input_column: optional override for the input field name.
    :param target_column: optional override for the target field name.
    :param actual_output_column: optional override for the output field name.
    """
    if not samples:
        return
    expected = {
        "input", "id", "actual_output", "target", "task",
        "choices", "retrieval_context", "metadata",
    }
    for column in (input_column, target_column, actual_output_column):
        if column:
            expected.add(column)
    aliases = {
        "output": "actual_output",
        "expected_output": "target",
        "expected": "target",
        "ground_truth": "target",
        "prediction": "actual_output",
        "generated_output": "actual_output",
    }
    problems = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        for key in sample:
            if key not in expected:
                problems.setdefault(key, aliases.get(key))
    if not problems:
        return
    hints = [
        f"'{key}'" + (f" (use '{replacement}')" if replacement else "")
        for key, replacement in problems.items()
    ]
    raise ValueError(
        "Invalid sample field names: "
        + "; ".join(hints)
        + ". Use the documented fields (actual_output, target) or pass "
        "explicit column mappings (actual_output_column, target_column) "
        "to run_eval."
    )


def _with_legacy_prediction_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a prediction row with both current and legacy field names.

    AuditKit 1.0 renamed the answer-browser sample fields to ``prompt`` /
    ``expected`` and moved the per-scorer breakdown into ``metadata.scores``
    (a list of ``{name, value, ...}``). Notebooks written against the earlier
    response read ``input`` / ``target`` and a top-level ``scores`` dict.
    Add those aliases without dropping the canonical keys so both older and
    newer callers keep working.

    :param row: one prediction row as returned by the runs API.
    :return: a shallow copy of *row* with legacy aliases filled in.
    """
    if not isinstance(row, dict):
        return row
    compat = dict(row)

    # A server that already emits legacy keys wins; only fill what is missing.
    aliases = {
        "input": ("prompt",),
        "target": ("expected",),
        "actual_output": ("raw_output", "parsed_answer"),
        "id": ("sample_id",),
    }
    for legacy, sources in aliases.items():
        if compat.get(legacy) is not None:
            continue
        for source in sources:
            value = compat.get(source)
            if value is not None:
                compat[legacy] = value
                break

    metadata = compat.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    score_docs = metadata.get("scores")
    if not isinstance(score_docs, list):
        score_docs = []
    if compat.get("scores") is None:
        compat["scores"] = {
            str(score["name"]): score.get("value")
            for score in score_docs
            if isinstance(score, dict) and score.get("name") is not None
        }
    if compat.get("score") is None:
        for score in score_docs:
            if isinstance(score, dict) and score.get("value") is not None:
                compat["score"] = score["value"]
                break
    # Generation diagnostics (provider errors, empty/truncated answers) live
    # in per-sample metadata; surface them on the row for notebooks.
    for key in ("error", "status", "empty", "finish_reason", "reasoning_tokens", "refusal"):
        if compat.get(key) is None and metadata.get(key) is not None:
            compat[key] = metadata[key]
    # Rows from older/cached runs may have no diagnostic; a blank generation
    # is still a failed inference, so surface one for the notebook.
    if "raw_output" in compat and not str(compat.get("raw_output") or "").strip():
        if compat.get("error") is None:
            compat["error"] = "No output generated (inference did not produce content)"
        if compat.get("status") is None:
            compat["status"] = "empty"
        compat.setdefault("empty", True)
    return compat


def build_url(base_url, data_connector_name, project_name, organization_id):
    """Compose a connector URL using project or organization context.

    :param base_url: Base API path.
    :param data_connector_name: Link service name.
    :param project_name: Project identifier if available.
    :param organization_id: Organization identifier if available.
    :return: Fully composed URL string.
    """
    url = None
    if project_name:
        url = f"{base_url}?project_name={project_name}&link_service_name={data_connector_name}"
    elif organization_id:
        url = f"{base_url}?organization_id={organization_id}&link_service_name={data_connector_name}"
    else:
        url = f"{base_url}?link_service_name={data_connector_name}"
    return url


def build_list_data_connector_url(base_url, project_name, organization_id):
    """Build list URL for data connectors within a project or organization.

    :param base_url: Base API path.
    :param project_name: Project identifier if available.
    :param organization_id: Organization identifier if available.
    :return: Fully composed URL string.
    """
    url = None
    if project_name and organization_id:
        url = (
            f"{base_url}?project_name={project_name}&organization_id={organization_id}"
        )
    elif project_name:
        url = f"{base_url}?project_name={project_name}"
    elif organization_id:
        url = f"{base_url}?organization_id={organization_id}"
    else:
        url = base_url
    return url

def split_cpu_gpu_servers(servers: list) -> dict:
    cpu_servers = []
    gpu_servers = []

    for server in servers:
        if "gpu" in server:
            gpu_servers.append(server)
        else:
            cpu_servers.append(server)

    return {
        "cpu_servers": cpu_servers,
        "gpu_servers": gpu_servers
    }

