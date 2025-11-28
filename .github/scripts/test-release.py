#!/usr/bin/env python3
import subprocess

default_main_version = "0.1.0"


def get_latest_main_release() -> str:
    """Return the most recent non-dev release tag (falls back to default)."""
    result = subprocess.run(
        [
            "gh",
            "release",
            "list",
            "--limit",
            "100",
            "--json",
            "tagName",
            "--jq",
            'map(.tagName) | map(select(contains(".dev") | not)) | .[0]',
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tag = result.stdout.strip()
    if tag:
        return tag
    print("No main release found, falling back to default")
    return default_main_version


def next_dev_version(base_version: str) -> str:
    """Return next incremental dev version for the given base version."""
    result = subprocess.run(
        [
            "gh",
            "release",
            "list",
            "--limit",
            "200",
            "--json",
            "tagName",
            "--jq",
            ".[].tagName",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    dev_tags = [
        t for t in tags
        if t.startswith(f"{base_version}.dev")
    ]
    if not dev_tags:
        return f"{base_version}.dev1"

    def dev_number(tag: str) -> int:
        suffix = tag.replace(f"{base_version}.dev", "", 1)
        try:
            return int(suffix)
        except ValueError:
            return 0

    next_num = max(dev_number(t) for t in dev_tags) + 1
    return f"{base_version}.dev{next_num}"


def create_new_dev_release():
    """Create a new dev release on GitHub based on the latest main release."""
    base_version = get_latest_main_release()
    dev_version = next_dev_version(base_version)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", dev_version],
        check=True,
    )


if __name__ == "__main__":
    create_new_dev_release()
