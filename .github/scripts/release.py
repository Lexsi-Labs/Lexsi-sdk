#!/usr/bin/env python3
import subprocess

default_version = "0.1.0"

def get_last_version() -> str:
    """Return the version number of the last release."""
    result = subprocess.run(
        ["gh", "release", "list", "--limit", "1", "--json", "tagName", "--jq", ".[0].tagName"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tag = result.stdout.strip()
    if not tag:
        return default_version
    return tag


def bump_patch_number(version_number: str) -> str:
    """Return a copy of `version_number` with the patch number incremented."""
    major, minor, patch = version_number.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def create_new_patch_release():
    """Create a new patch release on GitHub."""
    try:
        last_version_number = get_last_version()
    except subprocess.CalledProcessError as err: 
        # The project doesn't have any releases yet.
        new_version_number = default_version
        print(err)
        print(f'taking default version: {new_version_number}')
    else:
        new_version_number = bump_patch_number(last_version_number)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", f"{new_version_number}"],
        check=True,
    )


if __name__ == "__main__":
    create_new_patch_release()