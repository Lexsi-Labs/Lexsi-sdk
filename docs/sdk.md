# SDK Documentation

## Getting Started

SDK token can be generated from the [Lexsi Console](https://console.lexsi.ai) under  
**Dashboard → Access Token**:  
[https://console.lexsi.ai/dashboard/access-token](https://console.lexsi.ai/dashboard/access-token)

```python
from lexsi_sdk import lexsi

# Login using your Lexsi SDK Token
lexsi.login(sdk_access_token="YOUR_SDK_TOKEN")

```

::: lexsi_sdk.core.lexsi
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true
      filters:
        - "!case_profile"

## Working With Organizations
The recommended pattern is:

```python
organization = lexsi.organization("Your Organization name")
```
<p> You can use the following function with organization class :</p>

::: lexsi_sdk.core.organization
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

## Working With Workspaces
The recommended pattern is:

```python
workspace = organization.workspace("Your workspace name")
```
<p> You can use the following function with workspace class :</p>

::: lexsi_sdk.core.workspace
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

## Working With Projects
The recommended pattern is:

```python
project = workspace.project("Your Project name")
```
<p> You can use the following function with organization class :</p>

::: lexsi_sdk.core.project
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.case
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.dashboard
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.alert
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.text
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.tracer
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.model_summary
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.synthetic
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.agent
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true


## Data Classes
::: lexsi_sdk.common.types
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: false