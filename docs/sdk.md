# SDK Documentation

## Getting Started
The recommended pattern is:

```python
from lexsi_sdk import xai as lexsi

# Login using your Lexsi API key
lexsi.login(api_key="YOUR_API_KEY")

```

::: lexsi_sdk.core.xai
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

::: lexsi_sdk.core.text
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

::: lexsi_sdk.core.guardrails.guardrails_langgraph
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true

::: lexsi_sdk.core.guardrails.guardrails_openai
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_source: true
