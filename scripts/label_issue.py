#!/usr/bin/env python3
"""
Minimal GitHub Issues labeler for Actions.

Reads the GitHub event payload (issues: opened/edited), computes labels using
determine_labels(), and applies them to the issue using GITHUB_TOKEN.

Environment:
- GITHUB_EVENT_PATH (set by Actions)
- GITHUB_REPOSITORY (owner/repo)
- GITHUB_TOKEN (repo-scoped token)
"""

import json
import os
import sys
import time
from typing import List
import requests


def log(msg: str) -> None:
    print(msg, flush=True)


def get_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def load_event(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def determine_labels(title: str, body: str, author: str) -> List[str]:
    """
    Placeholder labeling logic. Customize freely.

    Examples:
    - If title/body mentions 'bug' => add 'bug'
    - If mentions 'feature' or 'enhancement' => add 'enhancement'
    - If mentions 'docs' or 'documentation' => add 'documentation'
    """
    t = (title or "").lower()
    b = (body or "").lower()
    labels: List[str] = []

    def add(label: str):
        if label not in labels:
            labels.append(label)

    text = f"{t}\n{b}"

    if "bug" in text or "error" in text or "exception" in text:
        add("bug")

    if "feature" in text or "enhancement" in text or "improvement" in text:
        add("enhancement")

    if "docs" in text or "documentation" in text or "readme" in text:
        add("documentation")

    # Example: label issues created by bots differently
    if author.endswith("[bot]"):
        add("bot")

    return labels


def ensure_labels_exist(owner: str, repo: str, labels: List[str], token: str) -> None:
    """
    Optionally create missing labels. By default, this function is a no-op
    because GITHUB_TOKEN does not grant label administration across all repos.
    Uncomment to attempt creation if you want that behavior and have permissions.

    Note: Creating labels requires 'issues: write' and sometimes repository admin
    permissions depending on org policies.

    Example (disabled):
    # for label in labels:
    #     if not label_exists(owner, repo, label, token):
    #         create_label(owner, repo, label, token, color='ededed')
    """
    return


def add_labels(owner: str, repo: str, issue_number: int, labels: List[str], token: str) -> None:
    if not labels:
        log("No labels to add. Exiting.")
        return

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/labels"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "issue-labeler-action"
    }
    payload = {"labels": labels}
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code in (200, 201):
        log(f"Labels added: {labels}")
        return

    # Retry once on 502/503/504
    if resp.status_code in (502, 503, 504):
        log(f"Transient error {resp.status_code}, retrying once...")
        time.sleep(2)
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            log(f"Labels added after retry: {labels}")
            return

    log(f"Failed to add labels: HTTP {resp.status_code} - {resp.text}")
    resp.raise_for_status()


def main() -> int:
    try:
        event_path = get_env("GITHUB_EVENT_PATH")
        repo_full = get_env("GITHUB_REPOSITORY")
        token = get_env("GITHUB_TOKEN")

        owner, repo = repo_full.split("/", 1)
        event = load_event(event_path)

        if event.get("action") not in ("opened", "edited"):
            log(f"Ignoring action: {event.get('action')}")
            return 0

        issue = event.get("issue") or {}
        issue_number = issue.get("number")
        title = issue.get("title") or ""
        body = issue.get("body") or ""
        user = (issue.get("user") or {}).get("login") or ""

        if not issue_number:
            log("No issue number in payload; exiting.")
            return 0

        labels = determine_labels(title, body, user)
        log(f"Computed labels: {labels}")

        ensure_labels_exist(owner, repo, labels, token)
        add_labels(owner, repo, int(issue_number), labels, token)
        return 0

    except Exception as e:
        log(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())