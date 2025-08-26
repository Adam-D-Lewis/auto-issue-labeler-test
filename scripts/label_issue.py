#!/usr/bin/env python3
"""
Minimal GitHub Issues labeler for Actions with LLM-assisted recommendations.

Environment:
- GITHUB_EVENT_PATH (set by Actions)
- GITHUB_REPOSITORY (owner/repo)
- GITHUB_TOKEN (repo-scoped token)
- OPENAI_API_KEY (GitHub secret)
- LABELER_DRY_RUN ("true"/"false")
- LABELER_MIN_CONF (e.g., "0.6")
- LABELER_MAX_LABELS (e.g., "3")
- OPENAI_MODEL (e.g., "gpt-4o-mini")
- OPENAI_TIMEOUT_SECS (e.g., "20")
"""

import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple
import requests

VERSION = "v1"

ALLOWED_LABELS = ["bug", "enhancement", "documentation"]

def log(msg: str) -> None:
    print(msg, flush=True)

def get_env(name: str, default: str = "") -> str:
    val = os.environ.get(name, default)
    if val == "" and default == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val

def load_event(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def safe_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

def safe_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def post_issue_comment(owner: str, repo: str, issue_number: int, body: str, token: str) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "issue-labeler-action"
    }
    payload = {"body": body}
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code in (200, 201):
        log("Posted comment.")
        return
    log(f"Failed to post comment: HTTP {resp.status_code} - {resp.text}")
    resp.raise_for_status()

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

    if resp.status_code in (502, 503, 504):
        log(f"Transient error {resp.status_code}, retrying once...")
        time.sleep(2)
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            log(f"Labels added after retry: {labels}")
            return

    log(f"Failed to add labels: HTTP {resp.status_code} - {resp.text}")
    resp.raise_for_status()

def build_openai_prompt(title: str, body: str, allowed: List[str]) -> str:
    return (
        "You are an assistant that assigns GitHub issue labels.\n"
        "Return ONLY a single line of comma-separated labels from the allowed set.\n"
        "Format example:\n"
        "bug, enhancement, documentation\n"
        "Rules:\n"
        f"- Allowed labels: {', '.join(allowed)}\n"
        "- Choose any number of labels from the allowed set (including zero).\n"
        "- If unsure, return an empty line.\n"
        "- Do not include any extra text, code fences, or explanations. Only the CSV line.\n\n"
        f"Issue title: {title}\n"
        f"Issue body:\n{body}\n"
    )

def call_openai(api_key: str, model: str, prompt: str, timeout_secs: int) -> str:
    """
    Calls OpenAI to get plain text output (CSV string of labels only).
    Tries Responses API first, then Chat Completions.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Try Responses API for plain text
    url_resp = "https://api.openai.com/v1/responses"
    data_resp = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 400,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(url_resp, headers=headers, json=data_resp, timeout=timeout_secs)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.post(url_resp, headers=headers, json=data_resp, timeout=timeout_secs)
        if resp.status_code // 100 == 2:
            out = resp.json()
            # Try to extract text in a variety of shapes
            if isinstance(out, dict):
                if "output_text" in out and isinstance(out["output_text"], str):
                    return out["output_text"]
                try:
                    parts = out.get("output", [])
                    if parts and isinstance(parts, list):
                        content = parts[0].get("content", [])
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                text_val = c.get("text")
                                if isinstance(text_val, str):
                                    return text_val
                except Exception:
                    pass
            # As a last resort, return stringified JSON
            return json.dumps(out)
        else:
            log(f"Responses API not successful (HTTP {resp.status_code}); falling back to Chat Completions.")
    except requests.RequestException as e:
        log(f"Responses API error: {e}; falling back to Chat Completions.")

    # Fallback: Chat Completions expecting plain text CSV of labels
    url_chat = "https://api.openai.com/v1/chat/completions"
    sys_prompt = (
        "You are an assistant that assigns GitHub issue labels. "
        "Return ONLY a single line, CSV of labels from the allowed set, no explanations."
    )
    data_chat = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        resp2 = requests.post(url_chat, headers=headers, json=data_chat, timeout=timeout_secs)
        if resp2.status_code == 429:
            time.sleep(2)
            resp2 = requests.post(url_chat, headers=headers, json=data_chat, timeout=timeout_secs)
        if resp2.status_code // 100 != 2:
            raise RuntimeError(f"OpenAI HTTP {resp2.status_code}: {resp2.text}")
        out2 = resp2.json()
        text = out2["choices"][0]["message"]["content"]
        return text
    except requests.RequestException as e:
        raise RuntimeError(f"OpenAI request error (chat fallback): {e}")

def parse_llm_labels_csv(s: str) -> List[str]:
    """
    Parse CSV returned by the LLM in the form:
    "bug, enhancement, documentation"
    - Ignores extra spaces and newlines
    - Accepts empty string (meaning no labels)
    - Returns a list of label names in order
    """
    if not isinstance(s, str):
        s = str(s)
    text = s.strip()
    if not text:
        return []

    # Strip common wrappers (code fences or quotes)
    if text.startswith("```") and text.endswith("```"):
        inner = text.strip("`")
        first_newline = inner.find("\n")
        if first_newline != -1:
            text = inner[first_newline + 1 :].strip()
        else:
            text = inner.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    items = [p.strip() for p in text.replace("\n", " ").split(",")]
    labels = [i for i in items if i]
    return labels

def select_labels_from_list(candidates: List[str], allowed: List[str]) -> List[str]:
    # Keep original order as produced by the model; filter against allowed; no cap
    filtered = [name for name in candidates if name in allowed]
    return filtered

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

        # Config
        dry_run = parse_bool(os.environ.get("LABELER_DRY_RUN", "true"))
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        timeout_secs = safe_int(os.environ.get("OPENAI_TIMEOUT_SECS", "20"), 20)
        openai_key = get_env("OPENAI_API_KEY")  # required

        # Build prompt and call OpenAI
        prompt = build_openai_prompt(title, body, ALLOWED_LABELS)
        log("Calling OpenAI for label recommendations...")
        raw_text = call_openai(openai_key, model, prompt, timeout_secs)
        label_list = parse_llm_labels_csv(raw_text)
        chosen = select_labels_from_list(label_list, ALLOWED_LABELS)

        log(f"LLM raw parsed labels: {label_list}")
        log(f"Chosen labels (filtered): {chosen}")

        # Comment or apply
        if dry_run or not chosen:
            lines = [
                f"Labeler ({VERSION}) recommendations:",
                "",
            ]
            if chosen:
                for name in chosen:
                    lines.append(f"- {name}")
            else:
                lines.append("- No labels recommended (abstain)")
            lines.append("")
            lines.append(f"Mode: {'dry-run' if dry_run else 'apply'}")
            body_text = "\n".join(lines)
            post_issue_comment(owner, repo, int(issue_number), body_text, token)
        else:
            add_labels(owner, repo, int(issue_number), chosen, token)

        return 0

    except Exception as e:
        log(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())