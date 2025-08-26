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
        "Return ONLY a compact JSON object with this schema:\n"
        '{ "labels": [{"name": "<one of allowed>", "confidence": 0.0}],'
        ' "rationale": "short explanation", "version": "v1" }\n'
        "Rules:\n"
        f"- Allowed labels: {', '.join(allowed)}\n"
        "- Choose up to 3 labels. Use confidences between 0 and 1.\n"
        "- If unsure, return an empty labels array and brief rationale.\n"
        "- Do not include any extra text outside the JSON.\n\n"
        f"Issue title: {title}\n"
        f"Issue body:\n{body}\n"
    )

def call_openai(api_key: str, model: str, prompt: str, timeout_secs: int) -> str:
    """
    Calls OpenAI responses endpoint to get a JSON-only reply.
    Using the 'responses' API is recommended; fallback to chat if needed.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 400,
        "temperature": 0.2,
        # Encourage JSON only
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=timeout_secs)
    except requests.RequestException as e:
        raise RuntimeError(f"OpenAI request error: {e}")

    if resp.status_code == 429:
        # simple backoff and single retry
        time.sleep(2)
        resp = requests.post(url, headers=headers, json=data, timeout=timeout_secs)

    if resp.status_code // 100 != 2:
        raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text}")

    out = resp.json()
    # Responses API returns content as array of outputs; extract text
    # Format: { "output": [{"content":[{"type":"output_text","text":"..."}]}], ... } OR
    # newer: { "output_text": "{...json...}" }
    if "output_text" in out and isinstance(out["output_text"], str):
        return out["output_text"]
    # fallback parse
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
    # As last resort, return entire string
    return json.dumps(out)

def validate_llm_json(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON: {e}")

    labels = obj.get("labels", [])
    rationale = obj.get("rationale", "")
    version = obj.get("version", VERSION)

    if not isinstance(labels, list):
        raise RuntimeError("Invalid schema: 'labels' must be a list")
    norm_labels = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        conf = item.get("confidence")
        if not isinstance(name, str):
            continue
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        norm_labels.append({"name": name.strip(), "confidence": max(0.0, min(1.0, conf_f))})

    return {"labels": norm_labels, "rationale": str(rationale), "version": str(version)}

def select_labels(candidates: List[Dict[str, Any]], allowed: List[str], min_conf: float, max_labels: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    filtered = [c for c in candidates if (c["name"] in allowed and c["confidence"] >= min_conf)]
    # sort by confidence desc
    filtered.sort(key=lambda x: x["confidence"], reverse=True)
    top = filtered[: max(0, max_labels)]
    chosen = [x["name"] for x in top]
    return chosen, filtered

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
        min_conf = safe_float(os.environ.get("LABELER_MIN_CONF", "0.6"), 0.6)
        max_labels = safe_int(os.environ.get("LABELER_MAX_LABELS", "3"), 3)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        timeout_secs = safe_int(os.environ.get("OPENAI_TIMEOUT_SECS", "20"), 20)
        openai_key = get_env("OPENAI_API_KEY")  # required

        # Build prompt and call OpenAI
        prompt = build_openai_prompt(title, body, ALLOWED_LABELS)
        log("Calling OpenAI for label recommendations...")
        raw_json_text = call_openai(openai_key, model, prompt, timeout_secs)
        parsed = validate_llm_json(raw_json_text)
        recs = parsed["labels"]
        chosen, filtered = select_labels(recs, ALLOWED_LABELS, min_conf, max_labels)

        log(f"LLM raw parsed: {parsed}")
        log(f"Chosen labels (after filtering): {chosen}")

        # Comment or apply
        if dry_run or not chosen:
            # Prepare a readable summary
            lines = [
                f"Labeler ({VERSION}) recommendations:",
                "",
            ]
            if filtered:
                for item in filtered[:max_labels]:
                    lines.append(f"- {item['name']}: {item['confidence']:.2f}")
            else:
                lines.append("- No confident labels (abstain)")
            if parsed.get("rationale"):
                lines.append("")
                lines.append(f"Rationale: {parsed['rationale']}")
            lines.append("")
            lines.append(f"Mode: {'dry-run' if dry_run else 'apply'} • Threshold: {min_conf} • Cap: {max_labels}")
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