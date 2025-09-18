#!/usr/bin/env python3
"""
Batch labeler: reads JSONL issues, queries OpenAI for label recommendations restricted
to a predefined set, and writes results to JSONL.

Usage:
  pixi run py scripts/batch_label_issues.py \
      --input scripts/data/export/issues.jsonl \
      --output scripts/data/export/generated-labels.jsonl \
      --model gpt-5 \
      --max-workers 4

Environment:
  - OPENROUTER_API_KEY (required)
  - OPENROUTER_TIMEOUT_SECS (optional, default 20)
  - OPENROUTER_MODEL (optional; overridden by --model if provided)

Notes:
  - Allowed label set mirrors scripts/label_issue.py
  - Robust to both Responses API and Chat Completions API, same approach as scripts/label_issue.py
"""

import argparse
import concurrent.futures
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
 
import requests
import yaml
 

 
def load_labels_from_yaml(path: str) -> Dict[str, str]:
    """
    Loads labels and descriptions from a YAML file formatted like .github/labels.yml
    Returns a dictionary of label_name: description
    """
    with open(path, "r", encoding="utf-8") as f:
        labels_data = yaml.safe_load(f)
 
    labels_map = {}
    if isinstance(labels_data, list):
        for item in labels_data:
            if isinstance(item, dict) and "name" in item and "description" in item:
                labels_map[item["name"]] = item["description"]
    return labels_map
 
 
ALLOWED_LABELS_WITH_DESC: Dict[str, str] = {}
 
VERSION = "batch-v1"
 
 
def log(msg: str) -> None:
    print(msg, flush=True)


def safe_int(v: Optional[str], default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def build_openai_prompt(title: str, body: str, allowed_with_desc: Dict[str, str]) -> str:
    allowed_list_str = "\n".join(
        [f'- "{name}": {desc}' for name, desc in allowed_with_desc.items()]
    )
    return (
        "You are an assistant that assigns GitHub issue labels.\n"
        "Return ONLY a single line of comma-separated labels from the allowed set.\n"
        "Format example:\n"
        "bug, enhancement, documentation\n"
        "Rules:\n"
        "- Choose any number of labels from the allowed set (including zero).\n"
        "- Do not include any extra text, code fences, or explanations. Only the CSV line.\n\n"
        "Allowed labels with descriptions:\n"
        f"{allowed_list_str}\n\n"
        f"Issue title: {title}\n"
        f"Issue body:\n{body}\n"
    )


def call_openai(api_key: str, model: str, prompt: str, timeout_secs: int) -> str:
    """
    Calls OpenRouter to get plain text output (CSV string of labels only).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Use OpenRouter's Chat Completions endpoint
    url_chat = "https://openrouter.ai/api/v1/chat/completions"
    sys_prompt = (
        "You are an assistant that assigns GitHub issue labels. "
        "Return ONLY a single line, CSV of labels from the allowed set, no explanations."
    )
    data_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    try:
        resp = requests.post(url_chat, headers=headers, json=data_chat, timeout=timeout_secs)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.post(url_chat, headers=headers, json=data_chat, timeout=timeout_secs)
        if resp.status_code // 100 != 2:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")
        out = resp.json()
        text = out["choices"][0]["message"]["content"]
        return text
    except requests.RequestException as e:
        raise RuntimeError(f"OpenRouter request error: {e}")


def parse_llm_labels_csv(s: str) -> List[str]:
    """
    Parse CSV returned by the LLM in the form:
    "bug, enhancement, documentation"
    - Ignores extra spaces and newlines
    - Accepts empty string (meaning no labels)
    - Returns a list of label names in order
    Mirrors scripts/label_issue.py
    """
    if not isinstance(s, str):
        s = str(s)
    text = s.strip()
    if not text:
        return []

    # Strip code fences or quotes
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
    # Keep original order; filter against allowed
    return [name for name in candidates if name in allowed]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield line_idx, obj
            except json.JSONDecodeError as e:
                log(f"Skipping invalid JSONL line {line_idx}: {e}")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_issue(
    issue: Dict[str, Any],
    api_key: str,
    model: str,
    timeout_secs: int,
) -> Tuple[Any, List[str], Optional[str]]:
    """
    Returns: (id, labels, error_message_if_any)
    """
    issue_id = issue.get("id")
    title = issue.get("title") or ""
    body = issue.get("body") or ""

    try:
        prompt = build_openai_prompt(title, body, ALLOWED_LABELS_WITH_DESC)
        raw_text = call_openai(api_key, model, prompt, timeout_secs)
        label_list = parse_llm_labels_csv(raw_text)
        allowed_names = list(ALLOWED_LABELS_WITH_DESC.keys())
        chosen = select_labels_from_list(label_list, allowed_names)
        return (issue_id, chosen, None)
    except Exception as e:
        return (issue_id, [], str(e))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch label GitHub issues via OpenRouter and write JSONL outputs.")
    p.add_argument("--input", required=True, help="Path to input issues.jsonl")
    p.add_argument("--output", help="Path to write generated-labels.jsonl (required unless using --stdout)")
    p.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-pro"), help="OpenRouter model (default: google/gemini-2.5-pro)")
    p.add_argument("--max-workers", type=int, default=4, help="Concurrency for API calls (default: 4)")
    p.add_argument("--limit", type=int, help="Limit processing to first N issues")
    p.add_argument("--stdout", action="store_true", help="Output results to stdout instead of file")
    p.add_argument("--skip-errors", action="store_true", help="Do not fail on per-issue errors; record empty labels and continue")
    p.add_argument("--clobber", action="store_true", help="Overwrite output file if it exists and is not empty")
    p.add_argument(
        "--labels-yaml",
        default=".github/labels.yml",
        help="Path to YAML file with label names and descriptions (default: .github/labels.yml)",
    )
    args = p.parse_args(argv)

    # Validate that output is provided when not using stdout
    if not args.stdout and not args.output:
        p.error("--output is required unless using --stdout")
 
    global ALLOWED_LABELS_WITH_DESC
    try:
        ALLOWED_LABELS_WITH_DESC = load_labels_from_yaml(args.labels_yaml)
        if not ALLOWED_LABELS_WITH_DESC:
            log(f"Warning: No labels loaded from {args.labels_yaml}. Check file format.")
    except Exception as e:
        log(f"Error loading labels from {args.labels_yaml}: {e}")
        return 3
 
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        log("Error: OPENROUTER_API_KEY is required in environment.")
        return 2

    timeout_secs = safe_int(os.environ.get("OPENROUTER_TIMEOUT_SECS"), 20)

    input_path = args.input
    output_path = args.output

    # Ensure output directory exists (skip if using stdout)
    if not args.stdout:
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

    # Check output file before proceeding (skip if using stdout)
    if not args.stdout and os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not args.clobber:
        log(f"Error: Output file {output_path} already exists and is not empty. Use --clobber to overwrite.")
        return 4

    issues: List[Tuple[int, Dict[str, Any]]] = list(read_jsonl(input_path))
    if args.limit:
        issues = issues[:args.limit]
    log(f"Loaded {len(issues)} issues from {input_path}")
 
    errors: List[str] = []
 
    # Clear output file before starting (skip if using stdout)
    if not args.stdout:
        with open(output_path, "w", encoding="utf-8") as f:
            pass

    # Process with limited concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = []
        for _, issue in issues:
            futures.append(
                ex.submit(process_issue, issue, api_key, args.model, timeout_secs)
            )
 
        completed_count = 0
        total_count = len(issues)
        for fut in concurrent.futures.as_completed(futures):
            completed_count += 1
            issue_id, labels, err = fut.result()
            if err:
                msg = f"Issue {issue_id}: {err}"
                errors.append(msg)
                log(f"[WARN] {msg}")
            else:
                rec = {"id": issue_id, "labels": labels}
                if args.stdout:
                    print(json.dumps(rec, ensure_ascii=False))
                else:
                    append_jsonl(output_path, rec)
            
            log(f"Processed {completed_count}/{total_count} issues...")
 
    if args.stdout:
        log(f"Output {completed_count} records to stdout")
    else:
        log(f"Wrote {completed_count} records to {output_path}")

    if errors and not args.skip_errors:
        log(f"Encountered {len(errors)} errors. Failing. Use --skip-errors to ignore.")
        return 1

    if errors:
        log(f"Completed with {len(errors)} errors (skipped).")
    else:
        log("Completed successfully with no errors.")

    return 0


if __name__ == "__main__":
    print(build_openai_prompt('title', 'body', load_labels_from_yaml('/home/balast/CodingProjects/auto-issue-labeler-test/.github/labels.yml')))
    # sys.exit(main())