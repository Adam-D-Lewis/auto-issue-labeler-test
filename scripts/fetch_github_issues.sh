#!/usr/bin/env bash
set -euo pipefail

# fetch_github_issues.sh
# Uses GitHub CLI (gh) to export issue title and body to JSONL.
# - Requires: gh CLI authenticated (gh auth login) with GITHUB_TOKEN or device auth.
# - Excludes PRs by default (GitHub returns PRs from "issue list"; we filter them out).
#
# Usage:
#   scripts/fetch_github_issues.sh --repo owner/name [--state open|closed|all] [--since ISO8601] [--label label] [--max N] [--include-prs] [--out path]
#
# Examples:
#   scripts/fetch_github_issues.sh --repo nebari-dev/nebari
#   scripts/fetch_github_issues.sh --repo nebari-dev/nebari --state all --max 1000 --out data/export/nebari_issues.jsonl
#   scripts/fetch_github_issues.sh --repo nebari-dev/nebari --since 2024-01-01T00:00:00Z
#
# Output: JSONL with fields:
#   { "id": ..., "number": ..., "title": "...", "body": "...", "state": "...", "created_at": "...", "updated_at": "...", "labels": ["..."], "url": "..." }

usage() {
  cat <<'EOF'
Usage:
  scripts/fetch_github_issues.sh --repo owner/name [options]

Options:
  --repo owner/name        Required. GitHub repository.
  --state STATE            open|closed|all (default: open)
  --since ISO8601          Filter issues updated since this time (e.g., 2024-01-01T00:00:00Z)
  --label NAME             Filter by a single label (can repeat --label)
  --max N                  Maximum number of issues to export (default: unlimited)
  --include-prs            Include pull requests (default: exclude PRs)
  --out PATH               Output JSONL path (default: data/export/issues.jsonl)
  -h, --help               Show help

Notes:
- Requires gh CLI. Install: https://cli.github.com/
- Auth: gh auth login (or set GITHUB_TOKEN). Check with: gh auth status
- gh flags used: gh issue list: -R/--repo, -s/--state, -L/--limit, -l/--label, --json, -q/--jq
EOF
}

# Defaults
REPO=""
STATE="open"
SINCE=""
LABELS=()
MAX=""
INCLUDE_PRS="false"
OUT="data/export/issues.jsonl"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="${2:-}"; shift 2 ;;
    --state)
      STATE="${2:-}"; shift 2 ;;
    --since)
      SINCE="${2:-}"; shift 2 ;;
    --label)
      LABELS+=("${2:-}"); shift 2 ;;
    --max)
      MAX="${2:-}"; shift 2 ;;
    --include-prs)
      INCLUDE_PRS="true"; shift 1 ;;
    --out)
      OUT="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "Error: --repo owner/name is required" >&2
  usage
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI not found. Install from https://cli.github.com/" >&2
  exit 1
fi

# Validate auth (but allow unauthenticated; warn about low rate limit)
if ! gh auth status >/dev/null 2>&1; then
  echo "Warning: gh not authenticated. Proceeding unauthenticated (rate limit ~60/hr). Run 'gh auth login' for higher limits." >&2
fi

# Ensure output directory
OUT_DIR="$(dirname "$OUT")"
mkdir -p "$OUT_DIR"

# Build gh issue list flags with correct syntax:
# - Use -R/--repo to select repo
# - Use -s/--state for state
# - Use -L/--limit to request a high ceiling; gh paginates internally
# - Use -l/--label for labels (repeatable)
# - Use --json for fields
# We then filter with jq locally for SINCE and for excluding PRs (no direct isPullRequest field in gh issue list JSON; we detect PRs by presence of "pull request" label unlikely reliable, so better to query REST via gh api. However gh issue list --json includes no PR marker. We'll use gh api to be precise.)
#
# To get a reliable isPullRequest flag, switch to gh api v3 endpoint:
#   GET /repos/{owner}/{repo}/issues
# which returns "pull_request" key for PRs. We will use gh api with pagination.

OWNER="${REPO%%/*}"
REPO_NAME="${REPO#*/}"

# Compose gh api URL and params
API_URL="/repos/$OWNER/$REPO_NAME/issues"
# Build query params: state, per_page=100, since (if provided), labels (comma-separated)
PARAMS=( -F state="$STATE" -F per_page=100 )
if [[ -n "$SINCE" ]]; then
  PARAMS+=( -F since="$SINCE" )
fi
if [[ ${#LABELS[@]} -gt 0 ]]; then
  IFS=',' read -r -a _unused <<< ""
  LABELS_CSV="$(printf "%s," "${LABELS[@]}")"
  LABELS_CSV="${LABELS_CSV%,}"
  PARAMS+=( -F labels="$LABELS_CSV" )
fi

# Fetch all pages using gh api --paginate
# Fields we need: number,title,body,state,created_at,updated_at,labels[].name,html_url, pull_request (to identify PRs)
TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT

if ! gh api -X GET "$API_URL" --paginate "${PARAMS[@]}" > "$TMP_JSON"; then
  echo "Error: gh api request failed for $REPO" >&2
  exit 1
fi

# Build jq filter
# - Exclude PRs unless --include-prs (PRs have a "pull_request" object)
# - Filter by SINCE already done server-side if provided; keep client-side guard
# - Apply MAX client-side
# - Include all labels: both names and full label objects for completeness
JQ_FILTER='
  map(
    select("'"$INCLUDE_PRS"'" == "true" or (has("pull_request") | not))
  )
  | map({
      id: .id,
      number: .number,
      title: (.title // ""),
      body: (.body // ""),
      state: .state,
      created_at: .created_at,
      updated_at: .updated_at,
      labels: (.labels // [] | map(.name)),
      labels_full: (.labels // []),
      url: .html_url
    })
'

# Client-side since guard if needed (for safety)
if [[ -n "$SINCE" ]]; then
  JQ_FILTER="$JQ_FILTER | map(select((.updated_at | fromdateiso8601) >= (\"$SINCE\" | fromdateiso8601)))"
fi

# Apply max limit if set
if [[ -n "$MAX" ]]; then
  JQ_FILTER="$JQ_FILTER | .[:($MAX|tonumber)]"
fi

# Write JSONL
if ! jq -r "$JQ_FILTER | .[] | @json" < "$TMP_JSON" > "$OUT"; then
  echo "Error: jq processing failed" >&2
  exit 1
fi

COUNT=$(wc -l < "$OUT" | tr -d ' ')
echo "Wrote $COUNT issues to $OUT"