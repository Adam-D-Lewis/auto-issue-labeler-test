import json
import argparse
from pathlib import Path

def filter_issue_fields(jsonl_file: Path, fields_to_keep: list[str]):
    """Reads a JSONL file and yields issues with only the specified fields."""
    with open(jsonl_file, "r") as f:
        for line in f:
            issue = json.loads(line)
            filtered_issue = {
                field: issue.get(field) for field in fields_to_keep
            }
            yield filtered_issue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter fields from a JSONL file of GitHub issues.")
    parser.add_argument("--fields", nargs="+", default=["title", "body"], help="The fields to keep in the output.")
    args = parser.parse_args()

    issues_file = Path(__file__).parent / "data/export/issues.jsonl"
    if not issues_file.exists():
        issues_file = Path("/tmp/issues-small.jsonl")
        if not issues_file.exists():
            print(f"Error: Default issues files not found.")
            print("Please run ./scripts/fetch_github_issues.sh first.")
            exit()

    for issue in filter_issue_fields(issues_file, args.fields):
        print(json.dumps(issue))