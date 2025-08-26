import json
import argparse
from pathlib import Path
from collections import Counter

def get_label_counts(jsonl_file: Path):
    """Reads a JSONL file and returns a Counter of label occurrences."""
    labels = Counter()
    with open(jsonl_file, "r") as f:
        for line in f:
            issue = json.loads(line)
            for label in issue.get("labels", []):
                if isinstance(label, dict):
                    label_name = label.get("name")
                    if label_name:
                        labels[label_name] += 1
                else:
                    labels[str(label)] += 1
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get unique labels from a JSONL file of GitHub issues.")
    parser.add_argument("--sort-by-count", action="store_true", help="Sort labels by the number of issues they appear in.")
    args = parser.parse_args()

    issues_file = Path(__file__).parent / "data/export/issues.jsonl"
    if not issues_file.exists():
        issues_file = Path("/tmp/issues-small.jsonl")
        if not issues_file.exists():
            print(f"Error: Default issues files not found.")
            print("Please run ./scripts/fetch_github_issues.sh first.")
            exit()

    label_counts = get_label_counts(issues_file)
    
    print("Unique labels found:")
    if args.sort_by_count:
        # Sort by count (descending), then alphabetically for ties
        sorted_labels = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        for label, count in sorted_labels:
            print(f"- {label} ({count})")
    else:
        # Sort alphabetically
        for label in sorted(label_counts.keys()):
            print(f"- {label}")