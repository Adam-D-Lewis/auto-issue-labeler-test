import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict

def create_dataset():
    """
    Merges issue data with labels and creates a Hugging Face dataset.
    """
    # Define paths
    data_dir = Path("scripts/data/export")
    issues_path = data_dir / "issues.jsonl"
    labels_path = data_dir / "generated-labels-full.jsonl"
    output_path = data_dir / "dataset.jsonl"

    # Load issues into a dictionary for easy lookup
    issues_dict = {}
    with open(issues_path, 'r') as f:
        for line in f:
            issue = json.loads(line)
            issues_dict[issue['id']] = {
                'title': issue.get('title', ''),
                'body': issue.get('body', '')
            }

    # Merge issues and labels
    merged_data = []
    with open(labels_path, 'r') as f:
        for line in f:
            label_data = json.loads(line)
            issue_id = label_data['id']
            if issue_id in issues_dict:
                issue_info = issues_dict[issue_id]
                merged_data.append({
                    'id': issue_id,
                    'title': issue_info['title'],
                    'body': issue_info['body'],
                    'labels': label_data['labels']
                })

    # Write the merged data to a new JSONL file
    with open(output_path, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')

    print(f"Merged dataset created at: {output_path}")

    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_json(str(output_path))
    print("\nHugging Face Dataset created:")
    print(hf_dataset)

    # Optional: Split into train/test/validation
    train_test_split = hf_dataset.train_test_split(test_size=0.2)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5)

    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': test_val_split['test'],
        'validation': test_val_split['train']
    })

    print("\nDataset split into train, test, and validation sets:")
    print(dataset_dict)

    # Optional: Save the dataset to disk
    dataset_dict.save_to_disk("scripts/data/export/nebari-issue-label-dataset")

if __name__ == "__main__":
    create_dataset()