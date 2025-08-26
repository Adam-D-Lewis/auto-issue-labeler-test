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

    # Upload to Hugging Face Hub
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("\nHugging Face token found. Uploading dataset to the Hub...")
        repo_name = "nebari-issue-label-dataset"
        dataset_dict.push_to_hub(repo_name, private=True, token=hf_token)
        print(f"Dataset successfully uploaded to the Hugging Face Hub as '{repo_name}'.")
    else:
        print("\nHF_TOKEN environment variable not set. Skipping upload to Hugging Face Hub.")
        print("To upload, set the HF_TOKEN environment variable with your Hugging Face API token.")

    # Optional: Save the dataset to disk
    dataset_dict.save_to_disk("scripts/data/export/nebari-issue-label-dataset")

if __name__ == "__main__":
    # Ensure the necessary libraries are installed
    try:
        import datasets
        import huggingface_hub
    except ImportError as e:
        print(f"A required library is not installed: {e.name}")
        print("Please ensure both 'datasets' and 'huggingface-hub' are listed in your pixi.toml and installed.")
        exit(1)
    create_dataset()