#!/usr/bin/env python3
"""
Test script to load the fine-tuned model and run inference on dataset examples.
"""

import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from transformers import TextStreamer

# Load the fine-tuned model
print("Loading fine-tuned model...")
model, tokenizer = FastModel.from_pretrained(
    model_name="lora_model",  # Our saved LoRA model
    max_seq_length=2048,
    load_in_4bit=False,
)

# Set up chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma3",
)

# Load dataset for testing
print("Loading dataset...")
dataset = load_dataset("AdamDLewis/nebari-issue-label-dataset", split="train[:100]")

# System prompt
prompt = '''You are an assistant that assigns GitHub issue labels.
Return ONLY a single line of comma-separated labels from the allowed set.
Format example:
bug, enhancement, documentation
Rules:
- Choose any number of labels from the allowed set (including zero).
- Do not include any extra text, code fences, or explanations. Only the CSV line.

Allowed labels with descriptions:
- "bug": A reported error or unexpected behavior in the software.
- "enhancement": A request for a new feature or an improvement to an existing one.
- "documentation": Issues related to improving or expanding the documentation.
- "question": A user question that requires clarification or guidance.
- "maintenance": Routine tasks, refactoring, and dependency updates.
- "ci/cd": Issues related to continuous integration and deployment pipelines.
- "testing": Tasks related to creating or improving tests.
- "release": Tasks and checklists related to software releases.
- "aws": Issues specific to Amazon Web Services (AWS) deployments.
- "gcp": Issues specific to Google Cloud Platform (GCP) deployments.
- "azure": Issues specific to Microsoft Azure deployments.
- "security": Issues related to security vulnerabilities or concerns.
- "performance": Issues related to performance, cost, or resource optimization.
- "ux/ui": Issues related to user experience and user interface design.
- "configuration": Issues related to setup, configuration, or deployment settings.
- "dependency-update": Tasks related to updating third-party dependencies.'''

def calculate_score(actual_labels, predicted_labels_str):
    """Calculate the score as number of correct predictions over total ground truth labels."""
    if predicted_labels_str == "No model response found" or not predicted_labels_str.strip():
        return 0, len(actual_labels), 0.0
    
    # Parse predicted labels
    predicted_labels = [label.strip() for label in predicted_labels_str.split(',') if label.strip()]
    actual_labels_set = set(actual_labels)
    predicted_labels_set = set(predicted_labels)
    
    # Count correct predictions (intersection)
    correct = len(actual_labels_set.intersection(predicted_labels_set))
    total_actual = len(actual_labels_set)
    score = correct / total_actual if total_actual > 0 else 0.0
    
    return correct, total_actual, score

def test_model_on_example(example_idx):
    """Test the model on a specific example from the dataset."""
    example = dataset[example_idx]
    
    print(f"\n{'='*80}")
    print(f"Testing Example {example_idx}")
    print(f"{'='*80}")
    
    # Print the original issue
    print(f"\nOriginal Issue:")
    print(f"Title: {example['title']}")
    print(f"Body: {example['body'][:500]}{'...' if len(example['body']) > 500 else ''}")
    print(f"\nActual Labels: {', '.join(example['labels'])}")
    
    # Prepare messages for inference
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f"Issue title: {example['title']}\nIssue body: {example['body']}"}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')
    
    print(f"\nModel Input:")
    print(f"Title: {example['title']}")
    print(f"Body: {example['body'][:200]}{'...' if len(example['body']) > 200 else ''}")
    
    # Generate prediction
    print(f"\nGenerating prediction...")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,  # Lower temperature for more consistent results
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the full generated response
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract just the model's response (after "model\n")
    if "\nmodel\n" in generated_text:
        predicted_labels = generated_text.split("\nmodel\n")[-1].strip()
    elif "model\n" in generated_text:
        predicted_labels = generated_text.split("model\n")[-1].strip()
    else:
        predicted_labels = "No model response found"
    
    # Clean up the prediction if it contains extra tokens
    if predicted_labels.endswith("<end_of_turn>"):
        predicted_labels = predicted_labels.replace("<end_of_turn>", "").strip()
    
    # Calculate score
    correct, total_actual, score = calculate_score(example['labels'], predicted_labels)
    
    print(f"\nResults:")
    print(f"Actual Labels:    {', '.join(example['labels'])}")
    print(f"Predicted Labels: {predicted_labels}")
    print(f"Score: {correct}/{total_actual} = {score:.2%}")
    
    return predicted_labels, correct, total_actual, score

def main():
    """Run tests on multiple examples."""
    print("Testing fine-tuned GitHub issue labeler model\n")
    
    # Test on multiple examples
    test_indices = [0, 5, 10, 15, 20]
    
    results = []
    total_correct = 0
    total_possible = 0
    
    for idx in test_indices:
        try:
            predicted, correct, total_actual, score = test_model_on_example(idx)
            results.append({
                'index': idx,
                'actual': ', '.join(dataset[idx]['labels']),
                'predicted': predicted,
                'correct': correct,
                'total_actual': total_actual,
                'score': score
            })
            total_correct += correct
            total_possible += total_actual
        except Exception as e:
            print(f"Error testing example {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Index':<6} {'Actual':<25} {'Predicted':<25} {'Score':<10}")
    print(f"{'-'*6} {'-'*25} {'-'*25} {'-'*10}")
    
    for result in results:
        actual = result['actual'][:23] + ".." if len(result['actual']) > 25 else result['actual']
        predicted = result['predicted'][:23] + ".." if len(result['predicted']) > 25 else result['predicted']
        score_str = f"{result['correct']}/{result['total_actual']}"
        print(f"{result['index']:<6} {actual:<25} {predicted:<25} {score_str:<10}")
    
    # Overall statistics
    overall_score = total_correct / total_possible if total_possible > 0 else 0.0
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Total Correct Labels: {total_correct}")
    print(f"Total Possible Labels: {total_possible}")
    print(f"Overall Accuracy: {overall_score:.2%}")
    print(f"Average Score per Example: {sum(r['score'] for r in results) / len(results):.2%}")
    
    # Score distribution
    perfect_scores = len([r for r in results if r['score'] == 1.0])
    partial_scores = len([r for r in results if 0 < r['score'] < 1.0])
    zero_scores = len([r for r in results if r['score'] == 0.0])
    
    print(f"\nScore Distribution:")
    print(f"Perfect matches (100%): {perfect_scores}/{len(results)}")
    print(f"Partial matches (>0%): {partial_scores}/{len(results)}")
    print(f"No matches (0%): {zero_scores}/{len(results)}")

if __name__ == "__main__":
    main()