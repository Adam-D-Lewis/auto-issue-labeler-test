#!/usr/bin/env python3
"""
Model evaluation script that loads both base and fine-tuned models,
runs inference on test dataset, and saves results for later analysis.
"""

import torch
import json
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from pathlib import Path

# Load dataset for testing
print("Loading dataset...")
dataset = load_dataset("AdamDLewis/nebari-issue-label-dataset", split="test")
print(f"Test dataset size: {len(dataset)} examples")

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

def test_model(model, tokenizer, example_idx, model_name):
    """Test a model on a specific example."""
    example = dataset[example_idx]
    
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
    
    # Generate prediction
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the full generated response
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract just the model's response
    if "\nmodel\n" in generated_text:
        predicted_labels = generated_text.split("\nmodel\n")[-1].strip()
    elif "model\n" in generated_text:
        predicted_labels = generated_text.split("model\n")[-1].strip()
    else:
        predicted_labels = "No model response found"
    
    # Clean up the prediction
    if predicted_labels.endswith("<end_of_turn>"):
        predicted_labels = predicted_labels.replace("<end_of_turn>", "").strip()
    
    return {
        'index': example_idx,
        'title': example['title'],
        'body': example['body'],
        'actual_labels': example['labels'],
        'predicted_labels': predicted_labels,
        'model_name': model_name
    }

def main():
    print("="*80)
    print("MODEL EVALUATION: Generating predictions for comparison")
    print("="*80)
    
    # Load base model
    print("\nLoading base model...")
    base_model, base_tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        max_seq_length=2048,
        load_in_4bit=False,
    )
    base_tokenizer = get_chat_template(base_tokenizer, chat_template="gemma3")
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = FastModel.from_pretrained(
        model_name="lora_model",
        max_seq_length=2048,
        load_in_4bit=False,
    )
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="gemma3")
    
    # Test examples - use all examples for comprehensive evaluation
    test_indices = list(range(len(dataset)))
    print(f"\nEvaluating {len(test_indices)} examples...")
    
    all_results = []
    
    for idx in test_indices:
        example = dataset[idx]
        print(f"Processing example {idx + 1}/{len(test_indices)}: {example['title'][:60]}...")
        
        # Test base model
        base_result = test_model(base_model, base_tokenizer, idx, "base")
        all_results.append(base_result)
        
        # Test fine-tuned model
        ft_result = test_model(ft_model, ft_tokenizer, idx, "fine_tuned")
        all_results.append(ft_result)
    
    # Save results to JSON file
    results_file = Path("model_evaluation_results.json")
    print(f"\nSaving results to {results_file}...")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Evaluation complete! Results saved to {results_file}")
    print(f"Total predictions generated: {len(all_results)}")
    print(f"Base model predictions: {len([r for r in all_results if r['model_name'] == 'base'])}")
    print(f"Fine-tuned model predictions: {len([r for r in all_results if r['model_name'] == 'fine_tuned'])}")

if __name__ == "__main__":
    main()