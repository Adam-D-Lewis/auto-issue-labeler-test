#!/usr/bin/env python3
"""
Comparison script to test both the original base model and fine-tuned model.
"""

import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

# Load dataset for testing
print("Loading dataset...")
dataset = load_dataset("AdamDLewis/nebari-issue-label-dataset", split="test")

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
    
    # Calculate score
    correct, total_actual, score = calculate_score(example['labels'], predicted_labels)
    
    return {
        'model_name': model_name,
        'predicted': predicted_labels,
        'correct': correct,
        'total_actual': total_actual,
        'score': score,
        'actual': example['labels']
    }

def main():
    print("="*80)
    print("MODEL COMPARISON: Base vs Fine-tuned")
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
    
    # Test examples - use specific indices for manageable testing
    test_indices = list(range(len(dataset)))

    print(f"\nTesting {len(test_indices)} examples...\n")
    
    all_results = []
    base_total_correct = 0
    base_total_possible = 0
    ft_total_correct = 0
    ft_total_possible = 0
    
    for idx in test_indices:
        example = dataset[idx]
        print(f"Example {idx}: {example['title'][:60]}...")
        print(f"Actual Labels: {', '.join(example['labels'])}")
        
        # Test base model
        base_result = test_model(base_model, base_tokenizer, idx, "Base")
        print(f"Base Predicted:     {base_result['predicted']}")
        print(f"Base Score:         {base_result['correct']}/{base_result['total_actual']} = {base_result['score']:.2%}")
        
        # Test fine-tuned model
        ft_result = test_model(ft_model, ft_tokenizer, idx, "Fine-tuned")
        print(f"Fine-tuned Predicted: {ft_result['predicted']}")
        print(f"Fine-tuned Score:     {ft_result['correct']}/{ft_result['total_actual']} = {ft_result['score']:.2%}")
        
        base_total_correct += base_result['correct']
        base_total_possible += base_result['total_actual']
        ft_total_correct += ft_result['correct']
        ft_total_possible += ft_result['total_actual']
        
        all_results.append({
            'index': idx,
            'actual': ', '.join(example['labels']),
            'base': base_result,
            'ft': ft_result
        })
        print("-" * 80)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Example':<8} {'Actual':<20} {'Base Score':<12} {'FT Score':<12} {'Improvement':<12}")
    print(f"{'-'*8} {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    improvements = []
    for result in all_results:
        actual_short = result['actual'][:18] + ".." if len(result['actual']) > 20 else result['actual']
        base_score = f"{result['base']['correct']}/{result['base']['total_actual']}"
        ft_score = f"{result['ft']['correct']}/{result['ft']['total_actual']}"
        improvement = result['ft']['score'] - result['base']['score']
        improvements.append(improvement)
        improvement_str = f"{improvement:+.2%}"
        
        print(f"{result['index']:<8} {actual_short:<20} {base_score:<12} {ft_score:<12} {improvement_str:<12}")
    
    # Overall statistics
    base_accuracy = base_total_correct / base_total_possible if base_total_possible > 0 else 0.0
    ft_accuracy = ft_total_correct / ft_total_possible if ft_total_possible > 0 else 0.0
    overall_improvement = ft_accuracy - base_accuracy
    
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Base Model:")
    print(f"  Total Correct: {base_total_correct}/{base_total_possible}")
    print(f"  Accuracy: {base_accuracy:.2%}")
    print(f"  Perfect Matches: {len([r for r in all_results if r['base']['score'] == 1.0])}/{len(all_results)}")
    
    print(f"\nFine-tuned Model:")
    print(f"  Total Correct: {ft_total_correct}/{ft_total_possible}")
    print(f"  Accuracy: {ft_accuracy:.2%}")
    print(f"  Perfect Matches: {len([r for r in all_results if r['ft']['score'] == 1.0])}/{len(all_results)}")
    
    print(f"\nImprovement:")
    print(f"  Accuracy Gain: {overall_improvement:+.2%}")
    print(f"  Average Per-Example Improvement: {sum(improvements)/len(improvements):+.2%}")
    
    # Detailed comparison
    print(f"\n{'='*80}")
    print("DETAILED PREDICTIONS")
    print(f"{'='*80}")
    
    for result in all_results:
        example = dataset[result['index']]
        print(f"\nExample {result['index']}: {example['title']}")
        print(f"Actual:     {result['actual']}")
        print(f"Base:       {result['base']['predicted']}")
        print(f"Fine-tuned: {result['ft']['predicted']}")
    
    # Calculate comprehensive statistics
    def calculate_model_stats(results, model_key):
        """Calculate comprehensive statistics for a model."""
        scores = [r[model_key]['score'] for r in results]
        predicted_labels = []
        actual_labels = []
        
        # STT-style error metrics
        total_insertions = 0  # False positives (extra labels)
        total_deletions = 0   # False negatives (missing labels)
        total_substitutions = 0  # Examples with both insertions and deletions
        total_correct_labels = 0
        total_predicted_labels = 0
        total_actual_labels = 0
        
        for r in results:
            # Parse predicted labels
            pred_str = r[model_key]['predicted']
            if pred_str != "No model response found" and pred_str.strip():
                pred_labels = [label.strip() for label in pred_str.split(',') if label.strip()]
            else:
                pred_labels = []
            
            actual_set = set(r['actual'].split(', '))
            pred_set = set(pred_labels)
            
            # Calculate per-example errors
            insertions = len(pred_set - actual_set)  # Extra labels predicted
            deletions = len(actual_set - pred_set)   # Missing labels
            correct = len(actual_set & pred_set)     # Correct labels
            
            total_insertions += insertions
            total_deletions += deletions
            total_correct_labels += correct
            total_predicted_labels += len(pred_set)
            total_actual_labels += len(actual_set)
            
            # Substitutions: examples with both insertions and deletions
            if insertions > 0 and deletions > 0:
                total_substitutions += 1
            
            predicted_labels.extend(pred_labels)
            actual_labels.extend(r['actual'].split(', '))
        
        # Calculate error rates
        total_operations = total_correct_labels + total_insertions + total_deletions
        insertion_rate = total_insertions / total_operations if total_operations > 0 else 0
        deletion_rate = total_deletions / total_operations if total_operations > 0 else 0
        error_rate = (total_insertions + total_deletions) / total_operations if total_operations > 0 else 0
        
        # Calculate statistics
        perfect_matches = len([s for s in scores if s == 1.0])
        partial_matches = len([s for s in scores if 0 < s < 1.0])
        no_matches = len([s for s in scores if s == 0.0])
        
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Label frequency analysis
        from collections import Counter
        pred_counter = Counter(predicted_labels)
        actual_counter = Counter(actual_labels)
        
        # Precision and Recall per label
        unique_labels = set(predicted_labels + actual_labels)
        label_stats = {}
        
        for label in unique_labels:
            pred_count = pred_counter.get(label, 0)
            actual_count = actual_counter.get(label, 0)
            
            # Calculate how many times this label was correctly predicted
            correct_predictions = 0
            for r in results:
                actual_set = set(r['actual'].split(', '))
                pred_str = r[model_key]['predicted']
                if pred_str != "No model response found" and pred_str.strip():
                    pred_set = set([label.strip() for label in pred_str.split(',') if label.strip()])
                else:
                    pred_set = set()
                
                if label in actual_set and label in pred_set:
                    correct_predictions += 1
            
            precision = correct_predictions / pred_count if pred_count > 0 else 0
            recall = correct_predictions / actual_count if actual_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            label_stats[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pred_count': pred_count,
                'actual_count': actual_count,
                'correct': correct_predictions
            }
        
        return {
            'scores': scores,
            'perfect_matches': perfect_matches,
            'partial_matches': partial_matches,
            'no_matches': no_matches,
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'total_predicted': len(predicted_labels),
            'unique_predicted': len(set(predicted_labels)),
            'avg_labels_per_example': len(predicted_labels) / len(results) if results else 0,
            'label_stats': label_stats,
            # STT-style error metrics
            'total_insertions': total_insertions,
            'total_deletions': total_deletions, 
            'total_substitutions': total_substitutions,
            'total_correct_labels': total_correct_labels,
            'insertion_rate': insertion_rate,
            'deletion_rate': deletion_rate,
            'error_rate': error_rate,
            'total_actual_labels': total_actual_labels,
            'total_predicted_labels': total_predicted_labels
        }
    
    # Calculate statistics for both models
    base_stats = calculate_model_stats(all_results, 'base')
    ft_stats = calculate_model_stats(all_results, 'ft')
    
    # Print comprehensive summary statistics
    print(f"\n{'='*100}")
    print("COMPREHENSIVE MODEL STATISTICS")
    print(f"{'='*100}")
    
    # Score distribution comparison
    print(f"\n{'METRIC':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'IMPROVEMENT':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    print(f"{'Perfect Matches':<30} {base_stats['perfect_matches']}/{len(all_results):<20} {ft_stats['perfect_matches']}/{len(all_results):<20} {ft_stats['perfect_matches'] - base_stats['perfect_matches']:+d}")
    print(f"{'Partial Matches':<30} {base_stats['partial_matches']}/{len(all_results):<20} {ft_stats['partial_matches']}/{len(all_results):<20} {ft_stats['partial_matches'] - base_stats['partial_matches']:+d}")
    print(f"{'No Matches':<30} {base_stats['no_matches']}/{len(all_results):<20} {ft_stats['no_matches']}/{len(all_results):<20} {ft_stats['no_matches'] - base_stats['no_matches']:+d}")
    print(f"{'Average Score':<30} {f'{base_stats["avg_score"]:.2%}':<20} {f'{ft_stats["avg_score"]:.2%}':<20} {ft_stats['avg_score'] - base_stats['avg_score']:+.2%}")
    print(f"{'Min Score':<30} {f'{base_stats["min_score"]:.2%}':<20} {f'{ft_stats["min_score"]:.2%}':<20} {ft_stats['min_score'] - base_stats['min_score']:+.2%}")
    print(f"{'Max Score':<30} {f'{base_stats["max_score"]:.2%}':<20} {f'{ft_stats["max_score"]:.2%}':<20} {ft_stats['max_score'] - base_stats['max_score']:+.2%}")
    
    # Prediction behavior comparison
    print(f"\n{'PREDICTION BEHAVIOR':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'DIFFERENCE':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    print(f"{'Total Labels Predicted':<30} {base_stats['total_predicted']:<20} {ft_stats['total_predicted']:<20} {ft_stats['total_predicted'] - base_stats['total_predicted']:+d}")
    print(f"{'Unique Labels Used':<30} {base_stats['unique_predicted']:<20} {ft_stats['unique_predicted']:<20} {ft_stats['unique_predicted'] - base_stats['unique_predicted']:+d}")
    print(f"{'Avg Labels per Example':<30} {f'{base_stats["avg_labels_per_example"]:.1f}':<20} {f'{ft_stats["avg_labels_per_example"]:.1f}':<20} {ft_stats['avg_labels_per_example'] - base_stats['avg_labels_per_example']:+.1f}")
    
    # STT-style error analysis
    print(f"\n{'ERROR ANALYSIS (STT-STYLE)':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'DIFFERENCE':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    print(f"{'Label Insertions (Extra)':<30} {base_stats['total_insertions']:<20} {ft_stats['total_insertions']:<20} {ft_stats['total_insertions'] - base_stats['total_insertions']:+d}")
    print(f"{'Label Deletions (Missing)':<30} {base_stats['total_deletions']:<20} {ft_stats['total_deletions']:<20} {ft_stats['total_deletions'] - base_stats['total_deletions']:+d}")
    print(f"{'Examples with Mixed Errors':<30} {base_stats['total_substitutions']:<20} {ft_stats['total_substitutions']:<20} {ft_stats['total_substitutions'] - base_stats['total_substitutions']:+d}")
    print(f"{'Insertion Rate':<30} {f'{base_stats["insertion_rate"]:.2%}':<20} {f'{ft_stats["insertion_rate"]:.2%}':<20} {ft_stats['insertion_rate'] - base_stats['insertion_rate']:+.2%}")
    print(f"{'Deletion Rate':<30} {f'{base_stats["deletion_rate"]:.2%}':<20} {f'{ft_stats["deletion_rate"]:.2%}':<20} {ft_stats['deletion_rate'] - base_stats['deletion_rate']:+.2%}")
    print(f"{'Overall Error Rate':<30} {f'{base_stats["error_rate"]:.2%}':<20} {f'{ft_stats["error_rate"]:.2%}':<20} {ft_stats['error_rate'] - base_stats['error_rate']:+.2%}")
    
    # Label accuracy breakdown
    print(f"\n{'LABEL ACCURACY BREAKDOWN':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'DIFFERENCE':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    print(f"{'Correct Labels':<30} {base_stats['total_correct_labels']:<20} {ft_stats['total_correct_labels']:<20} {ft_stats['total_correct_labels'] - base_stats['total_correct_labels']:+d}")
    print(f"{'Labels in Ground Truth':<30} {base_stats['total_actual_labels']:<20} {ft_stats['total_actual_labels']:<20} {ft_stats['total_actual_labels'] - base_stats['total_actual_labels']:+d}")
    print(f"{'Labels Predicted':<30} {base_stats['total_predicted_labels']:<20} {ft_stats['total_predicted_labels']:<20} {ft_stats['total_predicted_labels'] - base_stats['total_predicted_labels']:+d}")
    
    # Per-label performance comparison
    print(f"\n{'='*100}")
    print("PER-LABEL PERFORMANCE COMPARISON")
    print(f"{'='*100}")
    
    all_labels = set(list(base_stats['label_stats'].keys()) + list(ft_stats['label_stats'].keys()))
    sorted_labels = sorted(all_labels)
    
    print(f"\n{'LABEL':<20} {'BASE F1':<12} {'FT F1':<12} {'BASE PREC':<12} {'FT PREC':<12} {'BASE REC':<12} {'FT REC':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for label in sorted_labels:
        base_f1 = base_stats['label_stats'].get(label, {}).get('f1', 0)
        ft_f1 = ft_stats['label_stats'].get(label, {}).get('f1', 0)
        base_prec = base_stats['label_stats'].get(label, {}).get('precision', 0)
        ft_prec = ft_stats['label_stats'].get(label, {}).get('precision', 0)
        base_rec = base_stats['label_stats'].get(label, {}).get('recall', 0)
        ft_rec = ft_stats['label_stats'].get(label, {}).get('recall', 0)
        
        print(f"{label:<20} {f'{base_f1:.2f}':<12} {f'{ft_f1:.2f}':<12} {f'{base_prec:.2f}':<12} {f'{ft_prec:.2f}':<12} {f'{base_rec:.2f}':<12} {f'{ft_rec:.2f}':<12}")
    
    # Calculate macro averages
    base_macro_f1 = sum([stats.get('f1', 0) for stats in base_stats['label_stats'].values()]) / len(base_stats['label_stats']) if base_stats['label_stats'] else 0
    ft_macro_f1 = sum([stats.get('f1', 0) for stats in ft_stats['label_stats'].values()]) / len(ft_stats['label_stats']) if ft_stats['label_stats'] else 0
    
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'MACRO AVERAGE':<20} {f'{base_macro_f1:.2f}':<12} {f'{ft_macro_f1:.2f}':<12} {'':12} {'':12} {'':12} {'':12}")
    
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Fine-tuning Impact:")
    print(f"  • Macro F1 Score: {base_macro_f1:.3f} → {ft_macro_f1:.3f} ({ft_macro_f1 - base_macro_f1:+.3f})")
    print(f"  • Perfect Matches: {base_stats['perfect_matches']}/{len(all_results)} → {ft_stats['perfect_matches']}/{len(all_results)} ({ft_stats['perfect_matches'] - base_stats['perfect_matches']:+d})")
    print(f"  • Prediction Efficiency: {base_stats['avg_labels_per_example']:.1f} → {ft_stats['avg_labels_per_example']:.1f} labels/example ({base_stats['avg_labels_per_example'] - ft_stats['avg_labels_per_example']:+.1f})")
    print(f"  • Label Insertions (Extra): {base_stats['total_insertions']} → {ft_stats['total_insertions']} ({ft_stats['total_insertions'] - base_stats['total_insertions']:+d})")
    print(f"  • Label Deletions (Missing): {base_stats['total_deletions']} → {ft_stats['total_deletions']} ({ft_stats['total_deletions'] - base_stats['total_deletions']:+d})")
    print(f"  • Overall Error Rate: {base_stats['error_rate']:.1%} → {ft_stats['error_rate']:.1%} ({ft_stats['error_rate'] - base_stats['error_rate']:+.1%})")
    
    # Analysis of error improvements
    insertion_improvement = base_stats['total_insertions'] - ft_stats['total_insertions']
    deletion_change = ft_stats['total_deletions'] - base_stats['total_deletions']
    error_improvement = base_stats['error_rate'] - ft_stats['error_rate']
    
    print(f"\nError Analysis:")
    if insertion_improvement > 0:
        print(f"  ✅ Reduced over-predictions by {insertion_improvement} labels ({insertion_improvement/base_stats['total_insertions']:.1%})")
    elif insertion_improvement < 0:
        print(f"  ⚠️  Increased over-predictions by {abs(insertion_improvement)} labels")
    
    if deletion_change < 0:
        print(f"  ✅ Reduced under-predictions by {abs(deletion_change)} labels")
    elif deletion_change > 0:
        print(f"  ⚠️  Increased under-predictions by {deletion_change} labels")
    
    if error_improvement > 0:
        print(f"  ✅ Overall error rate improved by {error_improvement:.1%}")
    elif error_improvement < 0:
        print(f"  ⚠️  Overall error rate worsened by {abs(error_improvement):.1%}")
    
    if ft_macro_f1 > base_macro_f1:
        print(f"  🎯 Fine-tuning IMPROVED overall performance")
    elif ft_macro_f1 < base_macro_f1:
        print(f"  ⚠️  Fine-tuning DECREASED overall performance")
    else:
        print(f"  ➡️  Fine-tuning had NEUTRAL impact on overall performance")

if __name__ == "__main__":
    main()