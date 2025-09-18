#!/usr/bin/env python3
"""
Results analysis script that loads evaluation results and performs
comprehensive comparison between base and fine-tuned models.
"""

import json
import re
from pathlib import Path
from collections import Counter

def check_prediction_format(prediction_str):
    """Check if a prediction string is properly formatted. Returns (is_valid, reason)"""
    if prediction_str == "No model response found":
        return False, "no_response"
    
    if not prediction_str or not prediction_str.strip():
        return False, "empty_response"
    
    # Check for obvious formatting issues
    if '\n' in prediction_str:
        return False, "contains_newlines"
    if '`' in prediction_str:
        return False, "contains_backticks"
    if '[' in prediction_str or ']' in prediction_str:
        return False, "contains_brackets"
    if '{' in prediction_str or '}' in prediction_str:
        return False, "contains_braces"
    if len(prediction_str) > 200:  # Reasonable max length for CSV labels
        return False, "too_long"
    
    return True, "valid"

def normalize_labels(label_string_or_list):
    """Normalize labels by sorting them alphabetically with strict validation"""
    # Valid labels from the prompt
    valid_labels = {
        "bug", "enhancement", "documentation", "question", "maintenance",
        "ci/cd", "testing", "release", "aws", "gcp", "azure", "security",
        "performance", "ux/ui", "configuration", "dependency-update"
    }
    
    # Strict regex for valid label format: letters, numbers, hyphens, forward slashes only
    # No newlines, backticks, brackets, parentheses, etc.
    valid_label_pattern = re.compile(r'^[a-zA-Z0-9/_-]+$')
    
    if isinstance(label_string_or_list, list):
        # If it's already a list, filter and sort it
        filtered_labels = []
        for label in label_string_or_list:
            if isinstance(label, str):
                cleaned_label = label.strip().lower()
                # Check if it matches the strict pattern and is in valid set
                if valid_label_pattern.match(cleaned_label) and cleaned_label in valid_labels:
                    filtered_labels.append(cleaned_label)
        return sorted(filtered_labels)
    elif isinstance(label_string_or_list, str):
        # If it's a string, parse, filter and sort
        if not label_string_or_list or not label_string_or_list.strip():
            return []
        
        # Split by comma and clean up each label
        labels = []
        for label in label_string_or_list.split(','):
            cleaned_label = label.strip().lower()
            # Check strict pattern and valid set
            if (cleaned_label and 
                valid_label_pattern.match(cleaned_label) and 
                cleaned_label in valid_labels):
                labels.append(cleaned_label)
        return sorted(labels)
    else:
        return []

def calculate_score(actual_labels, predicted_labels_str):
    """Calculate Jaccard similarity, micro-average precision, and micro-average recall scores."""
    # Normalize actual labels first
    normalized_actual = normalize_labels(actual_labels)
    total_actual = len(normalized_actual)
    
    if predicted_labels_str == "No model response found" or not predicted_labels_str.strip():
        # If no prediction, return 0 scores
        return 0, total_actual, 0, 0, total_actual, 0.0
    
    # Normalize both actual and predicted labels (sort alphabetically)
    normalized_predicted = normalize_labels(predicted_labels_str)
    
    actual_labels_set = set(normalized_actual)
    predicted_labels_set = set(normalized_predicted)
    
    # Calculate Jaccard similarity: intersection / union
    intersection = len(actual_labels_set.intersection(predicted_labels_set))
    union = len(actual_labels_set.union(predicted_labels_set))
    jaccard_score = intersection / union if union > 0 else 1.0  # 1.0 if both sets are empty
    
    # Calculate components for micro-average precision and recall
    correct_predictions = intersection
    total_predictions = len(predicted_labels_set)
    
    return intersection, union, correct_predictions, total_predictions, total_actual, jaccard_score

def load_results(results_file="model_evaluation_results.json"):
    """Load evaluation results from JSON file."""
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_path, 'r') as f:
        return json.load(f)

def organize_results(raw_results):
    """Organize raw results by example index and model type."""
    organized = {}
    
    for result in raw_results:
        idx = result['index']
        model_name = result['model_name']
        
        if idx not in organized:
            organized[idx] = {
                'title': result['title'],
                'body': result['body'],
                'actual_labels': result['actual_labels']
            }
        
        # Add model-specific results
        organized[idx][model_name] = {
            'predicted_labels': result['predicted_labels']
        }
    
    return organized

def calculate_model_stats(results, model_key):
    """Calculate comprehensive statistics for a model."""
    scores = []
    predicted_labels = []
    actual_labels = []
    
    # STT-style error metrics
    total_insertions = 0  # False positives (extra labels)
    total_deletions = 0   # False negatives (missing labels)
    total_substitutions = 0  # Examples with both insertions and deletions
    total_correct_labels = 0
    total_predicted_labels = 0
    total_actual_labels = 0
    
    for example_data in results.values():
        if model_key not in example_data:
            continue
            
        actual = example_data['actual_labels']
        predicted_str = example_data[model_key]['predicted_labels']
        
        # Calculate score for this example
        intersection, union, correct, total_predictions, total_actual, jaccard_score = calculate_score(actual, predicted_str)
        scores.append(jaccard_score)
        
        # Parse and normalize predicted labels
        pred_labels = normalize_labels(predicted_str)
        
        # Normalize actual labels
        actual_normalized = normalize_labels(actual)
        
        actual_set = set(actual_normalized)
        pred_set = set(pred_labels)
        
        # Calculate per-example errors
        insertions = len(pred_set - actual_set)  # Extra labels predicted
        deletions = len(actual_set - pred_set)   # Missing labels
        correct_labels = len(actual_set & pred_set)     # Correct labels
        
        total_insertions += insertions
        total_deletions += deletions
        total_correct_labels += correct_labels
        total_predicted_labels += len(pred_set)
        total_actual_labels += len(actual_set)
        
        # Substitutions: examples with both insertions and deletions
        if insertions > 0 and deletions > 0:
            total_substitutions += 1
        
        predicted_labels.extend(pred_labels)
        actual_labels.extend(actual_normalized)
    
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
        for example_data in results.values():
            if model_key not in example_data:
                continue
                
            actual_set = set(normalize_labels(example_data['actual_labels']))
            pred_str = example_data[model_key]['predicted_labels']
            pred_set = set(normalize_labels(pred_str))
            
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
        'avg_labels_per_example': len(predicted_labels) / len(scores) if scores else 0,
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

def analyze_formatting_quality(organized_results):
    """Analyze the formatting quality of predictions for each model"""
    models = ['base', 'fine_tuned']
    formatting_stats = {}
    
    # Valid labels from the prompt
    valid_labels = {
        "bug", "enhancement", "documentation", "question", "maintenance",
        "ci/cd", "testing", "release", "aws", "gcp", "azure", "security",
        "performance", "ux/ui", "configuration", "dependency-update"
    }
    
    for model in models:
        total_predictions = 0
        valid_predictions = 0
        format_issues = Counter()
        
        # For valid label adherence analysis
        valid_format_predictions = 0
        valid_labels_only = 0
        invalid_labels_used = Counter()
        
        for example_data in organized_results.values():
            if model not in example_data:
                continue
                
            prediction_str = example_data[model]['predicted_labels']
            total_predictions += 1
            
            is_valid, reason = check_prediction_format(prediction_str)
            if is_valid:
                valid_predictions += 1
                valid_format_predictions += 1
                
                # Analyze label adherence for well-formatted predictions
                predicted_labels = normalize_labels(prediction_str)
                predicted_labels_set = set(predicted_labels)
                
                # Check if all predicted labels are in the valid set
                invalid_predicted = predicted_labels_set - valid_labels
                if len(invalid_predicted) == 0:
                    valid_labels_only += 1
                else:
                    # Count which invalid labels were used
                    for invalid_label in invalid_predicted:
                        invalid_labels_used[invalid_label] += 1
            else:
                format_issues[reason] += 1
        
        formatting_stats[model] = {
            'total': total_predictions,
            'valid': valid_predictions,
            'invalid': total_predictions - valid_predictions,
            'valid_rate': valid_predictions / total_predictions if total_predictions > 0 else 0,
            'issues': dict(format_issues),
            # Label adherence stats
            'valid_format_predictions': valid_format_predictions,
            'valid_labels_only': valid_labels_only,
            'valid_labels_rate': valid_labels_only / valid_format_predictions if valid_format_predictions > 0 else 0,
            'invalid_labels_used': dict(invalid_labels_used)
        }
    
    return formatting_stats

def main():
    print("="*80)
    print("MODEL COMPARISON ANALYSIS")
    print("="*80)
    
    # Load results
    print("Loading evaluation results...")
    try:
        raw_results = load_results()
        print(f"Loaded {len(raw_results)} total predictions")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run evaluate_models.py first to generate results.")
        return
    
    # Organize results by example
    organized_results = organize_results(raw_results)
    print(f"Organized into {len(organized_results)} examples")
    
    # Analyze formatting quality
    formatting_stats = analyze_formatting_quality(organized_results)
    
    # Report formatting quality
    print(f"\n{'='*80}")
    print("PREDICTION FORMATTING QUALITY")
    print(f"{'='*80}")
    
    print(f"{'MODEL':<15} {'TOTAL':<8} {'VALID':<8} {'INVALID':<8} {'VALID %':<10} {'MAIN ISSUES'}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*30}")
    
    for model_name, stats in formatting_stats.items():
        model_display = "Base" if model_name == "base" else "Fine-tuned"
        valid_pct = f"{stats['valid_rate']:.1%}"
        
        # Get top formatting issues
        top_issues = sorted(stats['issues'].items(), key=lambda x: x[1], reverse=True)[:3]
        issues_str = ", ".join([f"{issue}({count})" for issue, count in top_issues])
        
        print(f"{model_display:<15} {stats['total']:<8} {stats['valid']:<8} {stats['invalid']:<8} {valid_pct:<10} {issues_str}")
    
    # Label adherence analysis for well-formatted predictions
    print(f"\n{'='*80}")
    print(f"{'LABEL ADHERENCE (Well-formatted predictions only)'}")
    print(f"{'='*80}")

    print(f"{'MODEL':<15} {'WELL-FORMATTED':<15} {'VALID LABELS ONLY':<18} {'ADHERENCE %':<12} {'INVALID LABELS USED'}")
    print(f"{'-'*15} {'-'*15} {'-'*18} {'-'*12} {'-'*30}")
    
    for model_name, stats in formatting_stats.items():
        model_display = "Base" if model_name == "base" else "Fine-tuned"
        adherence_pct = f"{stats['valid_labels_rate']:.1%}"
        
        # Get top invalid labels used
        top_invalid = sorted(stats['invalid_labels_used'].items(), key=lambda x: x[1], reverse=True)[:3]
        invalid_str = ", ".join([f"{label}({count})" for label, count in top_invalid]) if top_invalid else "None"
        
        print(f"{model_display:<15} {stats['valid_format_predictions']:<15} {stats['valid_labels_only']:<18} {adherence_pct:<12} {invalid_str}")
    
    print(f"\nNote: Only properly formatted predictions will be included in the accuracy analysis below.")
    
    # Filter to only well-formatted predictions, but keep all examples with any valid predictions
    filtered_results = {}
    for idx, example_data in organized_results.items():
        filtered_example = {
            'title': example_data['title'],
            'body': example_data['body'],
            'actual_labels': example_data['actual_labels']
        }
        
        # Only include models with valid formatting
        for model in ['base', 'fine_tuned']:
            if model in example_data:
                prediction_str = example_data[model]['predicted_labels']
                is_valid, _ = check_prediction_format(prediction_str)
                if is_valid:
                    filtered_example[model] = example_data[model]
        
        # Include examples where at least one model has valid predictions
        if 'base' in filtered_example or 'fine_tuned' in filtered_example:
            filtered_results[idx] = filtered_example
    
    base_valid_count = len([ex for ex in filtered_results.values() if 'base' in ex])
    ft_valid_count = len([ex for ex in filtered_results.values() if 'fine_tuned' in ex])
    print(f"Valid examples: Base model {base_valid_count}, Fine-tuned model {ft_valid_count}")
    organized_results = filtered_results
    
    # Calculate scores for each example
    comparison_results = []
    base_total_correct = 0
    base_total_possible = 0
    ft_total_correct = 0
    ft_total_possible = 0
    
    # For precision tracking
    base_precision_correct = 0
    base_precision_total = 0
    ft_precision_correct = 0
    ft_precision_total = 0
    
    # For recall tracking
    base_recall_correct = 0
    base_recall_total = 0
    ft_recall_correct = 0
    ft_recall_total = 0
    
    for idx, example_data in organized_results.items():
        actual_labels = example_data['actual_labels']
        
        # Initialize result entry
        result_entry = {
            'index': idx,
            'title': example_data['title'],
            'actual': ', '.join(actual_labels)
        }
        
        # Process base model if available
        if 'base' in example_data:
            base_intersection, base_union, base_correct, base_total_pred, base_total_actual, base_jaccard = calculate_score(
                actual_labels, example_data['base']['predicted_labels']
            )
            
            # Jaccard tracking for base
            base_total_correct += base_intersection
            base_total_possible += base_union
            
            # Precision tracking for base
            base_precision_correct += base_correct
            base_precision_total += base_total_pred
            
            # Recall tracking for base
            base_recall_correct += base_correct
            base_recall_total += base_total_actual
            
            result_entry['base'] = {
                'predicted': example_data['base']['predicted_labels'],
                'intersection': base_intersection,
                'union': base_union,
                'jaccard_score': base_jaccard,
                'precision_score': base_correct / base_total_pred if base_total_pred > 0 else 0.0,
                'recall_score': base_correct / base_total_actual if base_total_actual > 0 else 0.0
            }
        
        # Process fine-tuned model if available
        if 'fine_tuned' in example_data:
            ft_intersection, ft_union, ft_correct, ft_total_pred, ft_total_actual, ft_jaccard = calculate_score(
                actual_labels, example_data['fine_tuned']['predicted_labels']
            )
            
            # Jaccard tracking for fine-tuned
            ft_total_correct += ft_intersection
            ft_total_possible += ft_union
            
            # Precision tracking for fine-tuned
            ft_precision_correct += ft_correct
            ft_precision_total += ft_total_pred
            
            # Recall tracking for fine-tuned
            ft_recall_correct += ft_correct
            ft_recall_total += ft_total_actual
            
            result_entry['ft'] = {
                'predicted': example_data['fine_tuned']['predicted_labels'],
                'intersection': ft_intersection,
                'union': ft_union,
                'jaccard_score': ft_jaccard,
                'precision_score': ft_correct / ft_total_pred if ft_total_pred > 0 else 0.0,
                'recall_score': ft_correct / ft_total_actual if ft_total_actual > 0 else 0.0
            }
        
        comparison_results.append(result_entry)
    
    # # Summary comparison
    # print(f"\n{'='*80}")
    # print("COMPARISON SUMMARY")
    # print(f"{'='*80}")
    
    # print(f"{'Example':<8} {'Actual':<20} {'Base Score':<12} {'FT Score':<12} {'Improvement':<12}")
    # print(f"{'-'*8} {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    # We can't calculate direct improvements since examples might not have both models
    # Just skip this section and calculate averages directly later
        
    #     print(f"{result['index']:<8} {actual_short:<20} {base_score:<12} {ft_score:<12} {improvement_str:<12}")
    
    # Overall statistics
    base_jaccard = base_total_correct / base_total_possible if base_total_possible > 0 else 0.0
    ft_jaccard = ft_total_correct / ft_total_possible if ft_total_possible > 0 else 0.0
    jaccard_improvement = ft_jaccard - base_jaccard
    
    base_precision = base_precision_correct / base_precision_total if base_precision_total > 0 else 0.0
    ft_precision = ft_precision_correct / ft_precision_total if ft_precision_total > 0 else 0.0
    precision_improvement = ft_precision - base_precision
    
    base_recall = base_recall_correct / base_recall_total if base_recall_total > 0 else 0.0
    ft_recall = ft_recall_correct / ft_recall_total if ft_recall_total > 0 else 0.0
    recall_improvement = ft_recall - base_recall
    
    # Calculate micro-average F1
    base_micro_f1 = 2 * base_precision * base_recall / (base_precision + base_recall) if (base_precision + base_recall) > 0 else 0.0
    ft_micro_f1 = 2 * ft_precision * ft_recall / (ft_precision + ft_recall) if (ft_precision + ft_recall) > 0 else 0.0
    micro_f1_improvement = ft_micro_f1 - base_micro_f1
    
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    
    # Calculate perfect matches with proper denominators
    base_results = [r for r in comparison_results if 'base' in r]
    ft_results = [r for r in comparison_results if 'ft' in r]
    
    base_perfect_jaccard = len([r for r in base_results if r['base']['jaccard_score'] == 1.0])
    base_perfect_precision = len([r for r in base_results if r['base']['precision_score'] == 1.0])
    base_perfect_recall = len([r for r in base_results if r['base']['recall_score'] == 1.0])
    ft_perfect_jaccard = len([r for r in ft_results if r['ft']['jaccard_score'] == 1.0])
    ft_perfect_precision = len([r for r in ft_results if r['ft']['precision_score'] == 1.0])
    ft_perfect_recall = len([r for r in ft_results if r['ft']['recall_score'] == 1.0])
    
    print(f"\n{'METRIC':<30} {'BASE MODEL':<25} {'FINE-TUNED':<25} {'IMPROVEMENT':<15}")
    print(f"{'-'*30} {'-'*25} {'-'*25} {'-'*15}")
    print(f"{'Jaccard Similarity':<30} {f'{base_jaccard:.2%} ({base_total_correct}/{base_total_possible})':<25} {f'{ft_jaccard:.2%} ({ft_total_correct}/{ft_total_possible})':<25} {jaccard_improvement:+.2%}")
    print(f"{'Micro-Avg Precision':<30} {f'{base_precision:.2%} ({base_precision_correct}/{base_precision_total})':<25} {f'{ft_precision:.2%} ({ft_precision_correct}/{ft_precision_total})':<25} {precision_improvement:+.2%}")
    print(f"{'Micro-Avg Recall':<30} {f'{base_recall:.2%} ({base_recall_correct}/{base_recall_total})':<25} {f'{ft_recall:.2%} ({ft_recall_correct}/{ft_recall_total})':<25} {recall_improvement:+.2%}")
    print(f"{'Micro-Avg F1':<30} {f'{base_micro_f1:.2%}':<25} {f'{ft_micro_f1:.2%}':<25} {micro_f1_improvement:+.2%}")
    base_jaccard_pct = (base_perfect_jaccard / len(base_results) * 100) if base_results else 0
    ft_jaccard_pct = (ft_perfect_jaccard / len(ft_results) * 100) if ft_results else 0
    jaccard_pct_improvement = ft_jaccard_pct - base_jaccard_pct
    
    base_precision_pct = (base_perfect_precision / len(base_results) * 100) if base_results else 0
    ft_precision_pct = (ft_perfect_precision / len(ft_results) * 100) if ft_results else 0
    precision_pct_improvement = ft_precision_pct - base_precision_pct
    
    base_recall_pct = (base_perfect_recall / len(base_results) * 100) if base_results else 0
    ft_recall_pct = (ft_perfect_recall / len(ft_results) * 100) if ft_results else 0
    recall_pct_improvement = ft_recall_pct - base_recall_pct
    
    print(f"{'Perfect Jaccard Matches':<30} {f'{base_perfect_jaccard}/{len(base_results)} ({base_jaccard_pct:.1f}%)':<25} {f'{ft_perfect_jaccard}/{len(ft_results)} ({ft_jaccard_pct:.1f}%)':<25} {jaccard_pct_improvement:+.1f}%")
    print(f"{'Perfect Precision Matches':<30} {f'{base_perfect_precision}/{len(base_results)} ({base_precision_pct:.1f}%)':<25} {f'{ft_perfect_precision}/{len(ft_results)} ({ft_precision_pct:.1f}%)':<25} {precision_pct_improvement:+.1f}%")
    print(f"{'Perfect Recall Matches':<30} {f'{base_perfect_recall}/{len(base_results)} ({base_recall_pct:.1f}%)':<25} {f'{ft_perfect_recall}/{len(ft_results)} ({ft_recall_pct:.1f}%)':<25} {recall_pct_improvement:+.1f}%")
    
    print(f"\n{'PER-EXAMPLE AVERAGES':<30} {'BASE MODEL':<25} {'FINE-TUNED':<25} {'COMPARISON':<15}")
    print(f"{'-'*30} {'-'*25} {'-'*25} {'-'*15}")
    
    # Calculate actual averages properly using separate denominators
    avg_jaccard_base = sum([r['base']['jaccard_score'] for r in base_results])/len(base_results) if base_results else 0
    avg_precision_base = sum([r['base']['precision_score'] for r in base_results])/len(base_results) if base_results else 0
    avg_recall_base = sum([r['base']['recall_score'] for r in base_results])/len(base_results) if base_results else 0
    
    avg_jaccard_ft = sum([r['ft']['jaccard_score'] for r in ft_results])/len(ft_results) if ft_results else 0
    avg_precision_ft = sum([r['ft']['precision_score'] for r in ft_results])/len(ft_results) if ft_results else 0
    avg_recall_ft = sum([r['ft']['recall_score'] for r in ft_results])/len(ft_results) if ft_results else 0
    
    jaccard_avg_improvement = (avg_jaccard_ft - avg_jaccard_base) * 100
    precision_avg_improvement = (avg_precision_ft - avg_precision_base) * 100
    recall_avg_improvement = (avg_recall_ft - avg_recall_base) * 100
    
    print(f"{'Avg Jaccard per Example':<30} {f'{avg_jaccard_base:.2%}':<25} {f'{avg_jaccard_ft:.2%}':<25} {jaccard_avg_improvement:+.2f}%")
    print(f"{'Avg Precision per Example':<30} {f'{avg_precision_base:.2%}':<25} {f'{avg_precision_ft:.2%}':<25} {precision_avg_improvement:+.2f}%")
    print(f"{'Avg Recall per Example':<30} {f'{avg_recall_base:.2%}':<25} {f'{avg_recall_ft:.2%}':<25} {recall_avg_improvement:+.2f}%")
    
    # Calculate comprehensive statistics
    base_stats = calculate_model_stats(organized_results, 'base')
    ft_stats = calculate_model_stats(organized_results, 'fine_tuned')
    
    # # Print comprehensive summary statistics
    # print(f"\n{'='*100}")
    # print("COMPREHENSIVE MODEL STATISTICS")
    # print(f"{'='*100}")
    
    # # Score distribution comparison
    # print(f"\n{'METRIC':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'IMPROVEMENT':<15}")
    # print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    # print(f"{'Perfect Matches':<30} {base_stats['perfect_matches']}/{len(comparison_results):<20} {ft_stats['perfect_matches']}/{len(comparison_results):<20} {ft_stats['perfect_matches'] - base_stats['perfect_matches']:+d}")
    # print(f"{'Partial Matches':<30} {base_stats['partial_matches']}/{len(comparison_results):<20} {ft_stats['partial_matches']}/{len(comparison_results):<20} {ft_stats['partial_matches'] - base_stats['partial_matches']:+d}")
    # print(f"{'No Matches':<30} {base_stats['no_matches']}/{len(comparison_results):<20} {ft_stats['no_matches']}/{len(comparison_results):<20} {ft_stats['no_matches'] - base_stats['no_matches']:+d}")
    # base_avg_str = f"{base_stats['avg_score']:.2%}"
    # ft_avg_str = f"{ft_stats['avg_score']:.2%}"
    # print(f"{'Average Score':<30} {base_avg_str:<20} {ft_avg_str:<20} {ft_stats['avg_score'] - base_stats['avg_score']:+.2%}")
    # base_min_str = f"{base_stats['min_score']:.2%}"
    # ft_min_str = f"{ft_stats['min_score']:.2%}"
    # print(f"{'Min Score':<30} {base_min_str:<20} {ft_min_str:<20} {ft_stats['min_score'] - base_stats['min_score']:+.2%}")
    # base_max_str = f"{base_stats['max_score']:.2%}"
    # ft_max_str = f"{ft_stats['max_score']:.2%}"
    # print(f"{'Max Score':<30} {base_max_str:<20} {ft_max_str:<20} {ft_stats['max_score'] - base_stats['max_score']:+.2%}")
    
    # # Prediction behavior comparison
    # print(f"\n{'PREDICTION BEHAVIOR':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'DIFFERENCE':<15}")
    # print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    # print(f"{'Total Labels Predicted':<30} {base_stats['total_predicted']:<20} {ft_stats['total_predicted']:<20} {ft_stats['total_predicted'] - base_stats['total_predicted']:+d}")
    # print(f"{'Unique Labels Used':<30} {base_stats['unique_predicted']:<20} {ft_stats['unique_predicted']:<20} {ft_stats['unique_predicted'] - base_stats['unique_predicted']:+d}")
    # base_avg_labels_str = f"{base_stats['avg_labels_per_example']:.1f}"
    # ft_avg_labels_str = f"{ft_stats['avg_labels_per_example']:.1f}"
    # print(f"{'Avg Labels per Example':<30} {base_avg_labels_str:<20} {ft_avg_labels_str:<20} {ft_stats['avg_labels_per_example'] - base_stats['avg_labels_per_example']:+.1f}")
    
    # # STT-style error analysis
    # print(f"\n{'ERROR ANALYSIS (STT-STYLE)':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20} {'DIFFERENCE':<15}")
    # print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    # print(f"{'Label Insertions (Extra)':<30} {base_stats['total_insertions']:<20} {ft_stats['total_insertions']:<20} {ft_stats['total_insertions'] - base_stats['total_insertions']:+d}")
    # print(f"{'Label Deletions (Missing)':<30} {base_stats['total_deletions']:<20} {ft_stats['total_deletions']:<20} {ft_stats['total_deletions'] - base_stats['total_deletions']:+d}")
    # print(f"{'Examples with Mixed Errors':<30} {base_stats['total_substitutions']:<20} {ft_stats['total_substitutions']:<20} {ft_stats['total_substitutions'] - base_stats['total_substitutions']:+d}")
    # base_ins_str = f"{base_stats['insertion_rate']:.2%}"
    # ft_ins_str = f"{ft_stats['insertion_rate']:.2%}"
    # print(f"{'Insertion Rate':<30} {base_ins_str:<20} {ft_ins_str:<20} {ft_stats['insertion_rate'] - base_stats['insertion_rate']:+.2%}")
    # base_del_str = f"{base_stats['deletion_rate']:.2%}"
    # ft_del_str = f"{ft_stats['deletion_rate']:.2%}"
    # print(f"{'Deletion Rate':<30} {base_del_str:<20} {ft_del_str:<20} {ft_stats['deletion_rate'] - base_stats['deletion_rate']:+.2%}")
    # base_err_str = f"{base_stats['error_rate']:.2%}"
    # ft_err_str = f"{ft_stats['error_rate']:.2%}"
    # print(f"{'Overall Error Rate':<30} {base_err_str:<20} {ft_err_str:<20} {ft_stats['error_rate'] - base_stats['error_rate']:+.2%}")
    
    # Label accuracy breakdown
    print(f"\n{'LABEL ACCURACY BREAKDOWN':<30} {'BASE MODEL':<20} {'FINE-TUNED':<20}")
    print(f"{'-'*30} {'-'*20} {'-'*20}")
    print(f"{'Correct Labels':<30} {base_stats['total_correct_labels']:<20} {ft_stats['total_correct_labels']:<20}")
    print(f"{'Labels in Ground Truth':<30} {base_stats['total_actual_labels']:<20} {ft_stats['total_actual_labels']:<20}")
    print(f"{'Labels Predicted':<30} {base_stats['total_predicted_labels']:<20} {ft_stats['total_predicted_labels']:<20}")
    
    # # Per-label performance comparison
    # print(f"\n{'='*100}")
    # print("PER-LABEL PERFORMANCE COMPARISON")
    # print(f"{'='*100}")
    
    all_labels = set(list(base_stats['label_stats'].keys()) + list(ft_stats['label_stats'].keys()))
    sorted_labels = sorted(all_labels)
    
    # print(f"\n{'LABEL':<20} {'BASE F1':<12} {'FT F1':<12} {'BASE PREC':<12} {'FT PREC':<12} {'BASE REC':<12} {'FT REC':<12}")
    # print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for label in sorted_labels:
        base_f1 = base_stats['label_stats'].get(label, {}).get('f1', 0)
        ft_f1 = ft_stats['label_stats'].get(label, {}).get('f1', 0)
        base_prec = base_stats['label_stats'].get(label, {}).get('precision', 0)
        ft_prec = ft_stats['label_stats'].get(label, {}).get('precision', 0)
        base_rec = base_stats['label_stats'].get(label, {}).get('recall', 0)
        ft_rec = ft_stats['label_stats'].get(label, {}).get('recall', 0)
        
        # print(f"{label:<20} {f'{base_f1:.2f}':<12} {f'{ft_f1:.2f}':<12} {f'{base_prec:.2f}':<12} {f'{ft_prec:.2f}':<12} {f'{base_rec:.2f}':<12} {f'{ft_rec:.2f}':<12}")
    
    # Calculate macro averages
    base_macro_f1 = sum([stats.get('f1', 0) for stats in base_stats['label_stats'].values()]) / len(base_stats['label_stats']) if base_stats['label_stats'] else 0
    ft_macro_f1 = sum([stats.get('f1', 0) for stats in ft_stats['label_stats'].values()]) / len(ft_stats['label_stats']) if ft_stats['label_stats'] else 0
    
    # print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    # print(f"{'MACRO AVERAGE':<20} {f'{base_macro_f1:.2f}':<12} {f'{ft_macro_f1:.2f}':<12} {'':12} {'':12} {'':12} {'':12}")
    
    # print(f"\n{'='*100}")
    # print("SUMMARY")
    # print(f"{'='*100}")
    # print(f"Fine-tuning Impact:")
    # print(f"  • Macro F1 Score: {base_macro_f1:.3f} → {ft_macro_f1:.3f} ({ft_macro_f1 - base_macro_f1:+.3f})")
    # print(f"  • Perfect Matches: {base_stats['perfect_matches']}/{len(comparison_results)} → {ft_stats['perfect_matches']}/{len(comparison_results)} ({ft_stats['perfect_matches'] - base_stats['perfect_matches']:+d})")
    # print(f"  • Prediction Efficiency: {base_stats['avg_labels_per_example']:.1f} → {ft_stats['avg_labels_per_example']:.1f} labels/example ({base_stats['avg_labels_per_example'] - ft_stats['avg_labels_per_example']:+.1f})")
    # print(f"  • Label Insertions (Extra): {base_stats['total_insertions']} → {ft_stats['total_insertions']} ({ft_stats['total_insertions'] - base_stats['total_insertions']:+d})")
    # print(f"  • Label Deletions (Missing): {base_stats['total_deletions']} → {ft_stats['total_deletions']} ({ft_stats['total_deletions'] - base_stats['total_deletions']:+d})")
    # print(f"  • Overall Error Rate: {base_stats['error_rate']:.1%} → {ft_stats['error_rate']:.1%} ({ft_stats['error_rate'] - base_stats['error_rate']:+.1%})")
    
    # Analysis of error improvements
    insertion_improvement = base_stats['total_insertions'] - ft_stats['total_insertions']
    deletion_change = ft_stats['total_deletions'] - base_stats['total_deletions']
    error_improvement = base_stats['error_rate'] - ft_stats['error_rate']
    
    # print(f"\nError Analysis:")
    # if insertion_improvement > 0:
    #     print(f"  ✅ Reduced over-predictions by {insertion_improvement} labels ({insertion_improvement/base_stats['total_insertions']:.1%})")
    # elif insertion_improvement < 0:
    #     print(f"  ⚠️  Increased over-predictions by {abs(insertion_improvement)} labels")
    
    # if deletion_change < 0:
    #     print(f"  ✅ Reduced under-predictions by {abs(deletion_change)} labels")
    # elif deletion_change > 0:
    #     print(f"  ⚠️  Increased under-predictions by {deletion_change} labels")
    
    # if error_improvement > 0:
    #     print(f"  ✅ Overall error rate improved by {error_improvement:.1%}")
    # elif error_improvement < 0:
    #     print(f"  ⚠️  Overall error rate worsened by {abs(error_improvement):.1%}")
    
    # if ft_macro_f1 > base_macro_f1:
    #     print(f"  🎯 Fine-tuning IMPROVED overall performance")
    # elif ft_macro_f1 < base_macro_f1:
    #     print(f"  ⚠️  Fine-tuning DECREASED overall performance")
    # else:
    #     print(f"  ➡️  Fine-tuning had NEUTRAL impact on overall performance")

if __name__ == "__main__":
    main()