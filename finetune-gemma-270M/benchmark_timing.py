#!/usr/bin/env python3
"""
Benchmark script to measure model loading and inference timing on CPU vs GPU.
Tests both base and fine-tuned models across 10 examples.
"""

import time
import torch
import statistics
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

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

def load_model_timed(model_path, device="cuda", max_seq_length=2048):
    """Load model and tokenizer with timing."""
    print(f"Loading model from {model_path} on {device}...")
    start_time = time.time()
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    
    # Move to device if specified
    if device == "cpu":
        model = model.cpu()
        # Convert to float32 for CPU compatibility
        model = model.float()
    else:
        model = model.cuda()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return model, tokenizer, load_time

def run_inference_timed(model, tokenizer, example, device="cuda"):
    """Run inference on a single example with timing."""
    # Prepare messages
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
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to("cuda")
    
    # Time the inference
    start_time = time.time()
    
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
    
    inference_time = time.time() - start_time
    
    # Decode result
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract prediction - debug the full output first
    print(f"    Generated text: {repr(generated_text[-200:])}")  # Show last 200 chars for debugging
    
    # Extract just the model's response after the last turn
    if "<start_of_turn>model\n" in generated_text:
        prediction = generated_text.split("<start_of_turn>model\n")[-1].strip()
    elif "model\n" in generated_text:
        prediction = generated_text.split("model\n")[-1].strip()
    else:
        prediction = "No model response found"
    
    # Clean up the prediction
    if prediction.endswith("<end_of_turn>"):
        prediction = prediction.replace("<end_of_turn>", "").strip()
    
    return prediction, inference_time

def benchmark_model(model_path, model_name, device="cuda"):
    """Benchmark a model on given device."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING {model_name.upper()} ON {device.upper()}")
    print(f"{'='*60}")
    
    # Load model and time it
    model, tokenizer, load_time = load_model_timed(model_path, device)
    
    # Load test dataset
    dataset = load_dataset("AdamDLewis/nebari-issue-label-dataset", split="test[:10]")
    
    # Run inference timing
    inference_times = []
    predictions = []
    
    print(f"\nRunning inference on {len(dataset)} examples...")
    
    for i, example in enumerate(dataset):
        print(f"Example {i+1}/10: {example['title'][:50]}...")
        prediction, inf_time = run_inference_timed(model, tokenizer, example, device)
        inference_times.append(inf_time)
        predictions.append(prediction)
        print(f"  Time: {inf_time:.3f}s | Prediction: {prediction[:60]}...")
    
    # Calculate statistics
    avg_inference_time = statistics.mean(inference_times)
    median_inference_time = statistics.median(inference_times)
    min_inference_time = min(inference_times)
    max_inference_time = max(inference_times)
    std_inference_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
    
    # Clean up memory
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        'model_name': model_name,
        'device': device,
        'load_time': load_time,
        'inference_times': inference_times,
        'avg_inference_time': avg_inference_time,
        'median_inference_time': median_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'std_inference_time': std_inference_time,
        'total_inference_time': sum(inference_times),
        'predictions': predictions
    }

def main():
    print("MODEL TIMING BENCHMARK")
    print("=" * 80)
    
    # Check device availability
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Models to test - just fine-tuned model
    models = [
        ("lora_model", "Fine-tuned Model")
    ]
    
    # Devices to test
    devices = ["cuda"] if has_cuda else ["cpu"]
    if has_cuda:
        devices.append("cpu")  # Test both if CUDA is available
    
    results = []
    
    # Run benchmarks
    for model_path, model_name in models:
        for device in devices:
            try:
                result = benchmark_model(model_path, model_name, device)
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {model_name} on {device}: {e}")
                continue
    
    # Print comparison table
    print(f"\n{'='*100}")
    print("TIMING COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    print(f"{'MODEL':<20} {'DEVICE':<8} {'LOAD TIME':<12} {'AVG INF':<12} {'MIN INF':<12} {'MAX INF':<12} {'TOTAL':<12}")
    print(f"{'-'*20} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for result in results:
        print(f"{result['model_name']:<20} "
              f"{result['device']:<8} "
              f"{result['load_time']:.2f}s{'':<6} "
              f"{result['avg_inference_time']:.3f}s{'':<5} "
              f"{result['min_inference_time']:.3f}s{'':<5} "
              f"{result['max_inference_time']:.3f}s{'':<5} "
              f"{result['total_inference_time']:.2f}s{'':<6}")
    
    # Performance comparisons
    if len(results) >= 2:
        print(f"\n{'='*100}")
        print("PERFORMANCE COMPARISONS")
        print(f"{'='*100}")
        
        # Find matching device comparisons
        for device in devices:
            device_results = [r for r in results if r['device'] == device]
            if len(device_results) == 2:
                base = next((r for r in device_results if "Base" in r['model_name']), None)
                ft = next((r for r in device_results if "Fine-tuned" in r['model_name']), None)
                
                if base and ft:
                    load_speedup = base['load_time'] / ft['load_time']
                    inf_speedup = base['avg_inference_time'] / ft['avg_inference_time']
                    
                    print(f"\n{device.upper()} Performance:")
                    print(f"  Load time - Base: {base['load_time']:.2f}s, Fine-tuned: {ft['load_time']:.2f}s")
                    print(f"  Load speedup: {load_speedup:.2f}x ({'Fine-tuned faster' if load_speedup > 1 else 'Base faster'})")
                    print(f"  Avg inference - Base: {base['avg_inference_time']:.3f}s, Fine-tuned: {ft['avg_inference_time']:.3f}s")
                    print(f"  Inference speedup: {inf_speedup:.2f}x ({'Fine-tuned faster' if inf_speedup > 1 else 'Base faster'})")
        
        # GPU vs CPU comparison for same model
        for model_path, model_name in models:
            model_results = [r for r in results if r['model_name'] == model_name]
            gpu_result = next((r for r in model_results if r['device'] == 'cuda'), None)
            cpu_result = next((r for r in model_results if r['device'] == 'cpu'), None)
            
            if gpu_result and cpu_result:
                load_speedup = cpu_result['load_time'] / gpu_result['load_time']
                inf_speedup = cpu_result['avg_inference_time'] / gpu_result['avg_inference_time']
                
                print(f"\n{model_name} - GPU vs CPU:")
                print(f"  Load time - GPU: {gpu_result['load_time']:.2f}s, CPU: {cpu_result['load_time']:.2f}s")
                print(f"  GPU load speedup: {load_speedup:.2f}x")
                print(f"  Avg inference - GPU: {gpu_result['avg_inference_time']:.3f}s, CPU: {cpu_result['avg_inference_time']:.3f}s")
                print(f"  GPU inference speedup: {inf_speedup:.2f}x")
    
    # Detailed timing breakdown
    print(f"\n{'='*100}")
    print("DETAILED TIMING STATISTICS")
    print(f"{'='*100}")
    
    for result in results:
        print(f"\n{result['model_name']} on {result['device'].upper()}:")
        print(f"  Load time: {result['load_time']:.2f}s")
        print(f"  Inference statistics:")
        print(f"    Mean: {result['avg_inference_time']:.3f}s")
        print(f"    Median: {result['median_inference_time']:.3f}s")
        print(f"    Min: {result['min_inference_time']:.3f}s")
        print(f"    Max: {result['max_inference_time']:.3f}s")
        print(f"    Std Dev: {result['std_inference_time']:.3f}s")
        print(f"    Total: {result['total_inference_time']:.2f}s")
    
    print(f"\n{'='*100}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()