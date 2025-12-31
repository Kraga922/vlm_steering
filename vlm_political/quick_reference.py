#!/usr/bin/env python3
"""
Quick Reference: VLM Political Steering

This file provides quick examples for common steering tasks.
Copy and adapt as needed for your experiments.
"""

import pickle
import torch
from pathlib import Path

# ==============================================================================
# 1. LOAD A SAVED STEERING DIRECTION
# ==============================================================================

def load_steering_direction(direction_file):
    """Load a previously computed steering direction."""
    with open(direction_file, 'rb') as f:
        data = pickle.load(f)
    direction = torch.tensor(data['direction'], dtype=torch.float32)
    return direction

# Example:
# direction = load_steering_direction('vlm_political/directions/conservative_liberal_direction_20240115_143022.pkl')


# ==============================================================================
# 2. APPLY STEERING TO GENERATE TEXT
# ==============================================================================

def generate_with_steering(model, processor, device, prompt, direction, 
                          coefficient=1.0, max_tokens=100):
    """Generate text with steering applied."""
    from your_notebook import apply_steering_to_model  # Import from notebook
    
    # Register hooks
    handles, _ = apply_steering_to_model(
        model,
        direction,
        coefficient=coefficient,
        target_type="decoder"
    )
    
    # Generate
    inputs = processor(text=prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    
    text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Cleanup
    for h in handles:
        h.remove()
    
    return text

# Example:
# baseline = generate_with_steering(model, processor, device, "What is your view on climate change?", direction, coefficient=0.0)
# liberal = generate_with_steering(model, processor, device, "What is your view on climate change?", direction, coefficient=1.0)
# conservative = generate_with_steering(model, processor, device, "What is your view on climate change?", direction, coefficient=-1.0)


# ==============================================================================
# 3. COMPARE BASELINE VS STEERED OUTPUTS
# ==============================================================================

def compare_steering(model, processor, device, prompt, direction, 
                     coefficients=[0.0, 1.0, -1.0], max_tokens=100):
    """Generate and compare outputs at different steering strengths."""
    results = {}
    
    for coef in coefficients:
        text = generate_with_steering(
            model, processor, device, prompt, direction,
            coefficient=coef, max_tokens=max_tokens
        )
        label = {0.0: "Baseline", 1.0: "Liberal", -1.0: "Conservative"}.get(coef, f"Coef {coef}")
        results[label] = text
    
    return results

# Example:
# results = compare_steering(model, processor, device, "What is your view on healthcare?", direction)
# for label, text in results.items():
#     print(f"\n{label}:\n{text}")


# ==============================================================================
# 4. BATCH EVALUATION
# ==============================================================================

def evaluate_multiple_topics(model, processor, device, direction, topics,
                            coefficients=[0.0, 1.0, -1.0], max_tokens=80):
    """Evaluate steering across multiple topics."""
    results = {}
    
    for topic in topics:
        prompt = f"What is your view on {topic}? Provide a balanced perspective."
        results[topic] = compare_steering(
            model, processor, device, prompt, direction,
            coefficients=coefficients, max_tokens=max_tokens
        )
    
    return results

# Example:
# topics = ["abortion rights", "climate change", "healthcare", "immigration"]
# results = evaluate_multiple_topics(model, processor, device, direction, topics)
# for topic, outputs in results.items():
#     print(f"\n{'='*50}")
#     print(f"TOPIC: {topic}")
#     print('='*50)
#     for label, text in outputs.items():
#         print(f"\n[{label}]")
#         print(text[:200] + "..." if len(text) > 200 else text)


# ==============================================================================
# 5. EXTRACT EMBEDDINGS FROM NEW IMAGES
# ==============================================================================

def get_image_embedding(model, processor, device, image_path, prompt=""):
    """Extract embedding for a single image."""
    from PIL import Image
    
    img = Image.open(image_path).convert("RGB")
    
    if not prompt:
        prompt = "This image shows a viewpoint. Describe the content."
    
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Average over sequence
    
    return embedding

# Example:
# emb = get_image_embedding(model, processor, device, "image.jpg", 
#                           prompt="This image shows a conservative viewpoint on immigration.")


# ==============================================================================
# 6. SWEEP COEFFICIENT VALUES
# ==============================================================================

def coefficient_sweep(model, processor, device, prompt, direction, 
                      min_coef=-2.0, max_coef=2.0, step=0.5, max_tokens=100):
    """Test a range of coefficient values."""
    coefficients = [c for c in [i*step for i in range(int(min_coef/step), int(max_coef/step)+1)]]
    
    results = {}
    for coef in coefficients:
        text = generate_with_steering(
            model, processor, device, prompt, direction,
            coefficient=coef, max_tokens=max_tokens
        )
        results[f"Coef={coef:.1f}"] = text
    
    return results

# Example:
# sweep = coefficient_sweep(model, processor, device, "What is your view on climate change?", direction)


# ==============================================================================
# 7. SAVE EXPERIMENT RESULTS
# ==============================================================================

def save_results(results, output_file="steering_results.pkl"):
    """Save experiment results to disk."""
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_file}")

def load_results(input_file="steering_results.pkl"):
    """Load previously saved results."""
    with open(input_file, 'rb') as f:
        results = pickle.load(f)
    return results


# ==============================================================================
# 8. ANALYZE DIRECTION PROPERTIES
# ==============================================================================

def analyze_direction(direction):
    """Print statistics about a steering direction."""
    print(f"Direction shape: {direction.shape}")
    print(f"Norm: {torch.norm(direction).item():.4f}")
    print(f"Min value: {direction.min().item():.4f}")
    print(f"Max value: {direction.max().item():.4f}")
    print(f"Mean value: {direction.mean().item():.4f}")
    print(f"Std value: {direction.std().item():.4f}")
    
    # Top 5 positive and negative dimensions
    top_pos_idx = torch.argsort(direction[0])[-5:]
    top_neg_idx = torch.argsort(direction[0])[:5]
    
    print(f"\nTop 5 liberal-leaning dims: {top_pos_idx.tolist()}")
    print(f"  Values: {direction[0, top_pos_idx].tolist()}")
    
    print(f"\nTop 5 conservative-leaning dims: {top_neg_idx.tolist()}")
    print(f"  Values: {direction[0, top_neg_idx].tolist()}")


# ==============================================================================
# 9. COMBINE MULTIPLE STEERING DIRECTIONS
# ==============================================================================

def blend_directions(direction1, direction2, weight=0.5):
    """Blend two steering directions together."""
    blended = (1 - weight) * direction1 + weight * direction2
    return blended / (torch.norm(blended) + 1e-8)  # Normalize

# Example:
# dir1 = load_steering_direction('direction_abortion.pkl')
# dir2 = load_steering_direction('direction_climate.pkl')
# blended = blend_directions(dir1, dir2, weight=0.5)


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    print("""
    VLM Political Steering - Quick Reference
    
    Import functions from this file and use them in your experiments:
    
        from quick_reference import *
        
        # Load a direction
        direction = load_steering_direction('path/to/direction.pkl')
        
        # Generate with steering
        text = generate_with_steering(model, processor, device, prompt, direction, coefficient=1.0)
        
        # Compare baseline vs steered
        results = compare_steering(model, processor, device, prompt, direction)
        for label, text in results.items():
            print(f"{label}: {text}")
        
        # Evaluate multiple topics
        topics = ["abortion", "climate change", "healthcare"]
        results = evaluate_multiple_topics(model, processor, device, direction, topics)
        
        # Analyze direction properties
        analyze_direction(direction)
    
    For more details, see USAGE.md and IMPLEMENTATION.md
    """)
