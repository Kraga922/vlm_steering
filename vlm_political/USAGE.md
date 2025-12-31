# VLM Political Steering Notebook

This notebook demonstrates how to steer Qwen3-VL-32B responses along a political axis using image-based steering directions.

## Setup

### Dataset Organization
Place political images in the following structure:
```
vlm_political/
├── imgs/
│   ├── conservative/
│   │   ├── abortion/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   ├── climate_change/
│   │   │   └── ...
│   │   └── ...
│   └── liberal/
│       ├── abortion/
│       │   ├── image1.jpg
│       │   └── image2.jpg
│       └── ...
├── notebooks/
│   └── steer_qwen_vl_political.ipynb
└── directions/
    ├── conservative_liberal_direction_[timestamp].pkl
    └── metadata_[timestamp].json
```

### Running the Notebook

1. **Install Dependencies**:
   ```bash
   pip install torch transformers PIL numpy scikit-learn tqdm
   ```

2. **Run the Notebook**:
   ```bash
   jupyter notebook vlm_political/notebooks/steer_qwen_vl_political.ipynb
   ```

3. **Execute Cells in Order**:
   - Cell 1: Environment setup & imports
   - Cell 2: Load Qwen3-VL-32B model
   - Cell 3: Discover dataset structure
   - Cell 4: Extract embeddings from images
   - Cell 5: Compute steering directions (logistic regression)
   - Cell 6: Define steering hook classes
   - Cell 7: Generate examples (baseline vs. steered)
   - Cell 8: Save directions to disk

## How It Works

### 1. Image Embedding Extraction
For each image, the notebook:
- Loads image from disk
- Constructs prompt: `"This image shows a [conservative|liberal] viewpoint on [topic]"`
- Passes image + prompt through Qwen3-VL-32B's vision encoder
- Extracts hidden state as embedding

### 2. Direction Computation
- Collects all embeddings for conservative images
- Collects all embeddings for liberal images
- Trains logistic regression classifier: `y = [0]*n_cons + [1]*n_lib`
- Uses classifier weight vector as steering direction

### 3. Steering Application
Registers forward hooks on decoder layers:
```python
handles, hook = apply_steering_to_model(
    model,
    steering_direction,
    coefficient=1.0,  # Shift toward liberal
    target_type="decoder"
)
# Generate text with hooks active
# ... model.generate() ...
# Remove hooks
for h in handles: h.remove()
```

### 4. Coefficient Control
- **`coefficient > 0`**: Shift toward liberal viewpoint
- **`coefficient < 0`**: Shift toward conservative viewpoint
- **`|coefficient|` controls strength**: Start with ±1.0

## Output Files

### Saved Directions
- **File**: `vlm_political/directions/conservative_liberal_direction_[timestamp].pkl`
- **Format**: Python pickle dictionary
- **Contents**:
  ```python
  {
      'direction': numpy array (1, hidden_dim),
      'direction_shape': tuple,
      'computation_method': 'logistic_regression',
      'num_conservative_images': int,
      'num_liberal_images': int,
      'timestamp': str
  }
  ```

### Loading Saved Directions
```python
import pickle
data = pickle.load(open('vlm_political/directions/conservative_liberal_direction_[timestamp].pkl', 'rb'))
steering_direction = torch.tensor(data['direction'], dtype=torch.float32)
```

## Example Prompts

The notebook tests steering with prompts like:
- "What is your view on abortion rights?"
- "What is your view on climate change?"
- "What is your view on healthcare?"
- "What is your view on immigration?"

For each prompt, it generates three versions:
1. **Baseline** (no steering)
2. **Liberal-steered** (coefficient=1.0)
3. **Conservative-steered** (coefficient=-1.0)

## Evaluation Ideas

1. **Classifier-based**: Train/use political sentiment classifier to measure shift
2. **Human evaluation**: Show outputs to human raters (blinded to condition)
3. **Keyword frequency**: Analyze changes in political terminology
4. **Stance detection**: Use existing stance detection models

## Limitations & Considerations

- **Image-dependent**: Direction quality depends on image dataset diversity and relevance
- **Topic generalization**: Direction trained on one topic may not transfer perfectly to others
- **Model-specific**: Direction is specific to Qwen3-VL-32B architecture
- **Prompt sensitivity**: Steering effectiveness may vary based on input prompt
- **Ethics**: Steering can amplify biases or create misleading outputs; use responsibly

## Advanced Usage

### Multiple Steering Vectors
Compute separate directions for different political topics:
```python
directions = {
    'abortion': compute_steering_directions(abort_cons, abort_lib),
    'climate': compute_steering_directions(climate_cons, climate_lib),
    'immigration': compute_steering_directions(immig_cons, immig_lib),
}
```

### Vision-Layer Steering
Also apply hooks to vision encoder:
```python
vision_handles, _ = apply_steering_to_model(
    model, steering_direction, coefficient=0.5, target_type="vision"
)
decoder_handles, _ = apply_steering_to_model(
    model, steering_direction, coefficient=1.0, target_type="decoder"
)
# Generate with both active
for h in vision_handles + decoder_handles: h.remove()
```

### Coefficient Sweep
Test different steering strengths:
```python
for coef in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    handles, _ = apply_steering_to_model(model, direction, coefficient=coef)
    # Generate and evaluate
    for h in handles: h.remove()
```

## References

- **Paper**: Inspired by RepE framework for representation engineering
- **Model**: [Qwen3-VL-32B on Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-32B)
- **Steering Hooks**: PyTorch `register_forward_hook()` API

## Troubleshooting

**Q: CUDA out of memory?**
A: Reduce batch size in `extract_image_embeddings()` or use smaller model variant

**Q: Embeddings look similar (low variance)?**
A: Check image dataset quality; ensure images are distinct between conservative/liberal

**Q: Steering has no effect?**
A: Try higher coefficient values (±2.0, ±5.0), different layers, or more training images

**Q: Hooks not removing?**
A: Save hook handles and explicitly call `.remove()` on each before next generation
