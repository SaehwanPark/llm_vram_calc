# LLM VRAM Calculator

A streamlined tool to calculate GPU memory requirements for working with Large Language Models (LLMs), covering training, fine-tuning, and inference scenarios.

## Overview

The LLM VRAM Calculator helps AI researchers and engineers estimate the GPU memory needed for various LLM operations. This information is crucial for planning hardware requirements and optimizing resource allocation for LLM projects.

## Installation

```bash
# Clone the repository
git clone https://github.com/SaehwanPark/llm_vram_calc.git
cd llm_vram_calc

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run llm_vram_calc.py
```

## Features

- **Training Calculator**: Estimate VRAM for training LLMs from scratch
- **Fine-tuning Calculator**: Calculate memory requirements for different fine-tuning methods
- **Inference Calculator**: Determine memory needs for model deployment and serving
- **GPU Compatibility**: View maximum batch sizes for common GPUs
- **Interactive UI**: Adjust parameters and see results in real-time

## Terminology

### General Terms

- **LLM**: Large Language Model - AI models with billions of parameters trained on text data
- **VRAM**: Video RAM - Memory available on GPUs for model storage and computations
- **Parameter**: A trainable weight in the neural network model
- **Batch Size**: Number of examples processed simultaneously
- **Sequence Length**: Number of tokens in each training or inference example

### Training Terms

- **Optimizer State**: Memory used to store optimizer-specific values for each parameter
- **Gradient Accumulation**: Technique to simulate larger batch sizes by accumulating gradients
- **Activation Checkpointing**: Memory optimization that trades computation for memory by recomputing activations

### Fine-tuning Terms

- **LoRA**: Low-Rank Adaptation - Parameter-efficient fine-tuning method
- **QLoRA**: Quantized LoRA - Combines quantization with LoRA for memory efficiency
- **Parameter-Efficient Fine-tuning**: Methods that update only a small subset of model parameters

### Inference Terms

- **KV Cache**: Key-Value cache storing intermediate attention results for efficient autoregressive generation
- **Quantization**: Technique to reduce model precision (e.g., FP16, INT8, INT4)
- **MHA**: Multi-head Attention - Standard attention mechanism with separate key-value pairs per head
- **MQA**: Multi-query Attention - Uses one key-value pair for all query heads
- **GQA**: Grouped-query Attention - Uses shared key-value pairs within defined groups

## License

Apache License 2.0

## Citation

If you use this tool in your research, please cite:

```
@software{llm_vram_calc,
  author = {Sae-HWan Park},
  title = {LLM VRAM Calculator},
  year = {2025},
  url = {https://github.com/SaehwanPark/llm_vram_calc}
}
```