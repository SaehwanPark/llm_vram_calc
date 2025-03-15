import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="LLM VRAM Calculator", layout="wide")

st.title("LLM VRAM Calculator")
st.markdown("""
This tool helps estimate VRAM requirements for working with Large Language Models (LLMs).
Choose a scenario below and input your parameters to get an estimate.
""")

# Helper functions
def bytes_to_human_readable(bytes_value):
    """Convert bytes to human-readable form with appropriate units."""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.2f} KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/(1024**2):.2f} MB"
    elif bytes_value < 1024**4:
        return f"{bytes_value/(1024**3):.2f} GB"
    else:
        return f"{bytes_value/(1024**4):.2f} TB"
    
def human_readable_to_bytes(size_str):
    """Convert human-readable size to bytes."""
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
    size = float(size_str.split()[0])
    unit = size_str.split()[1]
    return size * units[unit]

# Create tabs for different scenarios
tab1, tab2, tab3 = st.tabs(["Training", "Fine-tuning", "Inference"])

# Training tab
with tab1:
    st.header("Training LLMs from Scratch")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        
        model_size = st.number_input(
            "Number of Parameters (billions)",
            min_value=0.1,
            max_value=1000.0,
            value=7.0,
            step=0.1,
            help="Total number of parameters in billions (e.g., 7B, 13B, 70B)"
        )
        
        precision = st.selectbox(
            "Training Precision",
            options=["FP32 (4 bytes)", "FP16 (2 bytes)", "BF16 (2 bytes)"],
            index=1,
            help="Numerical precision used for model parameters"
        )
        
        optimizer = st.selectbox(
            "Optimizer",
            options=["Adam/AdamW (8-12 bytes per parameter)", "SGD with momentum (4-8 bytes)"],
            index=0,
            help="Optimization algorithm used for training"
        )
        
        activation_checkpointing = st.checkbox(
            "Use Activation Checkpointing",
            value=True,
            help="Reduces memory by recomputing activations during backpropagation"
        )
        
    with col2:
        st.subheader("Training Configuration")
        
        batch_size = st.number_input(
            "Batch Size (sequences)",
            min_value=1,
            max_value=1024,
            value=32,
            step=1,
            help="Number of sequences processed simultaneously"
        )
        
        sequence_length = st.number_input(
            "Sequence Length (tokens)",
            min_value=128,
            max_value=32768,
            value=2048,
            step=128,
            help="Maximum number of tokens per sequence"
        )
        
        gradient_accumulation = st.number_input(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=64,
            value=1,
            step=1,
            help="Number of steps to accumulate gradients before updating weights"
        )
        
        hidden_size = st.number_input(
            "Hidden Size",
            min_value=768,
            max_value=12288,
            value=4096,
            step=128,
            help="Dimension of the model's hidden representations"
        )
        
        num_layers = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=128,
            value=32,
            step=1,
            help="Number of transformer layers in the model"
        )
        
    # Calculate memory requirements
    # Parse precision value
    if "FP32" in precision:
        param_bytes = 4
    else:  # FP16 or BF16
        param_bytes = 2
        
    # Parse optimizer overhead
    if "Adam" in optimizer:
        optimizer_bytes = 10  # Average between 8-12
    else:  # SGD
        optimizer_bytes = 6   # Average between 4-8
    
    # Calculate memory components
    model_params_memory = model_size * 1e9 * param_bytes
    optimizer_memory = model_size * 1e9 * optimizer_bytes
    
    # Gradient memory (same size as model parameters)
    gradient_memory = model_size * 1e9 * param_bytes / gradient_accumulation
    
    # Activation memory
    if activation_checkpointing:
        activation_factor = 1.5  # Reduced by checkpointing
    else:
        activation_factor = 6  # Higher without checkpointing
    
    activation_memory = batch_size * sequence_length * num_layers * hidden_size * param_bytes * activation_factor / gradient_accumulation
    
    # Memory fragmentation and overhead (typically 10-20%)
    overhead_factor = 1.15
    
    # Total memory
    total_memory = (model_params_memory + optimizer_memory + gradient_memory + activation_memory) * overhead_factor
    
    # Display results
    st.subheader("Estimated VRAM Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Parameters", bytes_to_human_readable(model_params_memory))
        st.metric("Optimizer State", bytes_to_human_readable(optimizer_memory))
        st.metric("Gradients", bytes_to_human_readable(gradient_memory))
    
    with col2:
        st.metric("Activations", bytes_to_human_readable(activation_memory))
        st.metric("Overhead & Fragmentation", bytes_to_human_readable(total_memory - model_params_memory - optimizer_memory - gradient_memory - activation_memory))
        st.metric("Total VRAM Required", bytes_to_human_readable(total_memory))
    
    # Display efficiency metrics
    st.subheader("Training Efficiency Metrics")
    
    effective_batch_size = batch_size * gradient_accumulation
    st.info(f"Effective Batch Size: {effective_batch_size} sequences")
    
    tokens_per_batch = effective_batch_size * sequence_length
    st.info(f"Tokens per Batch: {tokens_per_batch:,} tokens")
    
    vram_per_billion_params = total_memory / (model_size * 1e9)
    st.info(f"VRAM per Billion Parameters: {bytes_to_human_readable(vram_per_billion_params)}")

# Fine-tuning tab
with tab2:
    st.header("Fine-tuning LLMs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Base Model")
        
        ft_model_size = st.number_input(
            "Base Model Size (billions)",
            min_value=0.1,
            max_value=1000.0,
            value=7.0,
            step=0.1,
            key="ft_model_size",
            help="Total number of parameters in billions (e.g., 7B, 13B, 70B)"
        )
        
        ft_method = st.selectbox(
            "Fine-tuning Method",
            options=["Full Fine-tuning", "LoRA", "QLoRA"],
            index=1,
            help="Method used for fine-tuning the model"
        )
        
        if ft_method in ["LoRA", "QLoRA"]:
            lora_rank = st.number_input(
                "LoRA Rank",
                min_value=1,
                max_value=256,
                value=8,
                step=1,
                help="Rank of low-rank adaptation matrices"
            )
            
            lora_alpha = st.number_input(
                "LoRA Alpha",
                min_value=1,
                max_value=256,
                value=16,
                step=1,
                help="LoRA scaling factor (typically 2x rank)"
            )
        
        if ft_method == "QLoRA":
            base_bits = st.selectbox(
                "Base Model Quantization",
                options=["4-bit", "8-bit"],
                index=0,
                help="Bit precision for quantizing the base model"
            )
        else:
            base_precision = st.selectbox(
                "Base Model Precision",
                options=["FP32 (4 bytes)", "FP16 (2 bytes)", "BF16 (2 bytes)"],
                index=1,
                key="ft_precision",
                help="Numerical precision for the base model"
            )
        
    with col2:
        st.subheader("Fine-tuning Configuration")
        
        ft_batch_size = st.number_input(
            "Batch Size (sequences)",
            min_value=1,
            max_value=512,
            value=8,
            step=1,
            key="ft_batch_size",
            help="Number of sequences processed simultaneously"
        )
        
        ft_sequence_length = st.number_input(
            "Sequence Length (tokens)",
            min_value=128,
            max_value=32768,
            value=2048,
            step=128,
            key="ft_sequence_length",
            help="Maximum number of tokens per sequence"
        )
        
        ft_gradient_accumulation = st.number_input(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=64,
            value=4,
            step=1,
            key="ft_grad_accum",
            help="Number of steps to accumulate gradients before updating weights"
        )
        
        ft_hidden_size = st.number_input(
            "Hidden Size",
            min_value=768,
            max_value=12288,
            value=4096,
            step=128,
            key="ft_hidden_size",
            help="Dimension of the model's hidden representations"
        )
        
        ft_num_layers = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=128,
            value=32,
            step=1,
            key="ft_num_layers",
            help="Number of transformer layers in the model"
        )
    
    # Calculate memory requirements
    # Parse base model precision
    if ft_method == "QLoRA":
        if "4-bit" in base_bits:
            base_param_bytes = 0.5  # 4-bit = 0.5 bytes per parameter
        else:  # 8-bit
            base_param_bytes = 1  # 8-bit = 1 byte per parameter
    else:
        if "FP32" in base_precision:
            base_param_bytes = 4
        else:  # FP16 or BF16
            base_param_bytes = 2
    
    # Base model memory
    base_model_memory = ft_model_size * 1e9 * base_param_bytes
    
    # Calculate trainable parameters
    if ft_method == "Full Fine-tuning":
        trainable_params = ft_model_size * 1e9
        optimizer_bytes = 10  # Adam/AdamW average
        optimizer_memory = trainable_params * optimizer_bytes
        additional_memory = 0
    elif ft_method in ["LoRA", "QLoRA"]:
        # Approximation for LoRA parameters (2 * rank * (d_in + d_out) for each adapter)
        # For transformers, adapters are typically applied to query, key, value, and output projections
        params_per_layer = 4 * (2 * lora_rank * (ft_hidden_size + ft_hidden_size))
        trainable_params = params_per_layer * ft_num_layers
        optimizer_bytes = 10  # Adam/AdamW average
        optimizer_memory = trainable_params * optimizer_bytes
        
        # Additional memory for QLoRA (dequantization buffers)
        if ft_method == "QLoRA":
            additional_memory = ft_model_size * 1e9 * 0.05  # Roughly 5% overhead for dequantization
        else:
            additional_memory = 0
    
    # Activation memory (reduced for LoRA methods)
    if ft_method == "Full Fine-tuning":
        activation_factor = 1.5  # With activation checkpointing
    else:  # LoRA or QLoRA
        activation_factor = 0.5  # Much lower for parameter-efficient methods
    
    activation_memory = ft_batch_size * ft_sequence_length * ft_num_layers * ft_hidden_size * base_param_bytes * activation_factor / ft_gradient_accumulation
    
    # Gradient memory (only for trainable parameters)
    if ft_method == "Full Fine-tuning":
        gradient_memory = ft_model_size * 1e9 * base_param_bytes / ft_gradient_accumulation
    else:  # LoRA or QLoRA
        gradient_memory = trainable_params * 2 / ft_gradient_accumulation  # 2 bytes per parameter (FP16)
    
    # Memory fragmentation and overhead (typically 10-20%)
    overhead_factor = 1.15
    
    # Total memory
    total_ft_memory = (base_model_memory + optimizer_memory + gradient_memory + activation_memory + additional_memory) * overhead_factor
    
    # Display results
    st.subheader("Estimated VRAM Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Base Model", bytes_to_human_readable(base_model_memory))
        st.metric("Trainable Parameters", f"{trainable_params/1e6:.2f}M params")
        st.metric("Optimizer State", bytes_to_human_readable(optimizer_memory))
    
    with col2:
        st.metric("Activations", bytes_to_human_readable(activation_memory))
        st.metric("Gradients", bytes_to_human_readable(gradient_memory))
        if ft_method == "QLoRA":
            st.metric("Dequantization Buffers", bytes_to_human_readable(additional_memory))
        
    st.metric("Total VRAM Required", bytes_to_human_readable(total_ft_memory))
    
    # Display efficiency metrics
    st.subheader("Fine-tuning Efficiency Metrics")
    
    effective_ft_batch_size = ft_batch_size * ft_gradient_accumulation
    st.info(f"Effective Batch Size: {effective_ft_batch_size} sequences")
    
    tokens_per_ft_batch = effective_ft_batch_size * ft_sequence_length
    st.info(f"Tokens per Batch: {tokens_per_ft_batch:,} tokens")
    
    param_efficiency = 100 * trainable_params / (ft_model_size * 1e9)
    if ft_method != "Full Fine-tuning":
        st.info(f"Parameter Efficiency: Training {param_efficiency:.4f}% of base model parameters")

# Inference tab
with tab3:
    st.header("Inference with LLMs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        inf_model_size = st.number_input(
            "Model Size (billions)",
            min_value=0.1,
            max_value=1000.0,
            value=7.0,
            step=0.1,
            key="inf_model_size",
            help="Total number of parameters in billions (e.g., 7B, 13B, 70B)"
        )
        
        inf_precision = st.selectbox(
            "Model Precision",
            options=[
                "FP32 (4 bytes)",
                "FP16/BF16 (2 bytes)",
                "INT8 (1 byte)",
                "INT4 (0.5 bytes)",
                "Mixed Precision"
            ],
            index=1,
            help="Numerical precision used for model weights"
        )
        
        if inf_precision == "Mixed Precision":
            mixed_precision_ratio = st.slider(
                "Ratio of INT4:INT8:FP16",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Proportion of weights in INT4 format (remaining split between INT8 and FP16)"
            )
            int8_ratio = st.slider(
                "Ratio of INT8 in non-INT4 weights",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Proportion of non-INT4 weights in INT8 format (remainder in FP16)"
            )
        
        attn_implementation = st.selectbox(
            "Attention Implementation",
            options=["Multi-head Attention (MHA)", "Multi-query Attention (MQA)", "Grouped-query Attention (GQA)"],
            index=0,
            help="Type of attention mechanism used"
        )
        
        if attn_implementation == "Grouped-query Attention (GQA)":
            gqa_groups = st.number_input(
                "Number of GQA Groups",
                min_value=1,
                max_value=128,
                value=8,
                step=1,
                help="Number of query groups in GQA"
            )
        
        kv_quantization = st.selectbox(
            "KV Cache Quantization",
            options=["None (same as model precision)", "INT8", "INT4"],
            index=0,
            help="Quantization applied to the KV cache"
        )
        
    with col2:
        st.subheader("Inference Configuration")
        
        inf_batch_size = st.number_input(
            "Batch Size (concurrent sequences)",
            min_value=1,
            max_value=128,
            value=1,
            step=1,
            key="inf_batch_size",
            help="Number of sequences processed simultaneously"
        )
        
        context_length = st.number_input(
            "Maximum Context Length (tokens)",
            min_value=128,
            max_value=128000,
            value=8192,
            step=128,
            help="Maximum context window size in tokens"
        )
        
        inf_hidden_size = st.number_input(
            "Hidden Size",
            min_value=768,
            max_value=12288,
            value=4096,
            step=128,
            key="inf_hidden_size",
            help="Dimension of the model's hidden representations"
        )
        
        inf_num_layers = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=128,
            value=32,
            step=1,
            key="inf_num_layers",
            help="Number of transformer layers in the model"
        )
        
        num_heads = st.number_input(
            "Number of Attention Heads",
            min_value=1,
            max_value=128,
            value=32,
            step=1,
            help="Number of attention heads in the model"
        )
        
        use_flash_attn = st.checkbox(
            "Use Flash Attention",
            value=True,
            help="Enable optimized attention implementation that reduces memory usage"
        )
    
    # Calculate model memory based on precision
    if inf_precision == "FP32 (4 bytes)":
        model_param_bytes = 4
    elif inf_precision == "FP16/BF16 (2 bytes)":
        model_param_bytes = 2
    elif inf_precision == "INT8 (1 byte)":
        model_param_bytes = 1
    elif inf_precision == "INT4 (0.5 bytes)":
        model_param_bytes = 0.5
    else:  # Mixed Precision
        # Calculate weighted average of bytes per parameter
        int4_bytes = 0.5
        int8_bytes = 1
        fp16_bytes = 2
        
        non_int4_ratio = 1 - mixed_precision_ratio
        int8_part = non_int4_ratio * int8_ratio
        fp16_part = non_int4_ratio * (1 - int8_ratio)
        
        model_param_bytes = (mixed_precision_ratio * int4_bytes) + (int8_part * int8_bytes) + (fp16_part * fp16_bytes)
    
    # Model weights memory
    model_weights_memory = inf_model_size * 1e9 * model_param_bytes
    
    # KV cache memory calculation
    # Determine KV cache precision
    if kv_quantization == "None (same as model precision)":
        if inf_precision == "Mixed Precision":
            kv_bytes = 2  # Default to FP16 for KV cache
        elif inf_precision == "FP32 (4 bytes)":
            kv_bytes = 4
        else:
            kv_bytes = model_param_bytes
    elif kv_quantization == "INT8":
        kv_bytes = 1
    else:  # INT4
        kv_bytes = 0.5
    
    # KV cache size based on attention implementation
    head_dim = inf_hidden_size // num_heads
    
    if attn_implementation == "Multi-head Attention (MHA)":
        # Both K and V stored for each head
        kv_cache_size = 2 * inf_batch_size * inf_num_layers * context_length * inf_hidden_size * kv_bytes
    elif attn_implementation == "Multi-query Attention (MQA)":
        # K and V shared across heads, so only one copy per layer
        kv_cache_size = inf_batch_size * inf_num_layers * context_length * (head_dim * 2) * kv_bytes
    else:  # Grouped-query Attention (GQA)
        # K and V shared within groups
        kv_per_group = inf_hidden_size // gqa_groups
        kv_cache_size = inf_batch_size * inf_num_layers * context_length * (kv_per_group * 2) * kv_bytes
    
    # Activation memory during inference
    # Flash attention significantly reduces memory footprint
    if use_flash_attn:
        activation_factor = 0.1
    else:
        activation_factor = 0.3
    
    # During inference, activations are much smaller than during training
    activation_memory_inference = inf_batch_size * inf_num_layers * inf_hidden_size * activation_factor * 2  # 2 bytes (FP16)
    
    # Memory overhead for scratch space and fragmentation
    overhead_factor_inference = 1.1
    
    # Total inference memory
    total_inference_memory = (model_weights_memory + kv_cache_size + activation_memory_inference) * overhead_factor_inference
    
    # Display results
    st.subheader("Estimated VRAM Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Weights", bytes_to_human_readable(model_weights_memory))
        st.metric("KV Cache", bytes_to_human_readable(kv_cache_size))
    
    with col2:
        st.metric("Activation Memory", bytes_to_human_readable(activation_memory_inference))
        st.metric("Overhead & Scratch Space", bytes_to_human_readable((total_inference_memory - model_weights_memory - kv_cache_size - activation_memory_inference)))
    
    st.metric("Total VRAM Required", bytes_to_human_readable(total_inference_memory))
    
    # Display efficiency metrics
    st.subheader("Inference Efficiency Metrics")
    
    kv_percent = 100 * kv_cache_size / total_inference_memory
    st.info(f"KV Cache Percentage: {kv_percent:.2f}% of total memory")
    
    tokens_per_mb = context_length / (kv_cache_size / 1024 / 1024)
    st.info(f"Tokens per MB of KV Cache: {tokens_per_mb:.2f} tokens/MB")
    
    # Calculate max possible batch size based on common GPU VRAM sizes
    common_gpus = {
        "RTX 3090 (24GB)": 24 * 1024**3,
        "RTX 4090 (24GB)": 24 * 1024**3,
        "A6000 (48GB)": 48 * 1024**3,
        "A100 (40GB)": 40 * 1024**3,
        "A100 (80GB)": 80 * 1024**3,
        "H100 (80GB)": 80 * 1024**3
    }
    
    # Calculate max batch size for each GPU
    gpu_max_batches = {}
    for gpu, vram in common_gpus.items():
        # Base memory without batch-dependent components
        base_memory = model_weights_memory * overhead_factor_inference
        
        # Per-batch memory components
        per_batch_kv = kv_cache_size / inf_batch_size
        per_batch_activation = activation_memory_inference / inf_batch_size
        
        # Available memory for batches
        available_for_batches = vram - base_memory
        
        # Max possible batch size (approximate)
        if available_for_batches > 0:
            max_batch = int(available_for_batches / (per_batch_kv + per_batch_activation))
            # Minimum batch size is 1
            max_batch = max(1, max_batch)
        else:
            max_batch = 0
        
        gpu_max_batches[gpu] = max_batch
    
    # Display max batch sizes
    st.subheader("Maximum Batch Size by GPU")
    
    batch_data = pd.DataFrame({
        "GPU": list(gpu_max_batches.keys()),
        "Max Batch Size": list(gpu_max_batches.values())
    })
    
    # Filter out GPUs that can't handle the model
    viable_gpus = batch_data[batch_data["Max Batch Size"] > 0]
    
    if not viable_gpus.empty:
        st.bar_chart(viable_gpus.set_index("GPU"))
    else:
        st.warning("The model is too large to fit on the listed GPUs with the current configuration.")

# Show footer
st.markdown("---")
st.markdown("""
**Note**: This calculator provides estimations based on commonly observed patterns and theoretical calculations.
Actual VRAM usage may vary depending on specific implementations, libraries, and hardware configurations.
""")