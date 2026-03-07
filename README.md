# Efficient Transformers

A high-performance PyTorch library implementing optimized transformer components with custom CUDA kernels, memory-efficient attention mechanisms, and production-ready inference optimizations.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Key Features

- **Flash Attention**: Custom Triton implementation achieving 2-4x speedup over standard attention
- **Fused Kernels**: LayerNorm+Dropout, GELU activation fusion for reduced memory bandwidth
- **Memory-Efficient Variants**: Linear Attention, Reformer-style LSH attention
- **Custom Autograd**: Hand-written backward passes optimized for memory and speed
- **KV-Cache**: Efficient inference with key-value caching for autoregressive generation
- **Quantization Support**: INT8/FP16 quantization-aware training
- **Production Ready**: JIT compilation, ONNX export, distributed training support

## 🚀 Performance Benchmarks

| Model | Attention Type | Memory (GB) | Speed (ms/batch) | Speedup |
|-------|---------------|-------------|------------------|---------|
| GPT-2 Small | Standard | 8.2 | 145 | 1.0x |
| GPT-2 Small | Flash Attention | 4.1 | 52 | **2.8x** |
| GPT-2 Small | Linear Attention | 3.8 | 38 | **3.8x** |
| ViT-Base | Standard | 12.4 | 89 | 1.0x |
| ViT-Base | Flash Attention | 6.8 | 34 | **2.6x** |

*Benchmarked on NVIDIA A100 (40GB), batch size 32, sequence length 512*

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/raghusai-09/efficient-transformers-library.git
cd efficient-transformers

# Install dependencies
pip install -r requirements.txt

# Install Triton for Flash Attention (optional but recommended)
pip install triton

# Install the package
pip install -e .
```

## 🔧 Quick Start

### Flash Attention

```python
import torch
from efficient_transformers import FlashAttention

# Initialize Flash Attention layer
attn = FlashAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    causal=True  # For autoregressive models
).cuda()

# Forward pass
x = torch.randn(32, 128, 512).cuda()  # (batch, seq_len, embed_dim)
output, attn_weights = attn(x)

# Memory-efficient: uses ~50% less GPU memory than standard attention
```

### Memory-Efficient Transformer Block

```python
from efficient_transformers import EfficientTransformerBlock

# Full transformer block with fused operations
block = EfficientTransformerBlock(
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4.0,
    use_flash_attn=True,
    use_fused_mlp=True,
    checkpoint_gradients=True  # For even lower memory usage
).cuda()

x = torch.randn(16, 512, 768).cuda()
output = block(x)
```

### Inference with KV-Cache

```python
from efficient_transformers import CachedTransformer

model = CachedTransformer(
    vocab_size=50257,
    num_layers=12,
    embed_dim=768,
    num_heads=12
).cuda()

# Enable KV-caching for fast autoregressive generation
output = model.generate(
    input_ids,
    max_length=100,
    use_cache=True,  # 3-5x faster inference
    temperature=0.9
)
```

## 🏗️ Architecture Components

### 1. Flash Attention (Triton Implementation)

Custom implementation of [Flash Attention](https://arxiv.org/abs/2205.14135) using Triton kernels:
- Tiled computation to fit in SRAM
- Fused softmax and dropout
- Recomputation in backward pass
- Supports causal and bidirectional variants

### 2. Fused Kernels

**LayerNorm + Dropout Fusion**
```python
from efficient_transformers.kernels import fused_ln_dropout

# Single kernel call instead of two separate operations
output = fused_ln_dropout(x, gamma, beta, dropout_p=0.1)
# ~1.5x faster, reduces memory traffic
```

**GELU Activation Fusion**
```python
from efficient_transformers.kernels import fused_gelu

# Fused with upstream operations
output = fused_gelu(x)  # 1.3x faster than separate ops
```

### 3. Linear Attention

Approximates attention in O(N) complexity instead of O(N²):
```python
from efficient_transformers import LinearAttention

# For long sequences (>2048 tokens)
attn = LinearAttention(embed_dim=512, num_heads=8)
output = attn(x)  # Works with sequences up to 16k tokens
```

### 4. Custom Backward Passes

```python
class EfficientAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p):
        # Memory-efficient forward pass
        # Only save minimal state for backward
        ...
        
    @staticmethod
    def backward(ctx, grad_output):
        # Recompute activations instead of storing
        # Reduces memory by 3-4x
        ...
```

## 📊 Detailed Benchmarks

### Memory Usage Comparison

![Memory Usage Graph](assets/memory_comparison.png)

### Training Speed

```bash
# Run benchmarks yourself
python benchmarks/benchmark_attention.py --batch-size 32 --seq-len 512
python benchmarks/benchmark_transformers.py --model gpt2 --implementation [standard|flash|linear]
```

Example output:
```
Running benchmark: GPT-2 (124M params)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standard Attention:
  Memory: 8.2 GB | Speed: 145 ms/batch
  
Flash Attention:
  Memory: 4.1 GB (↓50%) | Speed: 52 ms/batch (↑2.8x)
  
Linear Attention:
  Memory: 3.8 GB (↓54%) | Speed: 38 ms/batch (↑3.8x)
```

## 🧪 Testing & Validation

All implementations include:
- **Gradient checks**: Verify custom backward passes match PyTorch autograd
- **Numerical stability tests**: Check for NaN/Inf in extreme cases
- **Correctness tests**: Compare outputs against reference implementations
- **Performance regression tests**: Track speed/memory over commits

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_flash_attention.py -v

# Test with gradient checking
pytest tests/ --check-gradients
```

## 🔬 Implementation Details

### Flash Attention Algorithm

Our Triton implementation follows the algorithm from the original paper:

1. **Tiling**: Split Q, K, V into blocks that fit in SRAM
2. **Online Softmax**: Compute softmax statistics incrementally
3. **Recomputation**: Don't store attention matrix, recompute in backward
4. **Causal Masking**: Efficient causal attention without explicit masks

See [`efficient_transformers/attention/flash_attention.py`](efficient_transformers/attention/flash_attention.py) for full implementation.

### Custom CUDA Kernels

For operations where Triton isn't optimal, we provide CUDA kernels:
- Fused LayerNorm + Dropout
- Fused Bias + GELU
- Optimized RoPE (Rotary Position Embeddings)

Build CUDA extensions:
```bash
cd cuda_kernels
python setup.py install
```

## 🎓 Educational Resources

This repository includes detailed documentation explaining:
- How PyTorch autograd works under the hood
- Memory management in transformer training
- CUDA kernel optimization techniques
- Profiling and debugging GPU code

See [`docs/`](docs/) directory for tutorials.

## 🛠️ Development

### Project Structure

```
efficient-transformers/
├── efficient_transformers/
│   ├── attention/
│   │   ├── flash_attention.py      # Triton Flash Attention
│   │   ├── linear_attention.py     # O(N) linear attention
│   │   └── standard_attention.py   # Baseline implementation
│   ├── kernels/
│   │   ├── fused_ops.py           # Fused operations (Triton)
│   │   └── cuda/                  # Custom CUDA kernels
│   ├── layers/
│   │   ├── transformer_block.py   # Full transformer block
│   │   └── mlp.py                 # Feed-forward network
│   └── utils/
│       ├── kv_cache.py           # KV-cache for inference
│       └── profiling.py          # Performance profiling tools
├── benchmarks/                    # Performance benchmarks
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── examples/                     # Usage examples
```

### Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-optimization`)
3. Add tests for new functionality
4. Run benchmarks to verify performance improvements
5. Submit a pull request

## 📈 Roadmap

- [ ] FlashAttention-2 implementation
- [ ] Multi-query attention (MQA) and grouped-query attention (GQA)
- [ ] PagedAttention for inference serving
- [ ] Support for AMD GPUs (ROCm)
- [ ] Integration with HuggingFace Transformers
- [ ] Mobile deployment (quantization + pruning)
- [ ] Sparse attention patterns (Longformer, BigBird)

## 📚 References

- [Flash Attention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [Flash Attention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023  
- [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) - Rabe & Staats, 2021
- [Transformer Quality in Linear Time](https://arxiv.org/abs/2202.10447) - Hua et al., 2022

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Tri Dao for the Flash Attention algorithm
- OpenAI for Triton language
- PyTorch team for the excellent deep learning framework

## 📧 Contact

Questions? Open an issue or reach out:
- Email: kosanaraghusai@gmail.com
- LinkedIn: Raghu Sai Kosana (https://linkedin.com/in/raghusai09)

---

**⭐ If you find this useful, please star the repository!**
