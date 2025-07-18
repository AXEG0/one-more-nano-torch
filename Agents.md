# Development Plan

This project aims to implement a tiny Vision Transformer in pure Python.
The following directory structure outlines the planned modules and files.

```
tinyvit-pure/
├── README.md                ← high-level goals, setup, roadmap checklist
├── pyproject.toml           ← optional; only for dev tooling (pytest, black); runtime will stay stdlib
├── .gitignore
│
├── data/
│   ├── mnist.py             ← tiny MNIST loader (pure-Python IDX reader)
│   └── cache/               ← downloaded *.gz kept here
│
├── src/
│   ├── core/
│   │   ├── tensor.py        ← Stage 1: TorchShim → Stage 4+: Pure Tensor
│   │   ├── storage.py       ← Stage 3: flat-buffer backend (no external libs)
│   │   └── autograd.py      ← Stage 6: backward engine
│   │
│   ├── ops/                 ← elemental operations
│   │   ├── matmul.py
│   │   ├── elementwise.py
│   │   ├── reduction.py
│   │   └── __init__.py
│   │
│   ├── nn/                  ← “layers”, all backend-agnostic
│   │   ├── functional.py    ← relu, gelu, softmax, log_softmax …
│   │   ├── linear.py
│   │   ├── layernorm.py
│   │   ├── mha.py           ← multi-head attention
│   │   ├── ffn.py
│   │   ├── encoder.py
│   │   └── patch_embed.py
│   │
│   ├── models/
│   │   └── tiny_vit.py      ← stitches layers together
│   │
│   ├── optim/
│   │   ├── sgd.py
│   │   └── adamw.py
│   │
│   ├── training/
│   │   ├── loop.py          ← generic fit/eval helpers
│   │   └── metrics.py
│   │
│   ├── backends/            ← toggle at import time
│   │   ├── torch_backend.py ← thin wrappers around torch ops (Stages 0-2)
│   │   └── python_backend.py← pure-Python reference (Stage 4+)
│   │
│   └── utils/
│       ├── seed.py
│       └── profiler.py
│
├── examples/
│   └── colab_torch_subset.ipynb  ← the current torch baseline (subset ready)
│
├── tests/                   ← pytest unit tests
│   ├── test_tensor.py
│   ├── test_ops.py
│   ├── test_autograd.py
│   └── test_layers.py
│
└── scripts/
    ├── train_torch.py       ← Stage 0 baseline training (uses torch backend)
    ├── train_python.py      ← Stage 10+ pure-Python training
    └── export_weights.py    ← helper to copy weights between backends
```

