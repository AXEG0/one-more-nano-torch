## Implementation Roadmap (baseline script: `scripts/tinyvit_mnist_pytorch.py`)

0. **Baseline freeze**

   * Keep the existing PyTorch script exactly as is (train = 2 000, test = 1 000).
   * Save one random batch `(X_ref, y_ref)` and its logits after epoch 0 – we’ll reuse these to compare outputs.
   * Note: accuracy \~ 70 % on the subset is sufficient; we are not chasing SOTA numbers.

1. **Introduce `Tensor` shim (Torch backend)**

   * Create `src/core/tensor.py`: a thin wrapper around `torch.Tensor` with `.data`, `.grad`, `.requires_grad`.
   * Refactor the model, loss and training loop to **import this `Tensor`** instead of using `torch.Tensor` directly.
   * Sanity check: run the original script (still on Torch backend) – logs and accuracy must stay within 0.01 of the frozen baseline.

2. **Backend selector**

   * Add `src/backends/torch_backend.py` mapping every operation (`matmul`, `add`, `relu`, …) to `torch.*`.
   * Implement a `set_backend(name)` helper; default is `"torch"`.
   * The rest of the code now calls ops through the backend module.
   * Validation: training output identical to Stage 1.

3. **Python storage (forward only)**

   * Implement `src/core/storage.py`: flat `list[float]`, shape, strides, helpers `zeros`, `randn`.
   * Build `src/backends/python_backend.py` with element‑wise ops and simple 2‑D `matmul` **forward pass only**.
   * Switch `set_backend("python")` and run **inference‑only** on `X_ref`; differences to Torch logits should be < 1e‑4.

4. **Elementary ops + broadcast**

   * Fill out `python_backend.py` with `add`, `mul`, `sum(axis)`, `mean`, broadcast rules, etc.
   * Confirm forward parity on `X_ref` again.

5. **Autograd engine**

   * Write `src/core/autograd.py` (tape + backward traversal).
   * Provide gradient functions for each op implemented so far.
   * Gradient test: scalar function `f(x)=x³` – compare analytical vs finite difference.

6. **Basic layers (Linear, ReLU, Softmax, LayerNorm)**

   * Re‑implement these layers on the pure‑Python backend using the new ops and autograd.
   * Unit test: tiny MLP should learn XOR in < 1 000 steps.

7. **Single‑head attention**

   * Code Q/K/V projection, attention scores, softmax, context, output projection.
   * Compare forward and backward on a toy sequence to Torch within 1e‑3.

8. **Multi‑head attention + encoder block**

   * Generalise to `H>1`; add residual + pre‑LayerNorm; dropout can be a no‑op for now.
   * Encoder block output on `X_ref` should match Torch w/ single block.

9. **Patch embedding (loop, no convolution)**

   * Slice 28 × 28 images into 4 × 4 patches via nested loops, multiply by learnable weight matrix.
   * Shape check: `(batch, 49, dim)` identical to Torch.

10. **Full TinyViT forward on Python backend**

    * Assemble patch‑embed, positional embedding, encoder block(s), pooling, classifier.
    * Forward difference on `X_ref` < 1e‑3 vs Torch.

11. **Cross‑entropy loss and plain SGD**

    * Implement log‑softmax → CE loss; write simplest SGD (lr, momentum optional).
    * Train on the subset for **one epoch** – loss must decrease; accuracy may be lower than Torch but > 50 %.

12. **AdamW**

    * Add first‑ and second‑moment buffers; implement weight decay.
    * Two‑epoch run should roughly catch Torch’s 70 % benchmark.

13. **Pure‑Python MNIST loader**

    * `data/mnist.py` reads IDX files via `gzip` + `struct`; cache under `data/cache/`.
    * Replace torchvision loader in the Python training script.

14. **Training loop utility**

    * `training/loop.py` handles epochs, batching, metrics printing.
    * `scripts/train_python.py` becomes the canonical pure‑Python entry point (defaults to the subset sizes).

15. **Weight import/export**

    * Allow saving and loading parameters in a plain JSON/pickle dict.
    * Helper script copies weights from Torch run to Python model for regression checks.

16. **Unit‑test suite**

    * Add PyTest files for tensor math, autograd, attention, encoder, TinyViT forward.
    * CI target: `pytest -q` passes in < 20 s on subset data.

17. **Profiling and first optimisation pass**

    * Instrument `utils/profiler.py`; identify hotspots (likely `matmul`).
    * Blocked 8×8 matmul or simple `array('f')` vectorisation to cut runtime roughly in half.

18. **Medium dataset run (10 k / 2 k)**

    * Measure epoch time and accuracy – aim for ≥ 60 % without further tuning.

19. **Full dataset option**

    * Ensure the pipeline can iterate over all 60 k training samples (will be slow but must finish).
    * Provide command‑line flag `--full`.

20. **Documentation polish**

    * Update README with usage examples, badge, and a note that pure‑Python speed is meant for learning, not production.
    * Freeze roadmap checklist in `AGENTS.md` and tick off completed stages.

---

## Roadmap Tree

```
one-more-nano-torch/
├── data/
├── src/
│   ├── core/
│   ├── ops/
│   ├── nn/
│   ├── models/
│   ├── optim/
│   ├── training/
│   ├── backends/
│   └── utils/
├── scripts/
├── tests/
└── examples/
```

---

### Validation Rules

* **Always compare with `scripts/tinyvit_mnist_pytorch.py`** as long as any Torch code remains in the stack.
* Forward checks use the saved batch `X_ref`.
* For training stages we accept divergence in accuracy, **as long as loss goes down** and accuracy stays above random (\~10 %).
* The 2 000 / 1 000 subset and its \~70 % Torch benchmark remain our reference; we purposefully do **not** raise accuracy targets until after Stage 12.
