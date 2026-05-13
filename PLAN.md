# BallsDeep v2 — Expansion Plan

> Goal: Turn the v1 toy 1-layer NN demo into a small but real
> NumPy + PyTorch-flavored numerical / ML library that runs in
> Node.js **and** the browser with zero runtime dependencies.

## Why rewrite

v1 was a learning sketch: stringly-typed `Matrix` with closures, a
single hardcoded sigmoid network, ad-hoc CSV loading, no tests, no
broadcasting, no autograd. It worked for one demo and stopped there.

v2 is a real layered library:

```
NDArray (numpy-ish) → Tensor + autograd (torch-ish)
        ↓                       ↓
   stats / random        nn (layers, losses, optim)
        ↓                       ↓
            classical models (sklearn-ish)
                        ↓
              data utils (csv, scaler, splits)
```

## Architecture

| Layer     | Module                  | Role                                                                 |
| --------- | ----------------------- | -------------------------------------------------------------------- |
| Core      | `src/ndarray.js`        | Strided n-d array on `Float64Array`, broadcasting, slicing, reshape. |
| Core      | `src/ops.js`            | Element-wise + reduction + linalg ops on `NDArray`.                  |
| Core      | `src/random.js`         | Seedable RNG (mulberry32) + normal/uniform/bernoulli.                |
| Autograd  | `src/tensor.js`         | `Tensor` wraps `NDArray`, builds dynamic graph, `.backward()`.       |
| NN        | `src/nn/module.js`      | Base `Module` with parameter tracking, `.forward`, `.parameters()`.  |
| NN        | `src/nn/linear.js`      | `Linear`, `Sequential`.                                              |
| NN        | `src/nn/activations.js` | `ReLU`, `Sigmoid`, `Tanh`, `Softmax` as Tensor ops & modules.        |
| NN        | `src/nn/losses.js`      | MSE, BCE, Cross-entropy.                                             |
| Optim     | `src/optim/sgd.js`      | SGD + momentum.                                                      |
| Optim     | `src/optim/adam.js`     | Adam.                                                                |
| Models    | `src/models/*.js`       | `LinearRegression`, `LogisticRegression`, `KNN` — sklearn-style.     |
| Data      | `src/data/csv.js`       | Browser/Node CSV parser, no deps.                                    |
| Data      | `src/data/scaler.js`    | `StandardScaler`, `MinMaxScaler`.                                    |
| Data      | `src/data/dataset.js`   | `trainTestSplit`, batching, one-hot.                                 |

## Design rules

- ES modules everywhere. `"type": "module"`.
- Zero runtime deps. Pure JS, runs in Node ≥18 and modern browsers.
- All numeric storage is `Float64Array` for cache-friendliness.
- Immutable-by-default ops; in-place variants are `_`-suffixed
  (`add_`, `mul_`) when needed by optimizers.
- Broadcasting follows NumPy rules (right-aligned, 1-axes expand).
- `Tensor` graph is dynamic (PyTorch-style), built during forward.
- No globals. Each `Module` and optimizer is a plain class.

## Roadmap (this PR delivers ✅)

- ✅ NDArray with shape/strides/broadcasting/reshape/transpose.
- ✅ Element-wise + reductions + matmul.
- ✅ Seeded RNG.
- ✅ Tensor autograd: add/sub/mul/div/neg/matmul/sum/mean/relu/sigmoid/tanh/exp/log/pow/softmax/log_softmax/cross_entropy.
- ✅ `Module`, `Linear`, `Sequential`, activations as modules.
- ✅ Losses: MSE, BCE, CE.
- ✅ Optimizers: SGD (+momentum), Adam.
- ✅ Classical: `LinearRegression` (closed form), `LogisticRegression`
  (gradient via Tensor), `KNN`.
- ✅ Data utils: CSV, scalers, splits, one-hot.
- ✅ Tests via `node --test`.
- ✅ Examples: XOR (autograd MLP), binary classifier, linear regression,
  periodic-table feature regression.

## Future (deliberately out of scope here)

- GPU / WebGPU backend.
- Conv / RNN / Transformer layers.
- Sparse tensors.
- Serialization / model export.
- TypeScript types (`.d.ts`).

These are easy to bolt on because the core boundary is clean:
swap `NDArray`'s storage backend and the rest keeps working.
