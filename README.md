# BallsDeep 🧠⚙️

A small, dependency-free numerical computing and machine-learning library for
**Node.js and the browser**, written in modern ES modules. Think of it as a
mini-NumPy + mini-PyTorch + mini-scikit-learn — all in one package, all in
pure JS.

> v1 was a single hand-rolled sigmoid net. v2 is a real layered library. See
> [`PLAN.md`](./PLAN.md) for the architecture and the v1 → v2 rewrite story.
> The original code lives untouched in [`legacy/`](./legacy/).

## Install

```bash
npm install ballsdeep        # once published
# or use directly from the repo:
git clone https://github.com/rwrife/BallsDeep
```

Requires **Node.js ≥ 18**. In the browser, use a bundler (Vite, esbuild, etc.)
or import directly as ES modules — there are no native deps.

## What's in the box

| Layer    | Module     | Highlights                                           |
| -------- | ---------- | ---------------------------------------------------- |
| Core     | `NDArray`  | Strided n-d array on `Float64Array`, broadcasting, reshape, transpose. |
| Core     | `ops`      | Element-wise math, reductions (`sum/mean/max/min`), `matmul`, `argmax`. |
| Core     | `RNG`      | Seedable mulberry32 + normal/uniform/bernoulli sampling.               |
| Autograd | `Tensor`   | PyTorch-style dynamic graph + `.backward()`.                           |
| NN       | `nn`       | `Module`, `Linear`, `Sequential`, `ReLU/Sigmoid/Tanh/Softmax`, losses. |
| Optim    | `optim`    | `SGD` (+ momentum + weight decay), `Adam`.                             |
| Models   | `models`   | `LinearRegression`, `LogisticRegression`, `KNeighborsClassifier/Regressor`. |
| Data     | `data`     | CSV parser, `StandardScaler`/`MinMaxScaler`, `trainTestSplit`, one-hot, accuracy. |

## 30-second tour

### NumPy-style arrays

```js
import { NDArray, ops } from 'ballsdeep';

const a = NDArray.fromArray([[1, 2, 3], [4, 5, 6]]);   // (2,3)
const b = NDArray.fromArray([10, 20, 30]);             // (3,) broadcasts
console.log(ops.add(a, b).toArray());
// [[11, 22, 33], [14, 25, 36]]

console.log(ops.matmul(a, a.T).toArray());
// [[14, 32], [32, 77]]
```

### Autograd

```js
import { tensor } from 'ballsdeep';

const x = tensor(3, true);
const y = tensor(4, true);
const z = x.mul(y).add(x);   // 3*4 + 3
z.backward();
console.log(x.grad.toArray()); // 5
console.log(y.grad.toArray()); // 3
```

### Train an MLP on XOR (PyTorch flavor)

```js
import { Tensor, nn, optim, RNG } from 'ballsdeep';

const rng = new RNG(42);
const model = new nn.Sequential(
  new nn.Linear(2, 8, { rng }),
  new nn.ReLU(),
  new nn.Linear(8, 1, { rng }),
  new nn.Sigmoid(),
);

const X = Tensor.of([[0,0],[0,1],[1,0],[1,1]]);
const y = Tensor.of([[0],[1],[1],[0]]);
const opt = new optim.Adam(model.parameters(), { lr: 0.1 });

for (let i = 0; i < 1500; i++) {
  opt.zeroGrad();
  const loss = nn.bceLoss(model.forward(X), y);
  loss.backward();
  opt.step();
}
console.log(model.forward(X).toArray());
```

### Classical ML (sklearn flavor)

```js
import { models, data } from 'ballsdeep';

const X = [[0,0],[0,1],[1,0],[1,1]];
const y = [0, 1, 1, 0]; // not linearly separable, but illustrative

const lr = new models.LogisticRegression({ lr: 0.5, epochs: 1000 }).fit(X, y);
console.log(lr.predict(X));         // hard 0/1 predictions
console.log(lr.predictProba(X));    // probabilities

const knn = new models.KNeighborsClassifier({ k: 3 }).fit(X, y);
console.log(knn.predict([[0.1, 0.1]])); // -> [0]
```

### Load CSV → train → score

```js
import { readFileSync } from 'node:fs';
import { data, models } from 'ballsdeep';

const csv = readFileSync('binary.csv', 'utf8');
const { X, y } = data.csvToXY(csv, 'label');

const split = data.trainTestSplit(X, y, { testSize: 0.25 });
const m = new models.LogisticRegression({ epochs: 1000 }).fit(split.XTrain, split.yTrain);
console.log('test acc:', data.accuracy(m.predict(split.XTest), split.yTest));
```

## Examples

```bash
npm run example:xor        # MLP learns XOR via autograd
npm run example:binary     # logistic regression on binary.csv
npm run example:linear     # closed-form linear regression
npm run example:periodic   # KNN regressor predicts atomic weight
```

## Tests

```bash
npm test
```

22 tests covering NDArray broadcasting, matmul, reductions, autograd
(including a finite-difference gradient check), MLP training, classical
models, data utilities, and CSV parsing.

## Design notes

- **Pure JS, zero deps.** Same code runs in Node and modern browsers.
- **`Float64Array` storage** for cache-friendly numerical work.
- **Dynamic autograd** (define-by-run) — small, hackable, PyTorch-style.
- **Layered API.** Use whichever level fits: NDArray, Tensor + autograd,
  `nn` + `optim`, or the high-level `models.*` classifiers/regressors.

## Roadmap

See [`PLAN.md`](./PLAN.md) for the full plan. Future work that is **not** in
this version: GPU/WebGPU backend, conv/RNN/Transformer layers, sparse
tensors, model serialization, TypeScript types.

## Repo layout

```
src/
  ndarray.js       # n-d array + broadcasting
  ops.js           # numerical ops
  random.js        # seedable RNG
  tensor.js        # autograd
  nn/              # Module, Linear, activations, losses
  optim/           # SGD, Adam
  data/            # CSV, scalers, split, one-hot
  models/          # LinearRegression, LogisticRegression, KNN
test/              # node --test
examples/          # runnable scripts
legacy/            # original v1 code, kept for reference
PLAN.md            # architecture + roadmap
```

## License

ISC. See [LICENSE](./LICENSE) if present, otherwise the SPDX `ISC` identifier
in `package.json`.
