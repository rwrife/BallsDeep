// src/models/mlp.js
// Simple fully-connected (dense) feed-forward neural networks, sklearn-style.
//
//   const m = new MLPClassifier({ hiddenLayers: [16, 8], activation: 'relu' });
//   m.fit(X, y);
//   m.predict(X);
//
// Built on top of nn.Sequential + nn.Linear + activation modules + Adam/SGD.
import { Tensor } from '../tensor.js';
import { NDArray } from '../ndarray.js';
import { Linear, Sequential } from '../nn/linear.js';
import { ReLU, Sigmoid, Tanh } from '../nn/activations.js';
import { mseLoss, bceLoss, crossEntropyLoss } from '../nn/losses.js';
import { SGD } from '../optim/sgd.js';
import { Adam } from '../optim/adam.js';
import { defaultRng } from '../random.js';
import { oneHot } from '../data/dataset.js';

function makeActivation(name) {
  if (name == null) return null;
  switch (String(name).toLowerCase()) {
    case 'relu':    return new ReLU();
    case 'sigmoid': return new Sigmoid();
    case 'tanh':    return new Tanh();
    case 'identity':
    case 'linear':
    case 'none':    return null;
    default: throw new Error(`unknown activation: ${name}`);
  }
}

function buildMLP(inF, outF, hiddenLayers, activation, outputActivation, rng) {
  const layers = [];
  let prev = inF;
  for (const h of hiddenLayers) {
    layers.push(new Linear(prev, h, { rng }));
    const a = makeActivation(activation);
    if (a) layers.push(a);
    prev = h;
  }
  layers.push(new Linear(prev, outF, { rng }));
  const outA = makeActivation(outputActivation);
  if (outA) layers.push(outA);
  return new Sequential(...layers);
}

function makeOptimizer(name, params, lr) {
  switch ((name || 'adam').toLowerCase()) {
    case 'adam': return new Adam(params, { lr });
    case 'sgd':  return new SGD(params, { lr, momentum: 0.9 });
    default: throw new Error(`unknown optimizer: ${name}`);
  }
}

function* iterBatches(N, batchSize, rng, shuffle) {
  const idx = Array.from({ length: N }, (_, i) => i);
  if (shuffle) rng.shuffle(idx);
  const bs = batchSize && batchSize > 0 ? batchSize : N;
  for (let i = 0; i < N; i += bs) yield idx.slice(i, i + bs);
}

function gather(nd, rowIdx) {
  // nd: NDArray (N, D) → (rowIdx.length, D) NDArray copy.
  const D = nd.ndim === 1 ? 1 : nd.shape[1];
  const out = NDArray.zeros(nd.ndim === 1 ? [rowIdx.length] : [rowIdx.length, D]);
  const src = nd.contiguous();
  for (let i = 0; i < rowIdx.length; i++) {
    const r = rowIdx[i];
    if (nd.ndim === 1) {
      out.data[i] = src.data[r];
    } else {
      for (let j = 0; j < D; j++) out.data[i * D + j] = src.data[r * D + j];
    }
  }
  return out;
}

class _BaseMLP {
  constructor(opts = {}) {
    this.hiddenLayers = opts.hiddenLayers ?? [16];
    this.activation   = opts.activation   ?? 'relu';
    this.optimizer    = opts.optimizer    ?? 'adam';
    this.lr           = opts.lr           ?? 1e-2;
    this.epochs       = opts.epochs       ?? 200;
    this.batchSize    = opts.batchSize    ?? null;
    this.shuffle      = opts.shuffle      !== false;
    this.verbose      = opts.verbose      ?? false;
    this.rng          = opts.rng          || defaultRng;
    this.model = null;
    this.lossHistory = [];
  }

  parameters() { return this.model ? this.model.parameters() : []; }

  _forwardArray(X) {
    const Xn = X instanceof NDArray ? X : NDArray.fromArray(X);
    return this.model.forward(Tensor.of(Xn)).toArray();
  }

  _trainLoop(Xn, Yn, lossFn) {
    const N = Xn.shape[0];
    const opt = makeOptimizer(this.optimizer, this.model.parameters(), this.lr);
    this.lossHistory = [];
    const logEvery = Math.max(1, Math.floor(this.epochs / 10));
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let epochLoss = 0;
      let nBatches = 0;
      for (const idx of iterBatches(N, this.batchSize, this.rng, this.shuffle)) {
        const xb = Tensor.of(gather(Xn, idx));
        const yb = Tensor.of(gather(Yn, idx));
        opt.zeroGrad();
        const out = this.model.forward(xb);
        const loss = lossFn(out, yb, idx);
        loss.backward();
        opt.step();
        epochLoss += loss.toArray();
        nBatches += 1;
      }
      const avg = epochLoss / Math.max(1, nBatches);
      this.lossHistory.push(avg);
      if (this.verbose && epoch % logEvery === 0) {
        // eslint-disable-next-line no-console
        console.log(`epoch ${epoch}\tloss=${avg.toFixed(6)}`);
      }
    }
  }
}

/** Fully-connected MLP regressor. Output layer is linear (no activation). */
export class MLPRegressor extends _BaseMLP {
  fit(X, y) {
    const Xn = X instanceof NDArray ? X : NDArray.fromArray(X);
    let Yn = y instanceof NDArray ? y : NDArray.fromArray(y);
    if (Yn.ndim === 1) Yn = Yn.reshape([Yn.size, 1]);
    const inF = Xn.shape[1];
    const outF = Yn.shape[1];
    this.model = buildMLP(inF, outF, this.hiddenLayers, this.activation, null, this.rng);
    this._trainLoop(Xn, Yn, (out, yb) => mseLoss(out, yb));
    return this;
  }
  predict(X) {
    const out = this._forwardArray(X);
    // Squeeze trailing dim of 1 to mirror scalar-target regressors.
    if (Array.isArray(out) && Array.isArray(out[0]) && out[0].length === 1) {
      return out.map((row) => row[0]);
    }
    return out;
  }
}

/** Fully-connected MLP classifier. Binary → sigmoid + BCE; multi-class → softmax-CE from logits. */
export class MLPClassifier extends _BaseMLP {
  constructor(opts = {}) {
    super(opts);
    this.threshold = opts.threshold ?? 0.5;
    this.classes_ = null;
    this.binary_ = false;
  }

  fit(X, y) {
    const Xn = X instanceof NDArray ? X : NDArray.fromArray(X);
    const yArr = y instanceof NDArray ? Array.from(y.contiguous().data) : Array.from(y);
    // Discover classes (sorted for determinism).
    this.classes_ = Array.from(new Set(yArr)).sort((a, b) => a - b);
    this.binary_ = this.classes_.length <= 2;
    const inF = Xn.shape[1];

    if (this.binary_) {
      // Map class[1] → 1, class[0] → 0 (or single class → 0).
      const pos = this.classes_[this.classes_.length - 1];
      const yLabels = yArr.map((v) => (v === pos ? 1 : 0));
      const Yn = NDArray.fromArray(yLabels).reshape([yLabels.length, 1]);
      this.model = buildMLP(inF, 1, this.hiddenLayers, this.activation, 'sigmoid', this.rng);
      this._trainLoop(Xn, Yn, (out, yb) => bceLoss(out, yb));
    } else {
      // Multi-class: produce raw logits, use cross-entropy (which applies log-softmax internally).
      const classToIdx = new Map(this.classes_.map((c, i) => [c, i]));
      const yIdx = yArr.map((v) => classToIdx.get(v));
      // Encode targets so we can index back from the batch via a side channel.
      const Yn = NDArray.fromArray(yIdx); // 1-D
      this.model = buildMLP(inF, this.classes_.length, this.hiddenLayers, this.activation, null, this.rng);
      this._trainLoop(Xn, Yn.reshape([Yn.size, 1]), (out, _yb, idx) => {
        const targets = idx.map((i) => yIdx[i]);
        return crossEntropyLoss(out, targets);
      });
    }
    return this;
  }

  predictProba(X) {
    const out = this._forwardArray(X);
    if (this.binary_) {
      const probs = out.map((row) => (Array.isArray(row) ? row[0] : row));
      // Return [[p(neg), p(pos)], ...] for parity with multi-class.
      return probs.map((p) => [1 - p, p]);
    }
    // Apply softmax to logits.
    return out.map((row) => {
      const m = Math.max(...row);
      const ex = row.map((v) => Math.exp(v - m));
      const s = ex.reduce((a, b) => a + b, 0);
      return ex.map((v) => v / s);
    });
  }

  predict(X) {
    if (this.binary_) {
      const probs = this.predictProba(X).map((r) => r[1]);
      const pos = this.classes_[this.classes_.length - 1];
      const neg = this.classes_[0];
      return probs.map((p) => (p >= this.threshold ? pos : neg));
    }
    return this.predictProba(X).map((row) => {
      let best = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
      return this.classes_[best];
    });
  }
}
