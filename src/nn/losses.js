// src/nn/losses.js
import { Tensor } from '../tensor.js';
import { NDArray } from '../ndarray.js';

const _t = (x) => (x instanceof Tensor ? x : Tensor.of(x));

/** Mean squared error. */
export function mseLoss(pred, target) {
  const p = _t(pred), y = _t(target);
  const d = p.sub(y);
  return d.mul(d).mean();
}

/** Binary cross-entropy. `pred` should already be sigmoid probabilities. */
export function bceLoss(pred, target, eps = 1e-7) {
  const p = _t(pred), y = _t(target);
  // -[y log p + (1-y) log(1-p)]
  const one = Tensor.of(1);
  const a = y.mul(p.add(eps).log());
  const b = one.sub(y).mul(one.sub(p).add(eps).log());
  return a.add(b).neg().mean();
}

/** Cross-entropy from logits. `target` is an array (or 1-D Tensor) of class indices. */
export function crossEntropyLoss(logits, target) {
  const p = _t(logits);
  const ls = p.logSoftmax(); // (N, C)
  const tArr = target instanceof Tensor ? target.toArray() : target;
  const N = ls.shape[0], C = ls.shape[1];
  const oneHot = NDArray.zeros([N, C]);
  for (let i = 0; i < N; i++) oneHot.data[i * C + tArr[i]] = 1;
  return ls.mul(Tensor.of(oneHot)).sum(1).mean().neg();
}
