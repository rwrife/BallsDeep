// src/data/dataset.js
import { NDArray } from '../ndarray.js';
import { defaultRng } from '../random.js';

/** Train/test split with shuffling. */
export function trainTestSplit(X, y, opts = {}) {
  const testSize = opts.testSize ?? 0.25;
  const rng = opts.rng || defaultRng;
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  if (opts.shuffle !== false) rng.shuffle(idx);
  const nTest = Math.max(1, Math.floor(n * testSize));
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);
  const pick = (src, ids) => ids.map((i) => src[i]);
  return {
    XTrain: pick(X, trainIdx),
    XTest:  pick(X, testIdx),
    yTrain: pick(y, trainIdx),
    yTest:  pick(y, testIdx),
  };
}

/** Yield batches of (X, y) of given size. */
export function* batches(X, y, batchSize, opts = {}) {
  const rng = opts.rng || defaultRng;
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  if (opts.shuffle !== false) rng.shuffle(idx);
  for (let i = 0; i < n; i += batchSize) {
    const slice = idx.slice(i, i + batchSize);
    yield {
      X: slice.map((j) => X[j]),
      y: slice.map((j) => y[j]),
    };
  }
}

/** One-hot encode an array of integer labels. */
export function oneHot(labels, numClasses) {
  const C = numClasses ?? (Math.max(...labels) + 1);
  const a = NDArray.zeros([labels.length, C]);
  for (let i = 0; i < labels.length; i++) a.data[i * C + labels[i]] = 1;
  return a;
}

/** Compute classification accuracy. */
export function accuracy(yPred, yTrue) {
  if (yPred.length !== yTrue.length) {
    throw new Error(`length mismatch: ${yPred.length} vs ${yTrue.length}`);
  }
  let correct = 0;
  for (let i = 0; i < yPred.length; i++) if (yPred[i] === yTrue[i]) correct++;
  return correct / yPred.length;
}
