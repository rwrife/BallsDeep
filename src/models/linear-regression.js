// src/models/linear-regression.js
// Closed-form OLS:  w = (X'X)^-1 X' y   with bias prepended.
import { NDArray } from '../ndarray.js';
import * as O from '../ops.js';

/** Solve A w = b via Gauss-Jordan. A is square (n,n). b is (n,) or (n,k). */
function solve(A, b) {
  const n = A.shape[0];
  const k = b.ndim === 1 ? 1 : b.shape[1];
  const aug = NDArray.zeros([n, n + k]);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug.data[i * (n + k) + j] = A.data[i * n + j];
    if (b.ndim === 1) aug.data[i * (n + k) + n] = b.data[i];
    else for (let j = 0; j < k; j++) aug.data[i * (n + k) + n + j] = b.data[i * k + j];
  }
  // Forward elimination with partial pivoting
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(aug.data[r * (n + k) + i]) > Math.abs(aug.data[pivot * (n + k) + i])) {
        pivot = r;
      }
    }
    if (pivot !== i) {
      for (let j = 0; j < n + k; j++) {
        const t = aug.data[i * (n + k) + j];
        aug.data[i * (n + k) + j] = aug.data[pivot * (n + k) + j];
        aug.data[pivot * (n + k) + j] = t;
      }
    }
    const piv = aug.data[i * (n + k) + i];
    if (Math.abs(piv) < 1e-12) throw new Error('Singular matrix');
    for (let j = 0; j < n + k; j++) aug.data[i * (n + k) + j] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = aug.data[r * (n + k) + i];
      if (factor === 0) continue;
      for (let j = 0; j < n + k; j++) {
        aug.data[r * (n + k) + j] -= factor * aug.data[i * (n + k) + j];
      }
    }
  }
  if (b.ndim === 1) {
    const out = NDArray.zeros([n]);
    for (let i = 0; i < n; i++) out.data[i] = aug.data[i * (n + k) + n];
    return out;
  }
  const out = NDArray.zeros([n, k]);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) out.data[i * k + j] = aug.data[i * (n + k) + n + j];
  return out;
}

export class LinearRegression {
  constructor(opts = {}) {
    this.fitIntercept = opts.fitIntercept !== false;
    this.weights = null;
    this.intercept = 0;
  }
  fit(X, y) {
    let A = X instanceof NDArray ? X : NDArray.fromArray(X);
    let yv = y instanceof NDArray ? y : NDArray.fromArray(y);
    if (yv.ndim === 2) yv = yv.reshape([yv.shape[0]]);
    if (this.fitIntercept) {
      // Prepend a column of 1s for the intercept term.
      const M = A.shape[0], K = A.shape[1];
      const aug = NDArray.zeros([M, K + 1]);
      const ac = A.contiguous();
      for (let i = 0; i < M; i++) {
        aug.data[i * (K + 1)] = 1;
        for (let j = 0; j < K; j++) aug.data[i * (K + 1) + 1 + j] = ac.data[i * K + j];
      }
      A = aug;
    }
    const Xt = A.T;
    const XtX = O.matmul(Xt, A);
    const Xty = O.matmul(Xt, yv.reshape([yv.size, 1]));
    const w = solve(XtX, Xty.reshape([Xty.size]));
    if (this.fitIntercept) {
      this.intercept = w.data[0];
      this.weights = w.reshape([w.size]).data.slice(1);
    } else {
      this.intercept = 0;
      this.weights = w.reshape([w.size]).data.slice();
    }
    return this;
  }
  predict(X) {
    const A = X instanceof NDArray ? X : NDArray.fromArray(X);
    const w = NDArray.fromArray(Array.from(this.weights)).reshape([this.weights.length, 1]);
    const out = O.matmul(A, w).reshape([A.shape[0]]);
    return out.map((v) => v + this.intercept).toArray();
  }
}
