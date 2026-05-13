// src/data/scaler.js
import { NDArray } from '../ndarray.js';
import * as O from '../ops.js';

/** Standardize features to zero mean / unit variance per column. */
export class StandardScaler {
  constructor() { this.mean = null; this.std = null; }
  fit(X) {
    const a = X instanceof NDArray ? X : NDArray.fromArray(X);
    this.mean = O.mean(a, 0, true);
    const diff = O.sub(a, this.mean);
    const variance = O.mean(O.mul(diff, diff), 0, true);
    this.std = variance.map((v) => Math.sqrt(v) || 1);
    return this;
  }
  transform(X) {
    const a = X instanceof NDArray ? X : NDArray.fromArray(X);
    return O.div(O.sub(a, this.mean), this.std);
  }
  fitTransform(X) { return this.fit(X).transform(X); }
  inverseTransform(X) {
    const a = X instanceof NDArray ? X : NDArray.fromArray(X);
    return O.add(O.mul(a, this.std), this.mean);
  }
}

/** Min-max scale features to [0, 1] per column. */
export class MinMaxScaler {
  constructor() { this.min = null; this.range = null; }
  fit(X) {
    const a = X instanceof NDArray ? X : NDArray.fromArray(X);
    this.min = O.min(a, 0, true);
    const max = O.max(a, 0, true);
    this.range = O.sub(max, this.min).map((v) => v || 1);
    return this;
  }
  transform(X) {
    const a = X instanceof NDArray ? X : NDArray.fromArray(X);
    return O.div(O.sub(a, this.min), this.range);
  }
  fitTransform(X) { return this.fit(X).transform(X); }
  inverseTransform(X) {
    const a = X instanceof NDArray ? X : NDArray.fromArray(X);
    return O.add(O.mul(a, this.range), this.min);
  }
}
