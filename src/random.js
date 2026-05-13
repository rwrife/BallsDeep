// src/random.js
// Seedable RNG (mulberry32) + sampling helpers.
import { NDArray } from './ndarray.js';

export class RNG {
  constructor(seed = 0xC0FFEE) {
    this._s = seed >>> 0;
  }
  /** uniform in [0, 1) */
  next() {
    let t = (this._s += 0x6D2B79F5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(lo = 0, hi = 1) { return lo + (hi - lo) * this.next(); }
  /** Box–Muller standard normal */
  normal(mean = 0, std = 1) {
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mean + std * z;
  }
  bernoulli(p = 0.5) { return this.next() < p ? 1 : 0; }

  randn(shape) {
    const a = NDArray.zeros(shape);
    for (let i = 0; i < a.size; i++) a.data[i] = this.normal();
    return a;
  }
  rand(shape) {
    const a = NDArray.zeros(shape);
    for (let i = 0; i < a.size; i++) a.data[i] = this.next();
    return a;
  }
  /** Random integers in [lo, hi). */
  randint(lo, hi, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = lo + Math.floor(this.next() * (hi - lo));
    return out;
  }
  /** In-place Fisher–Yates shuffle. */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

/** Module-level default RNG so simple scripts work without setup. */
export const defaultRng = new RNG(42);
