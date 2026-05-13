// src/nn/linear.js
import { Module, Parameter } from './module.js';
import { Tensor } from '../tensor.js';
import { NDArray } from '../ndarray.js';
import { defaultRng } from '../random.js';

export class Linear extends Module {
  /**
   * y = x @ W + b
   * @param {number} inFeatures
   * @param {number} outFeatures
   * @param {object} [opts]
   * @param {boolean} [opts.bias=true]
   * @param {import('../random.js').RNG} [opts.rng]
   */
  constructor(inFeatures, outFeatures, opts = {}) {
    super();
    const rng = opts.rng || defaultRng;
    const bias = opts.bias !== false;
    // Kaiming-uniform-ish: U(-1/sqrt(in), 1/sqrt(in))
    const k = 1 / Math.sqrt(inFeatures);
    const w = NDArray.zeros([inFeatures, outFeatures]);
    for (let i = 0; i < w.size; i++) w.data[i] = rng.uniform(-k, k);
    this.registerParameter('weight', new Parameter(w));
    if (bias) {
      const b = NDArray.zeros([outFeatures]);
      for (let i = 0; i < b.size; i++) b.data[i] = rng.uniform(-k, k);
      this.registerParameter('bias', new Parameter(b));
    } else {
      this.bias = null;
    }
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
  }

  forward(x) {
    let y = x.matmul(this.weight);
    if (this.bias) y = y.add(this.bias);
    return y;
  }
}

export class Sequential extends Module {
  constructor(...layers) {
    super();
    this.layers = layers;
    layers.forEach((l, i) => this.registerModule(`l${i}`, l));
  }
  forward(x) {
    let y = x;
    for (const l of this.layers) y = l.forward(y);
    return y;
  }
}
