// src/optim/adam.js
import * as O from '../ops.js';
import { NDArray } from '../ndarray.js';

export class Adam {
  constructor(params, opts = {}) {
    this.params = params;
    this.lr    = opts.lr    ?? 1e-3;
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.eps   = opts.eps   ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0;
    this._m = params.map((p) => NDArray.zeros(p.shape));
    this._v = params.map((p) => NDArray.zeros(p.shape));
    this._t = 0;
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  step() {
    this._t += 1;
    const t = this._t;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      let g = p.grad;
      if (this.weightDecay) g = O.add(g, O.mul(p.data, this.weightDecay));

      this._m[i] = O.add(O.mul(this._m[i], this.beta1), O.mul(g, 1 - this.beta1));
      this._v[i] = O.add(O.mul(this._v[i], this.beta2), O.mul(O.mul(g, g), 1 - this.beta2));

      const mHat = O.div(this._m[i], 1 - Math.pow(this.beta1, t));
      const vHat = O.div(this._v[i], 1 - Math.pow(this.beta2, t));

      const update = O.div(mHat, O.add(O.sqrt(vHat), this.eps));
      p.data = O.sub(p.data, O.mul(update, this.lr));
    }
  }
}
