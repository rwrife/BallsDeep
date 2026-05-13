// src/optim/sgd.js
import * as O from '../ops.js';

export class SGD {
  /**
   * @param {import('../nn/module.js').Parameter[]} params
   * @param {object} opts
   * @param {number} [opts.lr=0.01]
   * @param {number} [opts.momentum=0]
   * @param {number} [opts.weightDecay=0]
   */
  constructor(params, opts = {}) {
    this.params = params;
    this.lr = opts.lr ?? 0.01;
    this.momentum = opts.momentum ?? 0;
    this.weightDecay = opts.weightDecay ?? 0;
    this._velocity = params.map(() => null);
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  step() {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      let g = p.grad;
      if (this.weightDecay) g = O.add(g, O.mul(p.data, this.weightDecay));
      if (this.momentum) {
        const v = this._velocity[i] == null
          ? g.copy()
          : O.add(O.mul(this._velocity[i], this.momentum), g);
        this._velocity[i] = v;
        p.data = O.sub(p.data, O.mul(v, this.lr));
      } else {
        p.data = O.sub(p.data, O.mul(g, this.lr));
      }
    }
  }
}
