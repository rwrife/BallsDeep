// src/nn/module.js
import { Tensor } from '../tensor.js';

export class Parameter extends Tensor {
  constructor(data) {
    super(data, { requiresGrad: true });
  }
}

export class Module {
  constructor() {
    this._modules = new Map();
    this._params = new Map();
    this.training = true;
  }

  /** Register a sub-module by name. */
  registerModule(name, m) {
    this._modules.set(name, m);
    this[name] = m;
    return m;
  }

  /** Register a Parameter by name. */
  registerParameter(name, p) {
    this._params.set(name, p);
    this[name] = p;
    return p;
  }

  parameters() {
    const out = [];
    for (const p of this._params.values()) out.push(p);
    for (const m of this._modules.values()) out.push(...m.parameters());
    return out;
  }

  zeroGrad() {
    for (const p of this.parameters()) p.zeroGrad();
  }

  train(flag = true) {
    this.training = flag;
    for (const m of this._modules.values()) m.train(flag);
    return this;
  }
  eval() { return this.train(false); }

  forward(/* x */) { throw new Error('Module.forward must be implemented'); }

  /** Convenience callable shorthand. */
  call(x) { return this.forward(x); }
}
