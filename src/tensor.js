// src/tensor.js
// PyTorch-style dynamic autograd. Each op records a backward closure
// that knows how to push a gradient back to its inputs.
import { NDArray } from './ndarray.js';
import * as O from './ops.js';

let _idCounter = 0;

export class Tensor {
  /**
   * @param {NDArray|number[]|number} data
   * @param {object} [opts]
   * @param {boolean} [opts.requiresGrad]
   * @param {Tensor[]} [opts.parents]
   * @param {Function} [opts._backward]
   */
  constructor(data, opts = {}) {
    this.data = data instanceof NDArray ? data : NDArray.fromArray(data);
    this.requiresGrad = !!opts.requiresGrad;
    this.grad = null; // NDArray once .backward() runs
    this._parents = opts.parents || [];
    this._backward = opts._backward || null;
    this._id = ++_idCounter;
  }

  static of(data, requiresGrad = false) {
    return new Tensor(data, { requiresGrad });
  }

  get shape() { return this.data.shape; }
  get ndim()  { return this.data.ndim; }
  get size()  { return this.data.size; }

  toArray() { return this.data.toArray(); }

  zeroGrad() { this.grad = null; }

  /** Topological backward pass starting from this scalar (or ones-shaped). */
  backward(seed) {
    // Build a topo order
    const topo = [];
    const visited = new Set();
    const visit = (t) => {
      if (visited.has(t._id)) return;
      visited.add(t._id);
      for (const p of t._parents) visit(p);
      topo.push(t);
    };
    visit(this);

    // Seed gradient
    if (seed === undefined) {
      if (this.size !== 1) {
        throw new Error('backward() on non-scalar Tensor requires explicit grad');
      }
      this.grad = NDArray.full(this.shape.length === 0 ? [] : this.shape, 1);
    } else {
      this.grad = seed instanceof NDArray ? seed : NDArray.fromArray(seed);
    }

    // Walk in reverse topo order
    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t._backward && t.grad) t._backward(t.grad);
    }
  }

  // ---------- helpers ----------
  static _accGrad(t, g) {
    if (!t.requiresGrad) return;
    // Reverse-broadcast g back to t.shape
    const reshaped = O.sumTo(g, t.shape.length === 0 ? [] : t.shape);
    if (!t.grad) t.grad = reshaped.copy();
    else t.grad = O.add(t.grad, reshaped);
  }

  // ---------- ops ----------
  add(other) {
    const b = _asTensor(other);
    const out = new Tensor(O.add(this.data, b.data), {
      requiresGrad: this.requiresGrad || b.requiresGrad,
      parents: [this, b],
    });
    out._backward = (g) => {
      Tensor._accGrad(this, g);
      Tensor._accGrad(b, g);
    };
    return out;
  }
  sub(other) {
    const b = _asTensor(other);
    const out = new Tensor(O.sub(this.data, b.data), {
      requiresGrad: this.requiresGrad || b.requiresGrad,
      parents: [this, b],
    });
    out._backward = (g) => {
      Tensor._accGrad(this, g);
      Tensor._accGrad(b, O.neg(g));
    };
    return out;
  }
  mul(other) {
    const b = _asTensor(other);
    const out = new Tensor(O.mul(this.data, b.data), {
      requiresGrad: this.requiresGrad || b.requiresGrad,
      parents: [this, b],
    });
    out._backward = (g) => {
      Tensor._accGrad(this, O.mul(g, b.data));
      Tensor._accGrad(b, O.mul(g, this.data));
    };
    return out;
  }
  div(other) {
    const b = _asTensor(other);
    const out = new Tensor(O.div(this.data, b.data), {
      requiresGrad: this.requiresGrad || b.requiresGrad,
      parents: [this, b],
    });
    out._backward = (g) => {
      Tensor._accGrad(this, O.div(g, b.data));
      // d/db (a/b) = -a/b^2
      const num = O.neg(O.mul(g, this.data));
      const den = O.mul(b.data, b.data);
      Tensor._accGrad(b, O.div(num, den));
    };
    return out;
  }
  neg() {
    const out = new Tensor(O.neg(this.data), {
      requiresGrad: this.requiresGrad,
      parents: [this],
    });
    out._backward = (g) => Tensor._accGrad(this, O.neg(g));
    return out;
  }
  pow(p) {
    // p as plain scalar for simplicity.
    const out = new Tensor(this.data.map((x) => Math.pow(x, p)), {
      requiresGrad: this.requiresGrad,
      parents: [this],
    });
    out._backward = (g) => {
      const local = this.data.map((x) => p * Math.pow(x, p - 1));
      Tensor._accGrad(this, O.mul(g, local));
    };
    return out;
  }
  matmul(other) {
    const b = _asTensor(other);
    const out = new Tensor(O.matmul(this.data, b.data), {
      requiresGrad: this.requiresGrad || b.requiresGrad,
      parents: [this, b],
    });
    out._backward = (g) => {
      // g is (M,N); a is (M,K); b is (K,N)
      Tensor._accGrad(this, O.matmul(g, b.data.T));
      Tensor._accGrad(b, O.matmul(this.data.T, g));
    };
    return out;
  }
  sum(axis = null, keepdims = false) {
    const out = new Tensor(O.sum(this.data, axis, keepdims), {
      requiresGrad: this.requiresGrad,
      parents: [this],
    });
    out._backward = (g) => {
      // broadcast g back to original shape
      let gExp = g;
      if (axis != null && !keepdims) {
        const sh = this.shape.slice();
        sh[axis < 0 ? axis + this.ndim : axis] = 1;
        gExp = g.reshape(sh);
      }
      // multiply by ones of original shape (broadcast-add handles it)
      Tensor._accGrad(this, O.add(NDArray.zeros(this.shape), gExp));
    };
    return out;
  }
  mean(axis = null, keepdims = false) {
    const denom = axis == null
      ? this.size
      : this.shape[axis < 0 ? axis + this.ndim : axis];
    return this.sum(axis, keepdims).mul(1 / denom);
  }
  exp() {
    const d = O.exp(this.data);
    const out = new Tensor(d, { requiresGrad: this.requiresGrad, parents: [this] });
    out._backward = (g) => Tensor._accGrad(this, O.mul(g, d));
    return out;
  }
  log() {
    const out = new Tensor(O.log(this.data), {
      requiresGrad: this.requiresGrad, parents: [this],
    });
    out._backward = (g) => Tensor._accGrad(this, O.div(g, this.data));
    return out;
  }
  relu() {
    const mask = this.data.map((x) => (x > 0 ? 1 : 0));
    const out = new Tensor(O.mul(this.data, mask), {
      requiresGrad: this.requiresGrad, parents: [this],
    });
    out._backward = (g) => Tensor._accGrad(this, O.mul(g, mask));
    return out;
  }
  sigmoid() {
    const s = O.sigmoid(this.data);
    const out = new Tensor(s, { requiresGrad: this.requiresGrad, parents: [this] });
    out._backward = (g) => {
      const local = O.mul(s, s.map((x) => 1 - x));
      Tensor._accGrad(this, O.mul(g, local));
    };
    return out;
  }
  tanh() {
    const t = O.tanh(this.data);
    const out = new Tensor(t, { requiresGrad: this.requiresGrad, parents: [this] });
    out._backward = (g) => {
      const local = t.map((x) => 1 - x * x);
      Tensor._accGrad(this, O.mul(g, local));
    };
    return out;
  }
  reshape(shape) {
    const out = new Tensor(this.data.reshape(shape), {
      requiresGrad: this.requiresGrad, parents: [this],
    });
    out._backward = (g) => Tensor._accGrad(this, g.reshape(this.shape));
    return out;
  }
  transpose(axes) {
    const out = new Tensor(this.data.transpose(axes), {
      requiresGrad: this.requiresGrad, parents: [this],
    });
    // For 2D: backward transpose is the inverse permutation; for 2D it's the
    // same swap so we just transpose g back.
    out._backward = (g) => Tensor._accGrad(this, g.transpose(axes));
    return out;
  }
  get T() { return this.transpose(); }

  /** softmax along last axis */
  softmax() {
    const shifted = this.sub(_asTensor(O.max(this.data, this.ndim - 1, true)));
    const e = shifted.exp();
    const denom = e.sum(this.ndim - 1, true);
    return e.div(denom);
  }

  /** log-softmax along last axis (numerically stable) */
  logSoftmax() {
    const m = _asTensor(O.max(this.data, this.ndim - 1, true));
    const shifted = this.sub(m);
    const lse = shifted.exp().sum(this.ndim - 1, true).log();
    return shifted.sub(lse);
  }
}

function _asTensor(x) {
  if (x instanceof Tensor) return x;
  if (typeof x === 'number') return new Tensor(NDArray.full([], x));
  return new Tensor(x);
}

export const tensor = (data, requiresGrad = false) => Tensor.of(data, requiresGrad);
