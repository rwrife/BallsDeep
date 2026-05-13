// src/ndarray.js
// Minimal numpy-style n-dimensional array on top of Float64Array.
// Strided, supports broadcasting, reshape, transpose, slicing.

const TYPED = Float64Array;

function computeStrides(shape) {
  const n = shape.length;
  const strides = new Array(n);
  let s = 1;
  for (let i = n - 1; i >= 0; i--) {
    strides[i] = s;
    s *= shape[i];
  }
  return strides;
}

function shapeSize(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function shapesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * Broadcast two shapes per NumPy rules. Returns the resulting shape
 * or throws if incompatible.
 */
export function broadcastShapes(a, b) {
  const out = [];
  const la = a.length, lb = b.length;
  const n = Math.max(la, lb);
  for (let i = 0; i < n; i++) {
    const da = i < n - la ? 1 : a[i - (n - la)];
    const db = i < n - lb ? 1 : b[i - (n - lb)];
    if (da === db || da === 1 || db === 1) {
      out.push(Math.max(da, db));
    } else {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
  }
  return out;
}

export class NDArray {
  /**
   * @param {Float64Array|number[]} data flat storage
   * @param {number[]} shape
   * @param {number[]} [strides]
   * @param {number} [offset]
   */
  constructor(data, shape, strides, offset = 0) {
    this.data = data instanceof TYPED ? data : TYPED.from(data);
    this.shape = shape.slice();
    this.strides = strides ? strides.slice() : computeStrides(shape);
    this.offset = offset;
    this.size = shapeSize(shape);
    this.ndim = shape.length;
  }

  // ---------- factories ----------
  static zeros(shape) {
    return new NDArray(new TYPED(shapeSize(shape)), shape);
  }
  static ones(shape) {
    const a = NDArray.zeros(shape);
    a.data.fill(1);
    return a;
  }
  static full(shape, v) {
    const a = NDArray.zeros(shape);
    a.data.fill(v);
    return a;
  }
  static fromArray(nested) {
    const shape = [];
    let cur = nested;
    while (Array.isArray(cur)) {
      shape.push(cur.length);
      cur = cur[0];
    }
    const flat = new TYPED(shapeSize(shape));
    let i = 0;
    const walk = (x, depth) => {
      if (depth === shape.length) {
        flat[i++] = x;
        return;
      }
      for (const item of x) walk(item, depth + 1);
    };
    walk(nested, 0);
    return new NDArray(flat, shape);
  }
  static arange(start, stop, step = 1) {
    if (stop === undefined) { stop = start; start = 0; }
    const n = Math.max(0, Math.ceil((stop - start) / step));
    const data = new TYPED(n);
    for (let i = 0; i < n; i++) data[i] = start + i * step;
    return new NDArray(data, [n]);
  }
  static eye(n) {
    const a = NDArray.zeros([n, n]);
    for (let i = 0; i < n; i++) a.data[i * n + i] = 1;
    return a;
  }

  // ---------- views ----------
  isContiguous() {
    const exp = computeStrides(this.shape);
    for (let i = 0; i < this.ndim; i++) if (this.strides[i] !== exp[i]) return false;
    return this.offset === 0;
  }

  /** Flat index of multi-index. */
  _flatIndex(idx) {
    let f = this.offset;
    for (let i = 0; i < idx.length; i++) f += idx[i] * this.strides[i];
    return f;
  }

  get(...idx) { return this.data[this._flatIndex(idx)]; }
  set(...args) {
    const v = args.pop();
    this.data[this._flatIndex(args)] = v;
    return this;
  }

  reshape(newShape) {
    // Allow one -1 inference.
    let infer = -1, prod = 1;
    for (let i = 0; i < newShape.length; i++) {
      if (newShape[i] === -1) {
        if (infer !== -1) throw new Error('Only one -1 allowed in reshape');
        infer = i;
      } else prod *= newShape[i];
    }
    const ns = newShape.slice();
    if (infer !== -1) ns[infer] = this.size / prod;
    if (shapeSize(ns) !== this.size) {
      throw new Error(`Cannot reshape size ${this.size} into [${ns}]`);
    }
    const src = this.contiguous();
    return new NDArray(src.data, ns);
  }

  transpose(axes) {
    const n = this.ndim;
    const ax = axes ? axes.slice() : Array.from({ length: n }, (_, i) => n - 1 - i);
    const newShape = ax.map(i => this.shape[i]);
    const newStrides = ax.map(i => this.strides[i]);
    return new NDArray(this.data, newShape, newStrides, this.offset);
  }

  get T() { return this.transpose(); }

  /** Copy into a fresh contiguous array. */
  contiguous() {
    if (this.isContiguous()) return this;
    const out = NDArray.zeros(this.shape);
    const idx = new Array(this.ndim).fill(0);
    for (let i = 0; i < this.size; i++) {
      out.data[i] = this.data[this._flatIndex(idx)];
      // increment multi-index
      for (let d = this.ndim - 1; d >= 0; d--) {
        if (++idx[d] < this.shape[d]) break;
        idx[d] = 0;
      }
    }
    return out;
  }

  copy() {
    const c = this.contiguous();
    return new NDArray(new TYPED(c.data), c.shape);
  }

  toArray() {
    const c = this.contiguous();
    const build = (depth, off) => {
      if (depth === this.ndim) return c.data[off];
      const out = [];
      const stride = c.strides[depth];
      for (let i = 0; i < this.shape[depth]; i++) {
        out.push(build(depth + 1, off + i * stride));
      }
      return out;
    };
    if (this.ndim === 0) return c.data[0];
    return build(0, 0);
  }

  // ---------- broadcasting iteration ----------
  /**
   * Apply a binary fn over two arrays with broadcasting.
   * @returns {NDArray}
   */
  static broadcastMap(a, b, fn) {
    const ashape = a.shape, bshape = b.shape;
    const outShape = broadcastShapes(ashape, bshape);
    const out = NDArray.zeros(outShape);
    const n = outShape.length;

    // Pad strides with leading zeros for missing dims; set stride to 0
    // wherever the original dim is 1 (broadcasting).
    const padStrides = (arr) => {
      const pad = n - arr.ndim;
      const s = new Array(n);
      for (let i = 0; i < n; i++) {
        if (i < pad) s[i] = 0;
        else {
          const dim = arr.shape[i - pad];
          s[i] = dim === 1 ? 0 : arr.strides[i - pad];
        }
      }
      return s;
    };
    const sa = padStrides(a), sb = padStrides(b);

    const idx = new Array(n).fill(0);
    let oa = a.offset, ob = b.offset, oo = 0;
    const total = out.size;
    for (let k = 0; k < total; k++) {
      out.data[oo] = fn(a.data[oa], b.data[ob]);
      // increment multi-index from rightmost
      for (let d = n - 1; d >= 0; d--) {
        if (++idx[d] < outShape[d]) {
          oa += sa[d]; ob += sb[d]; oo += 1;
          break;
        }
        idx[d] = 0;
        oa -= sa[d] * (outShape[d] - 1);
        ob -= sb[d] * (outShape[d] - 1);
        // oo continues incrementing
      }
    }
    return out;
  }

  /** Element-wise unary map (preserves shape, returns contiguous). */
  map(fn) {
    const c = this.contiguous();
    const out = new TYPED(c.size);
    for (let i = 0; i < c.size; i++) out[i] = fn(c.data[i], i);
    return new NDArray(out, c.shape);
  }
}
