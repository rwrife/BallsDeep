// src/ops.js
// NDArray operations. Pure functions returning new arrays unless suffixed _.
import { NDArray, broadcastShapes } from './ndarray.js';

const asArray = (x) => (x instanceof NDArray ? x : NDArray.fromArray(x));

const broadcast = (a, b, fn) => NDArray.broadcastMap(asArray(a), asArray(b), fn);

export const add = (a, b) => broadcast(a, b, (x, y) => x + y);
export const sub = (a, b) => broadcast(a, b, (x, y) => x - y);
export const mul = (a, b) => broadcast(a, b, (x, y) => x * y);
export const div = (a, b) => broadcast(a, b, (x, y) => x / y);
export const pow = (a, b) => broadcast(a, b, (x, y) => Math.pow(x, y));
export const maximum = (a, b) => broadcast(a, b, (x, y) => Math.max(x, y));
export const minimum = (a, b) => broadcast(a, b, (x, y) => Math.min(x, y));

export const neg = (a) => asArray(a).map((x) => -x);
export const exp = (a) => asArray(a).map((x) => Math.exp(x));
export const log = (a) => asArray(a).map((x) => Math.log(x));
export const sqrt = (a) => asArray(a).map((x) => Math.sqrt(x));
export const abs = (a) => asArray(a).map((x) => Math.abs(x));

export const sigmoid = (a) => asArray(a).map((x) => 1 / (1 + Math.exp(-x)));
export const tanh = (a) => asArray(a).map((x) => Math.tanh(x));
export const relu = (a) => asArray(a).map((x) => (x > 0 ? x : 0));

/** Reduce over `axis`. If axis is null, reduce over all elements. */
function reduce(a, axis, init, fn, keepdims = false) {
  const arr = asArray(a).contiguous();
  if (axis == null) {
    let acc = init;
    for (let i = 0; i < arr.size; i++) acc = fn(acc, arr.data[i]);
    if (keepdims) {
      const shape = new Array(arr.ndim).fill(1);
      const out = NDArray.zeros(shape);
      out.data[0] = acc;
      return out;
    }
    const out = new NDArray(new Float64Array([acc]), [1]);
    return out.reshape([]); // 0-d
  }
  if (axis < 0) axis += arr.ndim;
  const outShape = arr.shape.slice();
  outShape[axis] = 1;
  const out = NDArray.full(outShape, init);
  // walk source
  const idx = new Array(arr.ndim).fill(0);
  for (let i = 0; i < arr.size; i++) {
    // compute source flat index (contiguous)
    // and dest flat index from idx with axis fixed at 0
    let src = 0, dst = 0;
    for (let d = 0; d < arr.ndim; d++) {
      src += idx[d] * arr.strides[d];
      const di = d === axis ? 0 : idx[d];
      dst += di * out.strides[d];
    }
    out.data[dst] = fn(out.data[dst], arr.data[src]);
    for (let d = arr.ndim - 1; d >= 0; d--) {
      if (++idx[d] < arr.shape[d]) break;
      idx[d] = 0;
    }
  }
  if (keepdims) return out;
  // squeeze axis
  const sq = outShape.slice();
  sq.splice(axis, 1);
  return out.reshape(sq.length === 0 ? [] : sq);
}

export const sum = (a, axis = null, keepdims = false) =>
  reduce(a, axis, 0, (acc, v) => acc + v, keepdims);
export const mean = (a, axis = null, keepdims = false) => {
  const s = sum(a, axis, keepdims);
  const arr = asArray(a);
  const denom = axis == null ? arr.size : arr.shape[axis < 0 ? axis + arr.ndim : axis];
  return s.map((x) => x / denom);
};
export const max = (a, axis = null, keepdims = false) =>
  reduce(a, axis, -Infinity, (acc, v) => (v > acc ? v : acc), keepdims);
export const min = (a, axis = null, keepdims = false) =>
  reduce(a, axis, Infinity, (acc, v) => (v < acc ? v : acc), keepdims);

/** 2D matrix multiply. Supports (M,K) @ (K,N) -> (M,N). */
export function matmul(a, b) {
  const A = asArray(a).contiguous();
  const B = asArray(b).contiguous();
  if (A.ndim !== 2 || B.ndim !== 2) {
    throw new Error(`matmul expects 2D arrays, got ${A.ndim}D and ${B.ndim}D`);
  }
  const [M, K] = A.shape;
  const [K2, N] = B.shape;
  if (K !== K2) throw new Error(`matmul shape mismatch: [${A.shape}] x [${B.shape}]`);
  const out = NDArray.zeros([M, N]);
  const ad = A.data, bd = B.data, od = out.data;
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const aik = ad[i * K + k];
      if (aik === 0) continue;
      for (let j = 0; j < N; j++) {
        od[i * N + j] += aik * bd[k * N + j];
      }
    }
  }
  return out;
}

/** Argmax along an axis (returns plain array of ints). */
export function argmax(a, axis = -1) {
  const arr = asArray(a).contiguous();
  const ax = axis < 0 ? axis + arr.ndim : axis;
  const outShape = arr.shape.slice();
  outShape.splice(ax, 1);
  const out = new Array(outShape.reduce((x, y) => x * y, 1)).fill(0);
  const idx = new Array(arr.ndim).fill(0);
  const best = new Array(out.length).fill(-Infinity);
  for (let i = 0; i < arr.size; i++) {
    let src = 0;
    for (let d = 0; d < arr.ndim; d++) src += idx[d] * arr.strides[d];
    // dest index
    let dst = 0, mult = 1;
    for (let d = arr.ndim - 1; d >= 0; d--) {
      if (d === ax) continue;
      dst += idx[d] * mult;
      mult *= arr.shape[d];
    }
    if (arr.data[src] > best[dst]) {
      best[dst] = arr.data[src];
      out[dst] = idx[ax];
    }
    for (let d = arr.ndim - 1; d >= 0; d--) {
      if (++idx[d] < arr.shape[d]) break;
      idx[d] = 0;
    }
  }
  return out;
}

/** Concatenate along axis 0 only (sufficient for batching). */
export function concat0(arrays) {
  const arrs = arrays.map(asArray).map((a) => a.contiguous());
  const baseShape = arrs[0].shape.slice(1);
  let total = 0;
  for (const a of arrs) {
    total += a.shape[0];
    for (let i = 1; i < a.ndim; i++) {
      if (a.shape[i] !== baseShape[i - 1]) {
        throw new Error('concat0: incompatible trailing shapes');
      }
    }
  }
  const out = NDArray.zeros([total, ...baseShape]);
  let off = 0;
  for (const a of arrs) {
    out.data.set(a.data, off);
    off += a.size;
  }
  return out;
}

/** Sum-to shape: collapse via summation so result has `targetShape`.
 *  Used by autograd to reverse broadcasting. */
export function sumTo(a, targetShape) {
  const arr = asArray(a);
  let cur = arr;
  // First, collapse leading extra dims by summing
  const extra = cur.ndim - targetShape.length;
  for (let i = 0; i < extra; i++) {
    cur = sum(cur, 0);
  }
  // Now collapse axes where target is 1 but cur is >1
  for (let d = 0; d < targetShape.length; d++) {
    if (targetShape[d] === 1 && cur.shape[d] !== 1) {
      cur = sum(cur, d, true);
    }
  }
  return cur.reshape(targetShape);
}

export { broadcastShapes };
