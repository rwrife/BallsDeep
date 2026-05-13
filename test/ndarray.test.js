// test/ndarray.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NDArray } from '../src/ndarray.js';
import * as O from '../src/ops.js';

test('fromArray + toArray round-trip', () => {
  const a = NDArray.fromArray([[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(a.shape, [2, 3]);
  assert.deepEqual(a.toArray(), [[1, 2, 3], [4, 5, 6]]);
});

test('reshape and transpose', () => {
  const a = NDArray.arange(0, 6).reshape([2, 3]);
  assert.deepEqual(a.toArray(), [[0, 1, 2], [3, 4, 5]]);
  assert.deepEqual(a.T.toArray(), [[0, 3], [1, 4], [2, 5]]);
});

test('broadcasting add', () => {
  const a = NDArray.fromArray([[1, 2, 3], [4, 5, 6]]);  // (2,3)
  const b = NDArray.fromArray([10, 20, 30]);            // (3,)
  assert.deepEqual(O.add(a, b).toArray(), [[11, 22, 33], [14, 25, 36]]);
});

test('broadcasting mul with column vector', () => {
  const a = NDArray.fromArray([[1, 2, 3], [4, 5, 6]]);     // (2,3)
  const c = NDArray.fromArray([[10], [100]]);              // (2,1)
  assert.deepEqual(O.mul(a, c).toArray(), [[10, 20, 30], [400, 500, 600]]);
});

test('matmul correctness', () => {
  const a = NDArray.fromArray([[1, 2], [3, 4]]);
  const b = NDArray.fromArray([[5, 6], [7, 8]]);
  assert.deepEqual(O.matmul(a, b).toArray(), [[19, 22], [43, 50]]);
});

test('reductions', () => {
  const a = NDArray.fromArray([[1, 2, 3], [4, 5, 6]]);
  assert.equal(O.sum(a).toArray(), 21);
  assert.deepEqual(O.sum(a, 0).toArray(), [5, 7, 9]);
  assert.deepEqual(O.sum(a, 1).toArray(), [6, 15]);
  assert.equal(O.mean(a).toArray(), 3.5);
  assert.equal(O.max(a).toArray(), 6);
});

test('sumTo reverses broadcasting', () => {
  const g = NDArray.ones([3, 4]);
  const r = O.sumTo(g, [4]);
  assert.deepEqual(r.shape, [4]);
  assert.deepEqual(r.toArray(), [3, 3, 3, 3]);
});

test('argmax', () => {
  const a = NDArray.fromArray([[1, 5, 3], [9, 2, 4]]);
  assert.deepEqual(O.argmax(a, 1), [1, 0]);
});
