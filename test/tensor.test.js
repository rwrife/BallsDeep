// test/tensor.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Tensor } from '../src/tensor.js';

const close = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

test('autograd: scalar add and mul', () => {
  const x = Tensor.of(3, true);
  const y = Tensor.of(4, true);
  const z = x.mul(y).add(x); // 3*4 + 3 = 15
  z.backward();
  assert.equal(z.toArray(), 15);
  // dz/dx = y + 1 = 5; dz/dy = x = 3
  assert.equal(x.grad.toArray(), 5);
  assert.equal(y.grad.toArray(), 3);
});

test('autograd: vector matmul gradient', () => {
  const W = Tensor.of([[1, 2], [3, 4]], true);
  const x = Tensor.of([[1], [1]], true);
  const y = W.matmul(x);            // [[3],[7]]
  const loss = y.sum();
  loss.backward();
  assert.deepEqual(y.toArray(), [[3], [7]]);
  // dL/dW = ones @ x.T = [[1,1],[1,1]]
  assert.deepEqual(W.grad.toArray(), [[1, 1], [1, 1]]);
  // dL/dx = W.T @ ones = [[4],[6]]
  assert.deepEqual(x.grad.toArray(), [[4], [6]]);
});

test('autograd: sigmoid backward matches analytic', () => {
  const x = Tensor.of([[0.5]], true);
  const y = x.sigmoid().sum();
  y.backward();
  const s = 1 / (1 + Math.exp(-0.5));
  assert.ok(close(x.grad.toArray()[0][0], s * (1 - s)));
});

test('autograd: gradient check via finite diff (mse)', () => {
  const w = Tensor.of([[0.5], [-0.3]], true);
  const x = Tensor.of([[1, 2], [3, 4], [5, 6]]);
  const y = Tensor.of([[1], [0], [1]]);
  const pred = x.matmul(w);
  const diff = pred.sub(y);
  const loss = diff.mul(diff).mean();
  loss.backward();
  const ana = w.grad.toArray();
  // numerical
  const h = 1e-5;
  const numerical = w.toArray().map((row, i) => row.map((v, j) => {
    const wPlus  = w.toArray();
    const wMinus = w.toArray();
    wPlus[i][j]  += h;
    wMinus[i][j] -= h;
    const f = (W) => {
      const wt = Tensor.of(W);
      const d = x.matmul(wt).sub(y);
      return d.mul(d).mean().toArray();
    };
    return (f(wPlus) - f(wMinus)) / (2 * h);
  }));
  for (let i = 0; i < ana.length; i++) {
    for (let j = 0; j < ana[0].length; j++) {
      assert.ok(close(ana[i][j], numerical[i][j], 1e-4),
        `grad[${i},${j}] ana=${ana[i][j]} num=${numerical[i][j]}`);
    }
  }
});

test('softmax sums to 1 along last axis', () => {
  const x = Tensor.of([[1, 2, 3], [1, 1, 1]]);
  const s = x.softmax();
  const arr = s.toArray();
  for (const row of arr) {
    const sum = row.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-9);
  }
});
