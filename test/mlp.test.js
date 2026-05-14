// test/mlp.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MLPClassifier, MLPRegressor } from '../src/models/index.js';
import { RNG } from '../src/random.js';

test('MLPClassifier solves XOR', () => {
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const y = [0, 1, 1, 0];
  const m = new MLPClassifier({
    hiddenLayers: [8, 8],
    activation: 'relu',
    lr: 0.05,
    epochs: 800,
    rng: new RNG(0),
  }).fit(X, y);
  assert.deepEqual(m.predict(X), y);
});

test('MLPClassifier handles 3-class linearly-separable data', () => {
  const X = [], y = [];
  const centers = [[0, 0], [5, 5], [0, 5]];
  const rng = new RNG(7);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < 30; i++) {
      X.push([centers[c][0] + rng.normal() * 0.3, centers[c][1] + rng.normal() * 0.3]);
      y.push(c);
    }
  }
  const m = new MLPClassifier({
    hiddenLayers: [16],
    activation: 'tanh',
    lr: 0.05,
    epochs: 300,
    rng,
  }).fit(X, y);
  const preds = m.predict(X);
  let correct = 0;
  for (let i = 0; i < preds.length; i++) if (preds[i] === y[i]) correct++;
  assert.ok(correct / preds.length > 0.95, `acc=${correct / preds.length}`);
  // predictProba rows sum to 1.
  const probs = m.predictProba(X.slice(0, 3));
  for (const row of probs) {
    const s = row.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(s - 1) < 1e-6);
  }
});

test('MLPRegressor fits y = sin(x) approximately', () => {
  const rng = new RNG(2);
  const X = [], y = [];
  for (let i = 0; i < 200; i++) {
    const x = -3 + (6 * i) / 199;
    X.push([x]);
    y.push(Math.sin(x));
  }
  const m = new MLPRegressor({
    hiddenLayers: [32, 32],
    activation: 'tanh',
    lr: 0.01,
    epochs: 400,
    batchSize: 32,
    rng,
  }).fit(X, y);
  const preds = m.predict(X);
  let mse = 0;
  for (let i = 0; i < preds.length; i++) mse += (preds[i] - y[i]) ** 2;
  mse /= preds.length;
  assert.ok(mse < 0.01, `mse=${mse}`);
});
