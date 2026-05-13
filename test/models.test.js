// test/models.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LinearRegression, LogisticRegression, KNeighborsClassifier }
  from '../src/models/index.js';

test('LinearRegression recovers slope/intercept', () => {
  // Two truly-independent feature columns.
  const X = Array.from({ length: 30 }, (_, i) => [i, (i * 7) % 11]);
  const y = X.map(([a, b]) => 2 * a - 3 * b + 5);
  const m = new LinearRegression().fit(X, y);
  assert.ok(Math.abs(m.weights[0] - 2) < 1e-6);
  assert.ok(Math.abs(m.weights[1] - (-3)) < 1e-6);
  assert.ok(Math.abs(m.intercept - 5) < 1e-6);
});

test('LogisticRegression separates a linearly-separable cloud', () => {
  // Class 1 if x0 + x1 > 1
  const pts = [];
  const labels = [];
  for (let i = 0; i < 60; i++) {
    const a = (i % 8) / 4, b = (i % 5) / 4;
    pts.push([a, b]);
    labels.push(a + b > 1 ? 1 : 0);
  }
  const m = new LogisticRegression({ lr: 0.3, epochs: 400 }).fit(pts, labels);
  const preds = m.predict(pts);
  let correct = 0;
  for (let i = 0; i < preds.length; i++) if (preds[i] === labels[i]) correct++;
  assert.ok(correct / preds.length > 0.9, `acc=${correct / preds.length}`);
});

test('KNN classifier handles toy 2-cluster set', () => {
  const X = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]];
  const y = [0, 0, 0, 1, 1, 1];
  const m = new KNeighborsClassifier({ k: 3 }).fit(X, y);
  assert.deepEqual(m.predict([[0.5, 0.5], [10.5, 10.5]]), [0, 1]);
});
