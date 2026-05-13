// examples/linear-regression.js — closed-form OLS demo.
import { models } from '../src/index.js';

// Synthetic dataset: y = 1.5*a - 2*b + 4 + noise
const X = [], y = [];
for (let i = 0; i < 100; i++) {
  const a = i / 10;
  const b = (i % 7) - 3;
  X.push([a, b]);
  y.push(1.5 * a - 2 * b + 4 + (Math.random() - 0.5) * 0.05);
}

const m = new models.LinearRegression().fit(X, y);
console.log('weights:   ', Array.from(m.weights).map(v => v.toFixed(4)));
console.log('intercept: ', m.intercept.toFixed(4));
console.log('truth:     [1.5, -2]  intercept 4');

const sample = [[5, 1], [10, -2]];
console.log('\npredictions:');
for (const [i, x] of sample.entries()) {
  console.log(`  x=${x.join(',')}  ->  y_hat=${m.predict([x])[0].toFixed(4)}`);
}
