// examples/mlp.js — train MLPClassifier on XOR and MLPRegressor on a noisy sine.
import { models, RNG } from '../src/index.js';

console.log('--- MLPClassifier on XOR ---');
const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [0, 1, 1, 0];
const clf = new models.MLPClassifier({
  hiddenLayers: [8, 8],
  activation: 'relu',
  optimizer: 'adam',
  lr: 0.05,
  epochs: 800,
  rng: new RNG(0),
}).fit(X, y);
console.log('preds:', clf.predict(X), 'truth:', y);
console.log('probs:', clf.predictProba(X).map((r) => r.map((v) => +v.toFixed(3))));

console.log('\n--- MLPRegressor on noisy sine ---');
const rng = new RNG(1);
const Xr = [], yr = [];
for (let i = 0; i < 200; i++) {
  const x = -3 + (6 * i) / 199;
  Xr.push([x]);
  yr.push(Math.sin(x) + 0.05 * rng.normal());
}
const reg = new models.MLPRegressor({
  hiddenLayers: [32, 32],
  activation: 'tanh',
  optimizer: 'adam',
  lr: 0.01,
  epochs: 600,
  batchSize: 32,
  rng,
}).fit(Xr, yr);
const sample = [[-2], [-1], [0], [1], [2]];
console.log('x   ->  pred       truth');
for (const [i, [x]] of sample.entries()) {
  const p = reg.predict([sample[i]])[0];
  console.log(`${x.toFixed(2).padStart(5)} -> ${p.toFixed(4)}    ${Math.sin(x).toFixed(4)}`);
}
console.log(`final loss: ${reg.lossHistory[reg.lossHistory.length - 1].toFixed(6)}`);
