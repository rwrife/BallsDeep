// test/nn.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Tensor } from '../src/tensor.js';
import { Linear, Sequential, ReLU, Sigmoid } from '../src/nn/index.js';
import { mseLoss, bceLoss } from '../src/nn/index.js';
import { SGD, Adam } from '../src/optim/index.js';
import { RNG } from '../src/random.js';

test('XOR learnable with small MLP + Adam', () => {
  const rng = new RNG(7);
  const model = new Sequential(
    new Linear(2, 8, { rng }),
    new ReLU(),
    new Linear(8, 1, { rng }),
    new Sigmoid(),
  );
  const X = Tensor.of([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const y = Tensor.of([[0], [1], [1], [0]]);
  const opt = new Adam(model.parameters(), { lr: 0.1 });
  for (let i = 0; i < 1500; i++) {
    opt.zeroGrad();
    const out = model.forward(X);
    const loss = bceLoss(out, y);
    loss.backward();
    opt.step();
  }
  const preds = model.forward(X).toArray().map(([p]) => (p >= 0.5 ? 1 : 0));
  assert.deepEqual(preds, [0, 1, 1, 0]);
});

test('linear fit y = 2x + 1 with SGD', () => {
  const rng = new RNG(1);
  const model = new Linear(1, 1, { rng });
  const xs = Array.from({ length: 50 }, (_, i) => [i / 10]);
  const ys = xs.map(([x]) => [2 * x + 1]);
  const X = Tensor.of(xs);
  const y = Tensor.of(ys);
  const opt = new SGD(model.parameters(), { lr: 0.05 });
  for (let i = 0; i < 500; i++) {
    opt.zeroGrad();
    const loss = mseLoss(model.forward(X), y);
    loss.backward();
    opt.step();
  }
  const w = model.weight.toArray()[0][0];
  const b = model.bias.toArray()[0];
  assert.ok(Math.abs(w - 2) < 0.05, `w=${w}`);
  assert.ok(Math.abs(b - 1) < 0.1, `b=${b}`);
});
