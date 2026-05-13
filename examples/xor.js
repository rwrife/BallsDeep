// examples/xor.js — train a tiny MLP on XOR using the autograd Tensor API.
import { Tensor } from '../src/tensor.js';
import { nn, optim, RNG } from '../src/index.js';

const rng = new RNG(42);
const model = new nn.Sequential(
  new nn.Linear(2, 8, { rng }),
  new nn.ReLU(),
  new nn.Linear(8, 1, { rng }),
  new nn.Sigmoid(),
);

const X = Tensor.of([[0, 0], [0, 1], [1, 0], [1, 1]]);
const y = Tensor.of([[0], [1], [1], [0]]);

const opt = new optim.Adam(model.parameters(), { lr: 0.1 });

for (let epoch = 0; epoch <= 1500; epoch++) {
  opt.zeroGrad();
  const out = model.forward(X);
  const loss = nn.bceLoss(out, y);
  loss.backward();
  opt.step();
  if (epoch % 250 === 0) {
    console.log(`epoch ${epoch}\tloss=${loss.toArray().toFixed(6)}`);
  }
}

console.log('\nfinal predictions:');
for (const [row, p] of model.forward(X).toArray().entries()) {
  console.log(`  x=${X.toArray()[row].join(',')}  ->  ${p[0].toFixed(4)}`);
}
