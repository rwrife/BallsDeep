// src/models/logistic-regression.js
// Binary logistic regression trained by gradient descent via Tensor.
import { Tensor } from '../tensor.js';
import { NDArray } from '../ndarray.js';
import { Linear } from '../nn/linear.js';
import { Sigmoid } from '../nn/activations.js';
import { Sequential } from '../nn/linear.js';
import { bceLoss } from '../nn/losses.js';
import { SGD } from '../optim/sgd.js';

export class LogisticRegression {
  constructor(opts = {}) {
    this.lr        = opts.lr        ?? 0.1;
    this.epochs    = opts.epochs    ?? 200;
    this.batchSize = opts.batchSize ?? null;
    this.threshold = opts.threshold ?? 0.5;
    this.verbose   = opts.verbose   ?? false;
    this.model = null;
  }
  fit(X, y) {
    const Xn = X instanceof NDArray ? X : NDArray.fromArray(X);
    const yn = (y instanceof NDArray ? y : NDArray.fromArray(y)).reshape([Xn.shape[0], 1]);
    const inF = Xn.shape[1];
    this.model = new Sequential(new Linear(inF, 1), new Sigmoid());
    const opt = new SGD(this.model.parameters(), { lr: this.lr });
    const xT = Tensor.of(Xn);
    const yT = Tensor.of(yn);
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      opt.zeroGrad();
      const out = this.model.forward(xT);
      const loss = bceLoss(out, yT);
      loss.backward();
      opt.step();
      if (this.verbose && (epoch % Math.max(1, Math.floor(this.epochs / 10)) === 0)) {
        // eslint-disable-next-line no-console
        console.log(`epoch ${epoch}\tloss=${loss.toArray().toFixed(6)}`);
      }
    }
    return this;
  }
  predictProba(X) {
    const Xn = X instanceof NDArray ? X : NDArray.fromArray(X);
    const out = this.model.forward(Tensor.of(Xn));
    return out.toArray().map((row) => (Array.isArray(row) ? row[0] : row));
  }
  predict(X) {
    return this.predictProba(X).map((p) => (p >= this.threshold ? 1 : 0));
  }
}
