// src/models/knn.js
// k-Nearest-Neighbors classifier / regressor. Brute-force, no kd-tree.

function dist2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

export class KNeighborsClassifier {
  constructor(opts = {}) {
    this.k = opts.k ?? 5;
    this.weights = opts.weights ?? 'uniform'; // 'uniform' | 'distance'
    this.X = null; this.y = null;
  }
  fit(X, y) { this.X = X; this.y = y; return this; }
  _neighbors(x) {
    const dists = this.X.map((xi, i) => ({ d: dist2(xi, x), y: this.y[i] }));
    dists.sort((a, b) => a.d - b.d);
    return dists.slice(0, this.k);
  }
  predict(X) {
    return X.map((x) => {
      const nb = this._neighbors(x);
      const tally = new Map();
      for (const { d, y } of nb) {
        const w = this.weights === 'distance' ? 1 / (Math.sqrt(d) + 1e-9) : 1;
        tally.set(y, (tally.get(y) || 0) + w);
      }
      let bestK = null, bestV = -Infinity;
      for (const [k, v] of tally) if (v > bestV) { bestV = v; bestK = k; }
      return bestK;
    });
  }
}

export class KNeighborsRegressor {
  constructor(opts = {}) {
    this.k = opts.k ?? 5;
    this.weights = opts.weights ?? 'uniform';
    this.X = null; this.y = null;
  }
  fit(X, y) { this.X = X; this.y = y; return this; }
  predict(X) {
    return X.map((x) => {
      const dists = this.X.map((xi, i) => ({ d: dist2(xi, x), y: this.y[i] }));
      dists.sort((a, b) => a.d - b.d);
      const nb = dists.slice(0, this.k);
      if (this.weights === 'distance') {
        let num = 0, den = 0;
        for (const { d, y } of nb) {
          const w = 1 / (Math.sqrt(d) + 1e-9);
          num += w * y; den += w;
        }
        return num / den;
      }
      return nb.reduce((s, { y }) => s + y, 0) / nb.length;
    });
  }
}
