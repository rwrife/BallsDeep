// examples/binary.js — train logistic regression on binary.csv.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { data, models } from '../src/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const csv = readFileSync(join(__dirname, '..', 'binary.csv'), 'utf8');
const { X, y, featureNames } = data.csvToXY(csv, 'label');

console.log(`features: ${featureNames.join(', ')}`);
console.log(`samples:  ${X.length}`);

const model = new models.LogisticRegression({ lr: 0.5, epochs: 2000, verbose: true }).fit(X, y);

const probs = model.predictProba(X);
const preds = model.predict(X);
console.log('\npredictions vs truth:');
for (let i = 0; i < X.length; i++) {
  console.log(`  x=${X[i].join(',')}  pred=${preds[i]} (p=${probs[i].toFixed(3)})  truth=${y[i]}`);
}
console.log(`accuracy: ${data.accuracy(preds, y)}`);
