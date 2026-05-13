// examples/periodic.js — predict atomic weight from periodic.csv numeric features.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { data, models } from '../src/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const csv = readFileSync(join(__dirname, '..', 'periodic.csv'), 'utf8');
const { headers, rows } = data.parseCSV(csv);

// Drop non-numeric columns; keep numeric ones; predict Atomic Weight.
const dropCols = new Set([
  'Element', 'Symbol', 'Phase', 'Most Stable Crystal', 'Type',
  'Discoverer', 'Year of Discovery', 'Electron Configuration',
  'Display Row', 'Display Column',
]);
const target = 'Atomic Weight';
const keepIdx = headers.map((h, i) => ({ h, i }))
  .filter(({ h }) => !dropCols.has(h) && h !== target)
  .map(({ i }) => i);
const targetIdx = headers.indexOf(target);

const X = rows.map((r) => keepIdx.map((i) => parseFloat(r[i]) || 0));
const y = rows.map((r) => parseFloat(r[targetIdx]) || 0);

const split = data.trainTestSplit(X, y, { testSize: 0.2 });

// Standardize features so KNN distances are sane.
const scaler = new data.StandardScaler().fit(split.XTrain);
const XTrainS = scaler.transform(split.XTrain).toArray();
const XTestS  = scaler.transform(split.XTest).toArray();

const knn = new models.KNeighborsRegressor({ k: 3, weights: 'distance' })
  .fit(XTrainS, split.yTrain);
const preds = knn.predict(XTestS);

let mae = 0;
for (let i = 0; i < preds.length; i++) mae += Math.abs(preds[i] - split.yTest[i]);
mae /= preds.length;
console.log(`KNN regressor MAE on held-out elements: ${mae.toFixed(3)}`);

console.log('\nsample predictions:');
for (let i = 0; i < Math.min(8, preds.length); i++) {
  console.log(`  predicted=${preds[i].toFixed(2).padStart(7)}   truth=${split.yTest[i].toFixed(2)}`);
}
