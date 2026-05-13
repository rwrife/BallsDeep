// test/data.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV, csvToXY, StandardScaler, trainTestSplit, oneHot, accuracy }
  from '../src/data/index.js';

test('parseCSV handles quoted fields and CRLF', () => {
  const text = 'a,b,c\r\n1,"hi, there",3\r\n4,5,6\r\n';
  const { headers, rows } = parseCSV(text);
  assert.deepEqual(headers, ['a', 'b', 'c']);
  assert.deepEqual(rows, [['1', 'hi, there', '3'], ['4', '5', '6']]);
});

test('csvToXY peels label column', () => {
  const text = 'a,b,label\n1,2,0\n3,4,1\n';
  const { X, y, featureNames } = csvToXY(text, 'label');
  assert.deepEqual(X, [[1, 2], [3, 4]]);
  assert.deepEqual(y, [0, 1]);
  assert.deepEqual(featureNames, ['a', 'b']);
});

test('StandardScaler standardizes per-column', () => {
  const X = [[1, 100], [2, 200], [3, 300], [4, 400]];
  const s = new StandardScaler().fit(X);
  const out = s.transform(X).toArray();
  for (let j = 0; j < 2; j++) {
    const col = out.map((r) => r[j]);
    const m = col.reduce((a, b) => a + b, 0) / col.length;
    assert.ok(Math.abs(m) < 1e-9);
  }
});

test('trainTestSplit + oneHot + accuracy', () => {
  const X = Array.from({ length: 100 }, (_, i) => [i]);
  const y = Array.from({ length: 100 }, (_, i) => i % 3);
  const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize: 0.2 });
  assert.equal(XTrain.length + XTest.length, 100);
  assert.equal(yTrain.length, XTrain.length);
  const oh = oneHot([0, 1, 2, 1], 3);
  assert.deepEqual(oh.toArray(), [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]);
  assert.equal(accuracy([1, 1, 0], [1, 0, 0]), 2 / 3);
});
