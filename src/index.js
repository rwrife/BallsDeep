// src/index.js — single-import entry point.
export { NDArray, broadcastShapes } from './ndarray.js';
export * as ops from './ops.js';
export { RNG, defaultRng } from './random.js';
export { Tensor, tensor } from './tensor.js';
export * as nn from './nn/index.js';
export * as optim from './optim/index.js';
export * as data from './data/index.js';
export * as models from './models/index.js';
