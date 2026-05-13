// src/nn/activations.js
import { Module } from './module.js';

export class ReLU extends Module {
  forward(x) { return x.relu(); }
}
export class Sigmoid extends Module {
  forward(x) { return x.sigmoid(); }
}
export class Tanh extends Module {
  forward(x) { return x.tanh(); }
}
export class Softmax extends Module {
  forward(x) { return x.softmax(); }
}
export class LogSoftmax extends Module {
  forward(x) { return x.logSoftmax(); }
}
