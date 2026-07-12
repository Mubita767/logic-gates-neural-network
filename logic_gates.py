#!/usr/bin/env python3
"""
Logic Gates Neural Network
============================
AND, OR, NOT implementations with single neurons.
XOR implementation with multi-layer neural network.

Author  : Mupo Mubita
Email   : mubitamupo@outlook.com
WhatsApp: +260760457622
GitHub  : https://github.com/Mubita767
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(output):
    return output * (1 - output)


class SingleNeuron:
    """Single neuron with sigmoid activation for linearly separable gates."""

    def __init__(self, n_inputs, lr=0.5):
        self.w = np.random.randn(n_inputs) * 0.5
        self.b = np.random.randn() * 0.5
        self.lr = lr

    def forward(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

    def train(self, X, y, epochs=5000):
        y = y.flatten()
        for _ in range(epochs):
            out = self.forward(X)
            err = y - out
            d = err * sigmoid_deriv(out)
            self.w += self.lr * np.dot(X.T, d)
            self.b += self.lr * np.sum(d)

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


class NeuralNet:
    """Multi-layer neural network with one hidden layer."""

    def __init__(self, ni, nh, no, lr=0.8):
        self.lr = lr
        self.W1 = np.random.randn(ni, nh) * np.sqrt(2.0 / ni)
        self.b1 = np.zeros((1, nh))
        self.W2 = np.random.randn(nh, no) * np.sqrt(2.0 / nh)
        self.b2 = np.zeros((1, no))

    def forward(self, X):
        self.a1 = sigmoid(np.dot(X, self.W1) + self.b1)
        return sigmoid(np.dot(self.a1, self.W2) + self.b2)

    def train(self, X, y, epochs=20000):
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            out = self.forward(X)
            d2 = (y - out) * sigmoid_deriv(out)
            d1 = np.dot(d2, self.W2.T) * sigmoid_deriv(self.a1)
            self.W2 += self.lr * np.dot(self.a1.T, d2)
            self.b2 += self.lr * np.sum(d2, axis=0, keepdims=True)
            self.W1 += self.lr * np.dot(X.T, d1)
            self.b1 += self.lr * np.sum(d1, axis=0, keepdims=True)

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


def test_gate(name, neuron, X, y, inputs):
    preds = neuron.predict(X)
    correct = np.sum(preds.flatten() == y.flatten())
    print(f"\n{name} Gate: {correct}/{len(y)} correct")
    print("-" * 30)
    for i in range(len(y)):
        status = "PASS" if preds[i] == y[i] else "FAIL"
        print(f"  {inputs[i]} -> Expected: {y[i]}, Got: {preds[i]} [{status}]")
    return correct == len(y)


def main():
    print("=" * 50)
    print("LOGIC GATES - NEURAL NETWORK DEMONSTRATION")
    print("By Mupo Mubita")
    print("=" * 50)

    X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # AND Gate
    and_n = SingleNeuron(2, 0.5)
    and_n.train(X2, np.array([0, 0, 0, 1]), 5000)
    test_gate("AND", and_n, X2, [0, 0, 0, 1], ["0,0", "0,1", "1,0", "1,1"])

    # OR Gate
    or_n = SingleNeuron(2, 0.5)
    or_n.train(X2, np.array([0, 1, 1, 1]), 5000)
    test_gate("OR", or_n, X2, [0, 1, 1, 1], ["0,0", "0,1", "1,0", "1,1"])

    # NOT Gate
    X1 = np.array([[0], [1]])
    not_n = SingleNeuron(1, 0.5)
    not_n.train(X1, np.array([1, 0]), 5000)
    test_gate("NOT", not_n, X1, [1, 0], ["0", "1"])

    # XOR - Single Neuron (expected to fail)
    print("\n" + "=" * 50)
    print("XOR CHALLENGE")
    print("=" * 50)

    xor_single = SingleNeuron(2, 0.5)
    xor_single.train(X2, np.array([0, 1, 1, 0]), 10000)
    passed = test_gate("XOR (Single Neuron)", xor_single, X2, [0, 1, 1, 0], ["0,0", "0,1", "1,0", "1,1"])

    if not passed:
        print("\n  >> Single neuron cannot solve XOR (linearly inseparable)")
        print("  >> Switching to multi-layer neural network...")

        np.random.seed(123)
        xor_nn = NeuralNet(2, 4, 1, 0.8)
        xor_nn.train(X2, np.array([0, 1, 1, 0]), 20000)
        test_gate("XOR (4 Hidden Neurons)", xor_nn, X2, [0, 1, 1, 0], ["0,0", "0,1", "1,0", "1,1"])
        print("\n  >> Hidden layer enables non-linear decision boundary!")


if __name__ == "__main__":
    main()
