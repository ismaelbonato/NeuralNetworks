# nn-runtime

Small C++23 runtime inference library for simple neural-network style layers.
The project currently focuses on deterministic forward inference, explicit
parameter ownership on layers, and a compact test suite around runtime behavior.

## Features

- `Model` composes runtime `Layer` instances in order.
- Parameterized layers own their weights and biases directly.
- Supported runtime layers:
  - `DenseLayer`
  - `ConvolutionalLayer`
  - `FlattenLayer`
  - `HopfieldLayer`
- Shared tensor utilities for vectors, matrices, shaped storage, and basic
  operations used by inference.
- Runtime fixtures cover OR, AND, XOR, perceptron, Hopfield recall, and 1D
  convolution behavior.

## Build

Dependencies:

- CMake 3.20+
- C++23 compiler
- Catch2 3 for tests

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

When built directly, the project creates:

- `nn-runtime`: reusable runtime library
- `nn-runtime-main`: tiny executable that runs a static XOR inference fixture
- `nn-runtime-tests`: Catch2 test executable, when `BUILD_TESTING` is enabled
