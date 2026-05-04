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

## Dev Container / Docker Compose workflow

This repository ships a ready-to-use Docker Compose development container.

### 1) Generate local container environment

```sh
./setEnv.sh
```

This creates `.env` with your current user and group IDs, which are used by
`docker-compose.yml`.

### 2) Build and start the dev container

```sh
docker compose up -d --build
```

### 3) Enter the container shell

This opens a shell already positioned at the workspace folder
(`/home/${USER}/workspace`).

```sh
docker compose exec dev-cpp bash
```

### VS Code dev container (optional)

If you use VS Code, `./setEnv.sh` will run automatically when you open the repository, and you can use the "Remote Containers: Reopen in Container" command to enter the dev container with your editor.


### 4) Configure, build, and test

```sh
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug
```

Use `gcc-debug`, `gcc-release`, `clang-debug`, or `clang-release` depending on
the compiler and build type you want.

## Local build

```sh
cmake --preset gcc-debug
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

When built directly, the project creates:

- `nn-runtime`: reusable runtime library
- `nn-runtime-main`: tiny executable that runs a static XOR inference fixture
- `nn-runtime-tests`: Catch2 test executable, when `BUILD_TESTING` is enabled
