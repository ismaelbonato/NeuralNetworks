# Neural Networks: Hopfield, Perceptron, and Feedforward Networks

This project implements classic neural network models in C++ for associative memory, pattern recognition, and supervised learning. It now supports **Hopfield networks**, **single-layer perceptrons**, and **multilayer feedforward (MLP) networks** with modular, extensible OOP design.

## Features

- **Modular OOP design:** Core abstractions for `Model`, `Layer`, and `LearningRule` allow easy extension and swapping of components.
- **Feedforward (MLP) network:** Supports arbitrary depth, customizable layer sizes, and pluggable learning rules (SGD, etc.).
- **Perceptron:** Classic single-layer perceptron for supervised learning.
- **Hopfield network:** Associative memory with Hebbian learning for binary pattern storage and recall.
- **Flexible learning rules:** Learning algorithms (SGD, Hebbian, etc.) are implemented as separate classes and can be assigned per layer.

## Supported Networks

- **Feedforward (MLP) Network:**  
  - Arbitrary number of layers and neurons per layer.
  - Pluggable activation functions (sigmoid, ReLU, etc.).
  - Stochastic Gradient Descent (SGD) and other learning rules.
- **Perceptron Network:**  
  - Single-layer, supervised learning with step activation.
- **Hopfield Network:**  
  - Unsupervised associative memory with Hebbian learning.

## How It Works

1. **Define Patterns:**  
   Patterns (e.g., 8x8 or 16x16 binary images) are stored as arrays of -1s and +1s (Hopfield) or 0s and 1s (Perceptron/MLP).

2. **Learning:**  
   - **Feedforward/Perceptron:**  
     Supervised learning using SGD or other rules.  
   - **Hopfield:**  
     Unsupervised Hebbian rule:  

3. **Recall/Inference:**  
   - **Feedforward/Perceptron:**  
     Forward pass through the network to compute outputs.
   - **Hopfield:**  
     Iterative update until convergence to a stored pattern.

4. **Visualization:**  
   Patterns and outputs can be printed as ASCII art in the terminal for easy inspection.

## Build Instructions

### Dependencies

- **C++23 compiler** (e.g., g++ 13+ or clang++ 16+)
- **OpenCV 4** (for image loading)
- **CMake 3.10+**

## Building the Project

This project builds both a reusable library (`NetworkLib`) and an executable (`Network`) when built directly.

### Steps

```sh
# Clone the repository
git clone <your-repo-url>
cd NeuralNetworksUI/NeuralNetwork

# Create a build directory and compile
mkdir build
cd build
cmake ..
make
```

- The static/shared library `NetworkLib` will be built.
- The executable `Network` will be built if you build this project directly.

## Notes

- When used as a dependency in another CMake project, only the library is built.
- Make sure OpenCV is installed and discoverable by CMake.