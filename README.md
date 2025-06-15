# Neural Networks: Hopfield, Perceptron, and Feedforward Networks

This project implements classic neural network models in C++ for associative memory, pattern recognition, and supervised learning. It now supports **Hopfield networks**, **single-layer perceptrons**, and **multilayer feedforward (MLP) networks** with modular, extensible OOP design.

## Features

- **Modular OOP design:** Core abstractions for `Model`, `Layer`, and `LearningRule` allow easy extension and swapping of components.
- **Feedforward (MLP) network:** Supports arbitrary depth, customizable layer sizes, and pluggable learning rules (SGD, etc.).
- **Perceptron:** Classic single-layer perceptron for supervised learning.
- **Hopfield network:** Associative memory with Hebbian learning for binary pattern storage and recall.
- **Flexible learning rules:** Learning algorithms (SGD, Hebbian, etc.) are implemented as separate classes and can be assigned per layer.
- **Pattern visualization:** Print patterns and outputs as ASCII art in the terminal.
- **OpenCV integration:** Load and convert PNG images as patterns.
- **Hamming distance analysis:** Quantitatively compare pattern similarity.
- **Flexible pattern loading:** Easily add new patterns from images or code.

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
     For MLP:  
     \[
     w_{ij} \leftarrow w_{ij} - \eta \frac{\partial L}{\partial w_{ij}}
     \]
   - **Hopfield:**  
     Unsupervised Hebbian rule:  
     \[
     w_{ij} = \frac{1}{P} \sum_{p=1}^{P} s_i^p s_j^p
     \]

3. **Recall/Inference:**  
   - **Feedforward/Perceptron:**  
     Forward pass through the network to compute outputs.
   - **Hopfield:**  
     Iterative update until convergence to a stored pattern.

4. **Visualization:**  
   Patterns and outputs can be printed as ASCII art in the terminal for easy inspection.

## Build Instructions

### Prerequisites

- C++20 compiler (e.g., g++ 10+)
- [OpenCV 4](https://opencv.org/) (for image loading)
- [CMake 3.10+](https://cmake.org/)

### Build Steps

```sh
# Clone the repository
git clone <your-repo-url>
cd NeuralNetworks

# Create a build directory and compile
mkdir build
cd build
cmake ..
make
```

This will produce an executable named `Network`.

## Example

```cpp
Patterns inputs = {
    png_to_bits("pattern1.png"),
    png_to_bits("pattern2.png"),
    // ...
};
Patterns labels = {
    {1.0, 0.0},
    {0.0, 1.0},
    // ...
};

size_t N = inputs[0].size();
size_t O = labels[0].size();
std::vector<size_t> layers = { N, 16, 8, O };
FeedforwardNetwork mlp(layers);

Scalar learningRate = 0.1f;
size_t epochs = 10000;
mlp.learn(inputs, labels, learningRate, epochs);

for (const auto &input : inputs) {
    Pattern output = mlp.infer(input);
    // Print or process output
}
```
## License

This project is open source and available under the MIT License.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- Classic Hopfield network theory
- [stb_image](https://github.com/nothings/stb) (optional, for image loading)


