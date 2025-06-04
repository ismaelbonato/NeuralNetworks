# Hopfield Network

This project implements a classic **Hopfield neural network** in C++ for associative memory and pattern recognition. It supports learning and recalling binary patterns, such as 8x8 or 16x16 images (e.g., bitmap digits or font glyphs), using the Hebbian learning rule.



## Features

- **Modular OOP design:** The codebase is now structured around `Model`, `Layer`, and `LearningRule` classes, making it easy to extend or swap components.
- **Associative memory:** Store and recall binary patterns, even from noisy or incomplete inputs.
- **Hebbian learning:** Classic unsupervised learning rule for weight calculation, implemented as a separate class.
- **Pattern visualization:** Print patterns and outputs as ASCII art in the terminal.
- **OpenCV integration:** Load and convert PNG images as patterns.
- **Hamming distance analysis:** Quantitatively compare pattern similarity.
- **Flexible pattern loading:** Easily add new patterns from images or code.
- **Modern CMake build:** Uses C++20, extra warnings, and links OpenCV/Eigen automatically.



## How It Works

1. **Define Patterns:**  
   Patterns (e.g., 8x8 or 16x16 binary images) are stored as arrays of -1s and +1s.

2. **Learning:**  
   The network uses the Hebbian rule (now a dedicated class) to compute weights from the provided patterns:
   \[
   w_{ij} = \frac{1}{P} \sum_{p=1}^{P} s_i^p s_j^p
   \]
   where \( s_i^p \) is the bipolar value (-1/+1) of neuron \( i \) in pattern \( p \).

3. **Recall:**  
   Given a (possibly noisy) input pattern, the network updates neuron states to converge to the closest stored pattern.

4. **Visualization:**  
   Patterns and outputs can be printed as ASCII art in the terminal for easy inspection.


## Build Instructions

### Prerequisites

- C++20 compiler (e.g., g++ 10+)
- [OpenCV 4](https://opencv.org/) (for image loading)
- [CMake 3.10+](https://cmake.org/)
- [Eigen3](https://eigen.tuxfamily.org/) (header-only, for future extensibility)

### Build Steps

```sh
# Clone the repository
git clone <your-repo-url>
cd NeuralNetworks/HopfieldNetwork

# Create a build directory and compile
mkdir build
cd build
cmake ..
make
```

This will produce an executable named `hopfield`.


## Usage

1. **Edit `main.cpp`** to define your patterns or load them from images (e.g., PNGs in the `Misc/` folder).
2. **Run the executable:**
   ```sh
   ./hopfield
   ```
3. **View the output** in your terminal.


## Example

```cpp
std::vector<Pattern> patterns = {
    png_to_bits("pattern1.png"),
    png_to_bits("pattern2.png")
};

Hopfield net(std::make_unique<HebbianRule>(), patterns[0].size());
net.learn(patterns);

Pattern noisy = png_to_bits("noisy_pattern.png");
Pattern recalled = net.forward(noisy);

// Print recalled pattern as ASCII art
for (size_t i = 0; i < recalled.size(); ++i) {
    if (i % 64 == 0) std::cout << '\n';
    std::cout << (recalled[i] > 0 ? " " : "x");
}
```



## Capacity

A classic Hopfield network can reliably store about **0.138 × N** patterns, where **N** is the number of neurons (bits in your pattern).  
- For 8×8 patterns (64 bits): ~8 patterns
- For 16×16 patterns (256 bits): ~35 patterns

Storing more patterns increases the risk of recall errors and spurious states.



## License

This project is open source and available under the MIT License.



## Acknowledgements

- [OpenCV](https://opencv.org/)
- Classic Hopfield network theory
- [stb_image](https://github.com/nothings/stb) (optional, for image loading)

