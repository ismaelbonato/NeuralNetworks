# Hopfield Network

This project implements a classic **Hopfield neural network** in C++ for associative memory and pattern recognition. It supports learning and recalling binary patterns, such as 8x8 or 16x16 images (e.g., bitmap digits or font glyphs), using the Hebbian learning rule.

---

## Features

- **Associative memory:** Store and recall binary patterns, even from noisy or incomplete inputs.
- **Hebbian learning:** Classic unsupervised learning rule for weight calculation.
- **Pattern visualization:** Print patterns and outputs as ASCII art in the terminal.
- **OpenCV integration:** Load and convert PNG images as patterns.
- **Hamming distance analysis:** Quantitatively compare pattern similarity.
- **Flexible pattern loading:** Easily add new patterns from images or code.

---

## How It Works

1. **Define Patterns:**  
   Patterns (e.g., 8x8 or 16x16 binary images) are stored as arrays of 0s and 1s.

2. **Learning:**  
   The network uses the Hebbian rule to compute weights from the provided patterns:
   \[
   w_{ij} = \frac{1}{P} \sum_{p=1}^{P} s_i^p s_j^p
   \]
   where \( s_i^p \) is the bipolar value (-1/+1) of neuron \( i \) in pattern \( p \).

3. **Recall:**  
   Given a (possibly noisy) input pattern, the network updates neuron states to converge to the closest stored pattern.

4. **Visualization:**  
   Patterns and outputs can be printed as ASCII art in the terminal for easy inspection.

---

## Build Instructions

### Prerequisites

- C++17 compiler (e.g., g++ 9+)
- [OpenCV 4](https://opencv.org/) (for image loading)
- [CMake 3.10+](https://cmake.org/)

### Build Steps

```sh
# Clone the repository
git clone <your-repo-url>
cd NeuralNetworks/HopefieldNetwork

# Create a build directory and compile
mkdir build
cd build
cmake ..
make
```

This will produce an executable named `hopfield`.

---

## Usage

1. **Edit `main.cpp`** to define your patterns or load them from images (e.g., PNGs in the `Misc/` folder).
2. **Run the executable:**
   ```sh
   ./hopfield
   ```
3. **View the output** in your terminal.

---

## Example

```cpp
std::vector<Pattern> patterns = {
    {0,1,1,1,1,1,1,0, ...}, // Pattern for digit '2'
    {0,1,1,1,1,1,1,0, ...}  // Pattern for digit '8'
};

Network hopfieldNetwork;
auto learned_weights = hopfieldNetwork.hebbianLearning(patterns);
hopfieldNetwork.updateNeurons(learned_weights);

hopfieldNetwork.activation(patterns[0]);
hopfieldNetwork.printOutput();
```

---

## Capacity

A classic Hopfield network can reliably store about **0.138 × N** patterns, where **N** is the number of neurons (bits in your pattern).  
- For 8×8 patterns (64 bits): ~8 patterns
- For 16×16 patterns (256 bits): ~35 patterns

Storing more patterns increases the risk of recall errors and spurious states.

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- Classic Hopfield network theory
- [stb_image](https://github.com/nothings/stb) (optional, for image loading)

---