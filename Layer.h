#ifndef LAYER_H
#define LAYER_H

#include <vector>

using Pattern = std::vector<double>;

class Layer {
protected:
    size_t inputSize;
    size_t outputSize;
public:
    Layer(size_t in, size_t out) : inputSize(in), outputSize(out) {}

    // Constructor to receive a unique_ptr and a size_t
    size_t getInputSize() const { return inputSize; }
    size_t getOutputSize() const { return outputSize; }

    virtual ~Layer() = default;

    virtual int activation(double value) const = 0;
    virtual Pattern forward(const Pattern& input) const = 0;
    virtual void setWeights(const std::vector<Pattern>& ws) = 0;
   //virtual void learn(const std::vector<Pattern>& patterns) = 0;

    //virtual void backward(const std::vector<Pattern>& patterns);
};

#endif // LAYER_H