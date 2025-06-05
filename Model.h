#ifndef MODEL_H
#define MODEL_H
#include "Layer.h" // Include the header file where Layer is defined
#include "LearningRule.h"
#include <cstddef> // Include for size_t
#include <memory>
#include <vector>

class Model
{
public:
    Model() = default;

    // Constructor that takes ownership of a LearningRule and sets the number of neurons
    Model(size_t n)
        : nNeurons(n)
    {}

    virtual ~Model() = default;

protected:
    size_t nNeurons;
    std::vector<std::unique_ptr<Layer>> layers;

public:
    // Add a layer to the model
    void addLayer(std::unique_ptr<Layer> layer) { layers.push_back(std::move(layer)); }

    // Remove a layer by index
    void removeLayer(size_t index)
    {
        if (index < layers.size()) {
            layers.erase(
                layers.begin()
                + static_cast<std::vector<std::unique_ptr<Layer>>::difference_type>(
                    index));
        }
    }

    // Call the learning rule's learn method unsupervised
    void learn(const std::vector<Pattern> &patterns)
    {
        for (auto &layer : layers) {
            layer->learn(patterns);
        }
    }

    // Call the learning rule's learn method supervised
    void learn(const std::vector<Pattern> &inputs, const std::vector<Pattern> &labels)
    {
        for (auto &layer : layers) {
            layer->learn(inputs, labels);
        }
    }

    Pattern forward(const Pattern &input)
    {
        if (!layers.empty()) {
            return layers.front()->forward(input);
        }
        return input; // or throw an exception if no layers exist
    }
    //virtual OutputType predict(const InputType& input) const = 0;

private:
    // You can add private members or methods if needed
};

#endif // MODEL_H