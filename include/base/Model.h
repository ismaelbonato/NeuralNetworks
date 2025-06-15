#pragma once

#include "base/Layer.h" // Include the header file where Layer is defined
#include "base/LearningRule.h"
#include <cstddef> // Include for size_t
#include <memory>
#include <vector>

#include <omp.h>

class Model
{
public:
    Model() = default;

    // Constructor that takes ownership of a LearningRule and sets the number of neurons
    Model(size_t) {}

    virtual ~Model() = default;

    std::vector<std::unique_ptr<Layer>> layers;

public:
    // Add a layer to the model
    //void addLayer(std::unique_ptr<Layer> layer) { layers.push_back(layer); }

    // Remove a layer by index
    void removeLayer(size_t index)
    {
        if (index < layers.size()) {
            layers.erase(
                layers.begin()
                + static_cast<
                    std::vector<std::shared_ptr<Layer>>::difference_type>(
                    index));
        }
    }

    Pattern lossDerivative(const Pattern &output, const Pattern &target)
    {
        Pattern result(output.size(), 0.0);

        for (size_t i = 0; i < output.size(); ++i) {
            result[i] = output[i] - target[i];
        }
        return result;
    }




    // Apply the Hebbian learning rule to update weights
    virtual void learn(const std::vector<Pattern> &)
    {
        std::cerr << "Learn method not implemented for this model type."
                  << std::endl;
        throw std::runtime_error("Learn method not implemented for this model type.");
    }

    virtual void learn(const std::vector<Pattern> &inputs,
                       const std::vector<Pattern> &labels,
                       float learningRate = 0.1f,
                       size_t epochs = 100000)
        = 0;

    Pattern infer(const Pattern &input)
    {
        if (layers.empty()) {
            throw std::runtime_error(
                "No layers exist in the model to perform inference.");
        }

        Pattern output = input;
        for (auto &layer : layers) {
            output = layer->infer(output);
        }
        return output;
    }

private:
    // You can add private members or methods if needed
};
