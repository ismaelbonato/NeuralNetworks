#pragma once

#include "base/Layer.h" // Include the header file where Layer is defined
#include "base/LearningRule.h"
#include "base/Types.h" // Include for Pattern and Scalar types
#include <cstddef> // Include for size_t
#include <memory>
#include <vector>

class Model
{
protected:
    Layers layers;

public:

    Model() = default;

    Model(Layers &newLayers)
        : layers(newLayers)
    {}

    Model(const std::initializer_list<std::shared_ptr<Layer>> newLayers)
        : layers(newLayers)
    {}

    virtual ~Model() = default;

    virtual void addLayer(const std::shared_ptr<Layer> &layer)
    {
        layers.push_back(layer);
    }

    virtual void learn(const Patterns &inputs,
                       const Patterns &labels,
                       Scalar learningRate = 0.1f,
                       size_t epochs = 100000)
        = 0;

    virtual Pattern infer(const Pattern &input) = 0;

};
