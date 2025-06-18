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

    Model(Layers newLayers)
        : layers(std::move(newLayers))
    {}

    virtual ~Model() = default;

    virtual void addLayer(const Layer &l)
    {
        layers.emplace_back(l.clone());
    }

    virtual void learn(const Patterns &inputs,
                       const Patterns &labels,
                       Scalar learningRate = 0.1f,
                       size_t epochs = 100000)
        = 0;

    virtual Pattern infer(const Pattern &input) = 0;

};
