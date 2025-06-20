#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <random>
#include <stdexcept>

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete; // Default constructor is not allowed

    DenseLayer(const LayerConfig &newConfig)
        : Layer(newConfig)
    {
        initWeights();
    }

    ~DenseLayer() override = default;

    std::shared_ptr<Layer> clone() const override
    {
        return std::make_shared<DenseLayer>(config);
    }

    // each layer should initialize its weights
    void initWeights(Scalar value = Scalar{}) override
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Scalar> dis(-1.0, 1.0);

        if (weights.empty()) {
            weights.resize(config.outputSize, Pattern(config.inputSize, value));

            for (auto &row : weights)
                for (auto &w : row)
                    w = dis(gen);
        }
        if (biases.empty()) {
            biases.resize(config.outputSize, value);

            for (auto &b : biases)
                b = dis(gen);
        }
    }
};