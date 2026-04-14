#pragma once

#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <string>

struct LayerConfig
{
    std::shared_ptr<LearningRule<Scalar>> learningRule;
    std::shared_ptr<ActivationFunction<Scalar>> activation;
    size_t inputSize = 0;
    size_t outputSize = 0;
    std::string name;
    std::string type;
    std::string info;

    bool useBias = true;
    bool initWeights = true;
    Scalar weightInitScale = Scalar{1.0};
    Scalar biasInit = Scalar{0.0};

    bool isValid() const;
};

class Layer
{
protected:
    LayerConfig config;

public:
    Patterns weights;
    Pattern biases;

    Layer() = delete;
    Layer(const LayerConfig &newConfig);
    virtual ~Layer();

    virtual std::shared_ptr<Layer> clone() const = 0;
    virtual void initWeights(Scalar value = Scalar{}) = 0;

    size_t getInputSize() const;
    size_t getOutputSize() const;

    virtual void updateWeights(const Pattern &prev_activations,
                               const Pattern &layerDelta,
                               Scalar learningRate);

    virtual Pattern infer(const Pattern &input) const;
    virtual Pattern weightedSum(const Pattern &input) const;
    virtual Pattern activationDerivatives(const Pattern &values) const;
    virtual Pattern activate(const Pattern &values) const;

    Pattern backwardPass(const Pattern &layerDelta, const Pattern &preActivation);
    void naturalUpdateWeights(const Layer &l);
};
