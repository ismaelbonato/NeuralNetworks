#pragma once

#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/Initializer.h"
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

    std::shared_ptr<Initializer<Scalar>> weightInitializer =
        std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0}, Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer =
        std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});

    bool isValid() const;
};

struct LayerParameters
{
    Pattern weights;
    Pattern biases;
};

template<typename LayerType>
std::shared_ptr<LayerType> makeLayer(const LayerConfig &config);

class Layer
{
protected:
    LayerConfig config;
    Pattern weights;
    Pattern biases;

    virtual Shape expectedWeightShape() const = 0;
    virtual Shape expectedBiasShape() const = 0;
    bool hasBias() const;
    Pattern initializeParameter(const Shape &shape,
                                const std::shared_ptr<Initializer<Scalar>> &initializer,
                                Scalar fallbackValue = Scalar{});


public:

    Layer() = delete;
    Layer(const LayerConfig &newConfig);
    virtual ~Layer();

    void initializeParameters(Scalar value = Scalar{});
    size_t getInputSize() const;
    size_t getOutputSize() const;
    const Pattern &getWeights() const;
    const Pattern &getBiases() const;
    LayerParameters getParameters() const;
    void setParameters(const LayerParameters &parameters);
    void setWeights(const Pattern &newWeights);
    void setBiases(const Pattern &newBiases);
    bool isInitialized() const;
    void requireInitialized() const;

    virtual void updateWeights(const Pattern &prev_activations,
                               const Pattern &layerDelta,
                               Scalar learningRate);

    virtual Pattern infer(const Pattern &input) const;
    virtual Pattern weightedSum(const Pattern &input) const;
    virtual Pattern activationDerivatives(const Pattern &values) const;
    virtual Pattern activate(const Pattern &values) const;

    Pattern backwardPass(const Pattern &layerDelta, const Pattern &preActivation);
    LayerParameters naturalUpdatedParameters(const LayerParameters &parameters) const;
};
