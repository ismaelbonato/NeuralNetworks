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
    std::string name;
    std::string type;
    std::string info;
};

struct TrainableLayerConfig : LayerConfig
{
    std::shared_ptr<LearningRule<Scalar>> learningRule;
    std::shared_ptr<ActivationFunction<Scalar>> activation;

    std::shared_ptr<Initializer<Scalar>> weightInitializer =
        std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0}, Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer =
        std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});
};

struct DenseLayerConfig : TrainableLayerConfig
{
    size_t inputSize = 0;
    size_t outputSize = 0;
    Shape expectedInputShape;
    Shape expectedOutputShape;

    bool isValid() const;
};

struct HopfieldLayerConfig : TrainableLayerConfig
{
    size_t size = 0;
    Shape expectedShape;

    bool isValid() const;
};

struct FlattenLayerConfig : LayerConfig
{
    Shape expectedInputShape;

    bool isValid() const;
    Shape expectedOutputShape() const;
};

struct LayerParameters
{
    Pattern weights;
    Pattern biases;
};

template<typename LayerType, typename ConfigType>
std::unique_ptr<LayerType> makeLayer(const ConfigType &config);

class Layer
{
protected:
    LayerConfig config;
    Shape expectedInput;
    Shape expectedOutput;

    void requireInputShape(const Pattern &input) const;


public:

    Layer() = delete;
    Layer(const LayerConfig &newConfig,
          const Shape &newExpectedInput,
          const Shape &newExpectedOutput);
    virtual ~Layer();

    size_t getInputSize() const;
    size_t getOutputSize() const;
    const Shape &getExpectedInputShape() const;
    const Shape &getExpectedOutputShape() const;
    const Shape &getInputShape() const;
    const Shape &getOutputShape() const;
    virtual bool isTrainable() const;

    virtual Pattern infer(const Pattern &input) const = 0;
};

class TrainableLayer : public Layer
{
protected:
    TrainableLayerConfig trainableConfig;
    Pattern weights;
    Pattern biases;

    TrainableLayer(const TrainableLayerConfig &newConfig,
                   const Shape &newExpectedInput,
                   const Shape &newExpectedOutput);

    virtual Shape expectedWeightShape() const = 0;
    virtual Shape expectedBiasShape() const = 0;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern initializeParameter(const Shape &shape,
                                const std::shared_ptr<Initializer<Scalar>> &initializer,
                                Scalar fallbackValue = Scalar{});

public:
    ~TrainableLayer() override;

    bool isTrainable() const override;
    void initializeParameters(Scalar value = Scalar{});
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

    Pattern infer(const Pattern &input) const override;
    virtual Pattern weightedSum(const Pattern &input) const;
    virtual Pattern activationDerivatives(const Pattern &values) const;
    virtual Pattern activate(const Pattern &values) const;

    virtual Pattern backwardPass(const Pattern &layerDelta, const Pattern &preActivation) const;
    LayerParameters naturalUpdatedParameters(const LayerParameters &parameters,
                                             Scalar mutationStrength) const;
};
