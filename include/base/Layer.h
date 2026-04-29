#pragma once

#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/Initializer.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <string>

struct LayerRecipe
{
    std::string name;
    std::string type;
    std::string info;
    std::shared_ptr<ActivationFunction<Scalar>> activation;
};

struct ConvolutionalLayerRecipe : LayerRecipe
{
    size_t inputChannels = 0;
    size_t inputLength = 0;
    size_t outputChannels = 0;
    size_t kernelSize = 0;
    size_t stride = 1;
    size_t padding = 0;

    std::shared_ptr<Initializer<Scalar>> weightInitializer
        = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0},
                                                       Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer
        = std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});

    bool isValid() const;
};

struct DenseLayerRecipe : LayerRecipe
{
    size_t inputSize = 0;
    size_t outputSize = 0;
    Shape expectedInputShape;
    Shape expectedOutputShape;

    std::shared_ptr<Initializer<Scalar>> weightInitializer
        = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0},
                                                       Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer
        = std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});

    bool isValid() const;
};

struct HopfieldLayerRecipe : LayerRecipe
{
    size_t size = 0;
    Shape expectedShape;

    std::shared_ptr<Initializer<Scalar>> weightInitializer
        = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0},
                                                       Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer
        = std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});

    bool isValid() const;
};

struct FlattenLayerRecipe : LayerRecipe
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

template<typename LayerType, typename RecipeType>
std::unique_ptr<LayerType> makeLayer(const RecipeType &recipe);

class Layer
{
protected:
    LayerRecipe recipe;
    Shape expectedInput;
    Shape expectedOutput;

    void requireInputShape(const Pattern &input) const;
    virtual Pattern forward(const Pattern &input) const = 0;

public:
    Layer() = delete;
    Layer(const LayerRecipe &newRecipe,
          const Shape &newExpectedInput,
          const Shape &newExpectedOutput);
    virtual ~Layer();

    size_t getInputSize() const;
    size_t getOutputSize() const;
    const Shape &getExpectedInputShape() const;
    const Shape &getExpectedOutputShape() const;
    const Shape &getInputShape() const;
    const Shape &getOutputShape() const;
    const std::shared_ptr<ActivationFunction<Scalar>> &getActivation() const;

    Pattern infer(const Pattern &input) const;
};
