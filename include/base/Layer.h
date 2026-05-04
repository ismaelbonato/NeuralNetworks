#pragma once

#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/Parameters.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <optional>
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

    bool isValid() const;
};

struct DenseLayerRecipe : LayerRecipe
{
    size_t inputSize = 0;
    size_t outputSize = 0;
    Shape expectedInputShape;
    Shape expectedOutputShape;

    bool isValid() const;
};

struct HopfieldLayerRecipe : LayerRecipe
{
    size_t size = 0;
    Shape expectedShape;

    bool isValid() const;
};

struct FlattenLayerRecipe : LayerRecipe
{
    Shape expectedInputShape;

    bool isValid() const;
    Shape expectedOutputShape() const;
};

class Layer
{
protected:
    LayerRecipe recipe;
    Shape expectedInput;
    Shape expectedOutput;
    std::optional<Parameters> ownedParameters;

    void requireInputShape(const Pattern &input) const;
    virtual Pattern forward(const Pattern &input) const;
    virtual Pattern forward(const Pattern &input,
                            const Parameters &parameters) const;
    virtual Pattern weightedInput(const Pattern &input,
                                  const Parameters &parameters) const;
    Pattern activate(const Pattern &values) const;

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
    virtual bool usesParameters() const;
    virtual Shape expectedWeightShape() const;
    virtual Shape expectedBiasShape() const;
    virtual bool acceptsParameters(const Parameters &parameters) const;
    virtual void requireValidParameters(const Parameters &parameters) const;
    std::optional<Parameters> parameters() const;
    const Parameters &getParameters() const;
    void setParameters(const Parameters &parameters);
    void requireParameters() const;

    Pattern infer(const Pattern &input) const;
};
