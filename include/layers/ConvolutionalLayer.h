#pragma once

#include "base/Layer.h"

namespace nn {

struct ConvolutionalLayerRecipe : LayerRecipe
{
    size_t inputChannels = 0;
    size_t inputLength = 0;
    size_t outputChannels = 0;
    size_t kernelSize = 0;
    size_t stride = 1;
    size_t padding = 0;

    Shape getInputShape() const override;
    Shape getOutputShape() const override;
    void validateRecipe() const override;
};

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer() = delete;
    explicit ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe);
    ~ConvolutionalLayer() override;

    size_t getInputChannels() const;
    size_t getInputLength() const;
    size_t getOutputChannels() const;
    size_t getKernelSize() const;
    size_t getStride() const;
    size_t getPadding() const;

protected:
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const override;

public:
    using Layer::requireParameters;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
private:
    const ConvolutionalLayerRecipe &recipeConfig() const;
    bool hasWeights() const;
    bool hasBias() const;
};

} // namespace nn
