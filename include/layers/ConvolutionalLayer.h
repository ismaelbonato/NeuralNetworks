#pragma once

#include "base/Layer.h"

namespace nn {

struct ConvolutionalLayerRecipe : LayerRecipe
{
    void validateRecipe() const override;
    size_t kernelSize = 0;
    size_t stride = 1;
    size_t padding = 0;
};

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer() = delete;
    explicit ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe);
    ~ConvolutionalLayer() override;

private:
    size_t inputChannelsFromShape() const;
    size_t inputLengthFromShape() const;
    size_t outputChannelsFromShape() const;
    size_t outputLengthFromShape() const;
    size_t outputLengthFromGeometry() const;
    size_t getKernelSize() const;
    size_t getStride() const;
    size_t getPadding() const;

protected:
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const override;

public:
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

    LayerSnapshot snapshot() const override;

private:
    const ConvolutionalLayerRecipe &recipeConfig() const;
};

} // namespace nn
