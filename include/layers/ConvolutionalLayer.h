#pragma once

#include "base/Layer.h"

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer() = delete;
    explicit ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe);
    ~ConvolutionalLayer() override;

    const ConvolutionalLayerRecipe &getConvolutionalRecipe() const;

protected:
    Pattern forward(const Pattern &input) const override;
    Pattern forward(const Pattern &input,
                    const Parameters &parameters) const override;

private:
    ConvolutionalLayerRecipe convolutionalRecipe;

public:
    using Layer::requireParameters;

    bool usesParameters() const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
    bool acceptsParameters(const Parameters &parameters) const override;
    void requireValidParameters(const Parameters &parameters) const override;

private:
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const;
    Pattern activate(const Pattern &values) const;
};
