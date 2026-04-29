#pragma once

#include "base/Layer.h"

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer() = delete;
    explicit ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe);
    ~ConvolutionalLayer() override;

    const Pattern &getWeights() const;
    const Pattern &getBiases() const;
    LayerParameters getParameters() const;
    void setParameters(const LayerParameters &parameters);
    void setWeights(const Pattern &newWeights);
    void setBiases(const Pattern &newBiases);
    bool isInitialized() const;
    void requireInitialized() const;

    const ConvolutionalLayerRecipe &getConvolutionalRecipe() const;

protected:
    Pattern forward(const Pattern &input) const override;
    Pattern forward(const Pattern &input,
                    const LayerParameters &parameters) const override;

private:
    ConvolutionalLayerRecipe convolutionalRecipe;
    Pattern weights;
    Pattern biases;

    Shape expectedWeightShape() const;
    Shape expectedBiasShape() const;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input) const;
    Pattern weightedInput(const Pattern &input,
                          const LayerParameters &parameters) const;
    Pattern activate(const Pattern &values) const;
    bool isInitialized(const LayerParameters &parameters) const;
    void requireInitialized(const LayerParameters &parameters) const;
};
