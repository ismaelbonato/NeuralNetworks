#pragma once

#include "base/Layer.h"

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer() = delete;
    explicit ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe);
    ~ConvolutionalLayer() override;

    void initializeParameters(Scalar value = Scalar{});
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

private:
    ConvolutionalLayerRecipe convolutionalRecipe;
    std::shared_ptr<Initializer<Scalar>> weightInitializer;
    std::shared_ptr<Initializer<Scalar>> biasInitializer;
    Pattern weights;
    Pattern biases;

    Shape expectedWeightShape() const;
    Shape expectedBiasShape() const;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern preActivation(const Pattern &input) const;
    Pattern activate(const Pattern &values) const;
    Pattern initializeParameter(
        const Shape &shape,
        const std::shared_ptr<Initializer<Scalar>> &initializer,
        Scalar fallbackValue = Scalar{});
};
