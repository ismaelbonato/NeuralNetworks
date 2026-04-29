#pragma once

#include "base/Layer.h"

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete;
    explicit DenseLayer(const DenseLayerRecipe &newRecipe);
    ~DenseLayer() override;

    const Pattern &getWeights() const;
    const Pattern &getBiases() const;
    LayerParameters getParameters() const;
    void setParameters(const LayerParameters &parameters);
    void setWeights(const Pattern &newWeights);
    void setBiases(const Pattern &newBiases);
    bool isInitialized() const;
    void requireInitialized() const;

protected:
    Pattern forward(const Pattern &input) const override;

private:
    Pattern weights;
    Pattern biases;

    Shape expectedWeightShape() const;
    Shape expectedBiasShape() const;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input) const;
    Pattern activate(const Pattern &values) const;
};
