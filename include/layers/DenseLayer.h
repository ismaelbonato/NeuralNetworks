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
    Pattern forward(const Pattern &input,
                    const LayerParameters &parameters) const override;

private:
    Pattern weights;
    Pattern biases;

    bool usesParameters() const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input) const;
    Pattern weightedInput(const Pattern &input,
                          const LayerParameters &parameters) const;
    Pattern activate(const Pattern &values) const;
    bool isInitialized(const LayerParameters &parameters) const override;
    void requireInitialized(const LayerParameters &parameters) const override;
};
