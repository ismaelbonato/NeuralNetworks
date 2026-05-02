#pragma once

#include "base/Layer.h"

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete;
    explicit DenseLayer(const DenseLayerRecipe &newRecipe);
    ~DenseLayer() override;

protected:
    Pattern forward(const Pattern &input) const override;
    Pattern forward(const Pattern &input,
                    const LayerParameters &parameters) const override;

private:
    bool usesParameters() const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input,
                          const LayerParameters &parameters) const;
    Pattern activate(const Pattern &values) const;
    bool isInitialized(const LayerParameters &parameters) const override;
    void requireInitialized(const LayerParameters &parameters) const override;
};
