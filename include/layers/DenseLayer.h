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
                    const Parameters &parameters) const override;

public:
    using Layer::requireInitialized;

    bool usesParameters() const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
    bool isInitialized(const Parameters &parameters) const override;
    void requireInitialized(const Parameters &parameters) const override;

private:
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const;
    Pattern activate(const Pattern &values) const;
};
