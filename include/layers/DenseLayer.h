#pragma once

#include "base/Layer.h"

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete;
    explicit DenseLayer(const DenseLayerRecipe &newRecipe);
    ~DenseLayer() override;

protected:
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const override;

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
};
