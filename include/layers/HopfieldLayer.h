#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = delete;
    explicit HopfieldLayer(const HopfieldLayerRecipe &newRecipe);
    ~HopfieldLayer() override;

protected:
    Pattern forward(const Pattern &input) const override;
    Pattern forward(const Pattern &input,
                    const Parameters &parameters) const override;

private:
    bool usesParameters() const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const;
    Pattern activate(const Pattern &values) const;
    Pattern recall(const Pattern &input,
                   const Parameters &parameters) const;
    bool isInitialized(const Parameters &parameters) const override;
    void requireInitialized(const Parameters &parameters) const override;
};
