#pragma once

#include "base/Layer.h"
#include "base/Types.h"

namespace nn {

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
                          const Parameters &parameters) const override;
    Pattern recall(const Pattern &input,
                   const Parameters &parameters) const;
};

} // namespace nn
