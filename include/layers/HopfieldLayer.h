#pragma once

#include "base/Layer.h"
#include "base/Types.h"

namespace nn {

struct HopfieldLayerRecipe : LayerRecipe
{
    Shape stateShape;

    Shape getInputShape() const override;
    Shape getOutputShape() const override;
    void validateRecipe() const override;
};

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
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
private:
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const override;
    Pattern recall(const Pattern &input, const Parameters &parameters) const;
};

} // namespace nn
