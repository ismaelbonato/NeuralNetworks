#pragma once

#include "base/Layer.h"
#include "base/Types.h"

namespace nn {

struct HopfieldLayerRecipe : LayerRecipe
{
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
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
    Pattern weightedInput(const Pattern &input,
                          const Parameters &parameters) const override;
    Pattern recall(const Pattern &input, const Parameters &parameters) const;
};

} // namespace nn
