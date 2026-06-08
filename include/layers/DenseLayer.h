#pragma once

#include "base/Layer.h"

namespace nn {

struct DenseLayerRecipe : LayerRecipe
{
    virtual void validateRecipe() const override;
};

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
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
};

} // namespace nn
