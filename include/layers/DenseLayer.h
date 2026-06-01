#pragma once

#include "base/Layer.h"

namespace nn {

struct DenseLayerRecipe : LayerRecipe
{
    Shape inputShape;
    Shape outputShape;

    Shape getInputShape() const override;
    Shape getOutputShape() const override;
    void validateRecipe() const override;
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
    using Layer::requireParameters;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
private:
};

} // namespace nn
