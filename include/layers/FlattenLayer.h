#pragma once

#include "base/Layer.h"
#include "base/Types.h"

namespace nn {

struct FlattenLayerRecipe : LayerRecipe
{
    Shape inputShape;

    Shape getInputShape() const override;
    Shape getOutputShape() const override;
};

class FlattenLayer : public Layer
{
public:
    FlattenLayer() = delete;
    explicit FlattenLayer(const FlattenLayerRecipe &newRecipe);
    ~FlattenLayer() override;

protected:
    Pattern forward(const Pattern &input) const override;
};

} // namespace nn
