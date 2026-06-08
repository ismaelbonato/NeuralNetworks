#pragma once

#include "base/Layer.h"
#include "base/Types.h"

namespace nn {

struct FlattenLayerRecipe : LayerRecipe
{
    virtual void validateRecipe() const override;
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
