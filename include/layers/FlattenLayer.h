#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class FlattenLayer : public Layer
{
public:
    FlattenLayer() = delete;
    explicit FlattenLayer(const FlattenLayerRecipe &newRecipe);
    ~FlattenLayer() override;

protected:
    Pattern forward(const Pattern &input) const override;
};
