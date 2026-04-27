#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class FlattenLayer : public Layer
{
public:
    FlattenLayer() = delete;
    ~FlattenLayer() override;

    Pattern backwardPass(const Pattern &layerDelta, const Pattern &preActivation) const;

protected:
    Pattern forward(const Pattern &input) const override;

    explicit FlattenLayer(const FlattenLayerConfig &newConfig);

private:
    template<typename LayerType, typename ConfigType>
    friend std::unique_ptr<LayerType> makeLayer(const ConfigType &config);
};
