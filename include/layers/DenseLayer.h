#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class DenseLayer : public TrainableLayer
{
public:
    DenseLayer() = delete;
    ~DenseLayer() override;

protected:
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
    explicit DenseLayer(const DenseLayerConfig &newConfig);

    template<typename LayerType, typename ConfigType>
    friend std::unique_ptr<LayerType> makeLayer(const ConfigType &config);
};
