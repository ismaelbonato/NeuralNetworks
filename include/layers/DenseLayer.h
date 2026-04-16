#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete;
    ~DenseLayer() override;

protected:
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
    explicit DenseLayer(const LayerConfig &newConfig);

    template<typename LayerType>
    friend std::shared_ptr<LayerType> makeLayer(const LayerConfig &config);
};
