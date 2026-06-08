#pragma once

#include "base/Layer.h"
#include "nn/model.pb.h"

#include <memory>

namespace nn::serialization {

void fillLayer(nn::proto::Layer &protoLayer, const Layer &layer);
std::unique_ptr<Layer> layerFromProto(const nn::proto::Layer &protoLayer);

} // namespace nn::serialization
