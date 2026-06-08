#include "serialization/ModelSerialization.h"

#include "nn/model.pb.h"
#include "serialization/LayerSerialization.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace nn::serialization {
namespace {
constexpr uint32_t modelFormatVersion = 1;

nn::proto::Model toProto(const Model &model)
{
    nn::proto::Model protoModel;
    protoModel.set_format_version(modelFormatVersion);

    for (size_t index = 0; index < model.numLayers(); ++index) {
        fillLayer(*protoModel.add_layers(), model.getLayer(index));
    }

    return protoModel;
}

Model modelFromProto(const nn::proto::Model &protoModel)
{
    if (protoModel.format_version() != modelFormatVersion) {
        throw std::runtime_error("Unsupported model format version.");
    }

    Model model;
    for (const auto &protoLayer : protoModel.layers()) {
        model.addLayer(layerFromProto(protoLayer));
    }

    return model;
}

} // namespace

void saveModelToFile(const Model &model, const std::string &path)
{
    const nn::proto::Model protoModel = toProto(model);

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open model file for writing: "
                                 + path);
    }

    if (!protoModel.SerializeToOstream(&output)) {
        throw std::runtime_error("Could not serialize model file: " + path);
    }
}

Model loadModelFromFile(const std::string &path)
{
    nn::proto::Model protoModel;

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open model file for reading: "
                                 + path);
    }

    if (!protoModel.ParseFromIstream(&input)) {
        throw std::runtime_error("Could not parse model file: " + path);
    }

    return modelFromProto(protoModel);
}

} // namespace nn::serialization
