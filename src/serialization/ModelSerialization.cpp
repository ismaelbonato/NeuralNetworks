#include "serialization/ModelSerialization.h"

#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/HopfieldLayer.h"
#include "nn/model.pb.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace nn::serialization {
namespace {
constexpr uint32_t modelFormatVersion = 1;

void fillShape(nn::proto::Tensor &protoTensor, const Shape &shape)
{
    for (const size_t dimension : shape.dimensions) {
        protoTensor.add_shape(static_cast<uint64_t>(dimension));
    }
}

void fillShape(google::protobuf::RepeatedField<uint64_t> &protoShape,
               const Shape &shape)
{
    for (const size_t dimension : shape.dimensions) {
        protoShape.Add(static_cast<uint64_t>(dimension));
    }
}

void fillTensor(nn::proto::Tensor &protoTensor, const Pattern &tensor)
{
    fillShape(protoTensor, Shape{tensor.shape()});
    for (const Scalar value : tensor) {
        protoTensor.add_values(value);
    }
}

void fillParameters(nn::proto::Parameters &protoParameters,
                    const Parameters &parameters)
{
    fillTensor(*protoParameters.mutable_weights(), parameters.weights);
    fillTensor(*protoParameters.mutable_biases(), parameters.biases);
}

std::string activationName(const Layer &layer)
{
    const auto &activation = layer.getActivation();
    if (!activation) {
        throw std::runtime_error("Cannot serialize a layer without activation.");
    }

    const std::string name{activation->name()};
    if (name.empty() || name == "unknown") {
        throw std::runtime_error("Cannot serialize an unnamed activation.");
    }

    return name;
}

void fillCommonLayer(nn::proto::Layer &protoLayer,
                     const Layer &layer,
                     const std::string &fallbackType)
{
    const LayerRecipe &recipe = layer.getRecipe();
    protoLayer.set_name(recipe.name);
    protoLayer.set_type(recipe.type.empty() ? fallbackType : recipe.type);
    protoLayer.set_info(recipe.info);
    protoLayer.set_activation(activationName(layer));

    if (const auto parameters = layer.parameters()) {
        fillParameters(*protoLayer.mutable_parameters(), *parameters);
    }
}

void fillDenseLayer(nn::proto::Layer &protoLayer, const DenseLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "DenseLayer");

    auto &dense = *protoLayer.mutable_dense();
    dense.set_input_size(static_cast<uint64_t>(layer.getInputSize()));
    dense.set_output_size(static_cast<uint64_t>(layer.getOutputSize()));
    fillShape(*dense.mutable_expected_input_shape(), layer.getExpectedInputShape());
    fillShape(*dense.mutable_expected_output_shape(), layer.getExpectedOutputShape());
}

void fillConvolutionalLayer(nn::proto::Layer &protoLayer,
                            const ConvolutionalLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "ConvolutionalLayer");

    const auto &recipe = layer.getConvolutionalRecipe();
    auto &convolutional = *protoLayer.mutable_convolutional();
    convolutional.set_input_channels(static_cast<uint64_t>(recipe.inputChannels));
    convolutional.set_input_length(static_cast<uint64_t>(recipe.inputLength));
    convolutional.set_output_channels(static_cast<uint64_t>(recipe.outputChannels));
    convolutional.set_kernel_size(static_cast<uint64_t>(recipe.kernelSize));
    convolutional.set_stride(static_cast<uint64_t>(recipe.stride));
    convolutional.set_padding(static_cast<uint64_t>(recipe.padding));
}

void fillFlattenLayer(nn::proto::Layer &protoLayer, const FlattenLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "FlattenLayer");

    auto &flatten = *protoLayer.mutable_flatten();
    fillShape(*flatten.mutable_expected_input_shape(),
              layer.getExpectedInputShape());
}

void fillHopfieldLayer(nn::proto::Layer &protoLayer, const HopfieldLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "HopfieldLayer");

    auto &hopfield = *protoLayer.mutable_hopfield();
    hopfield.set_size(static_cast<uint64_t>(layer.getInputSize()));
    fillShape(*hopfield.mutable_expected_shape(), layer.getExpectedInputShape());
}

void fillLayer(nn::proto::Layer &protoLayer, const Layer &layer)
{
    if (const auto *dense = dynamic_cast<const DenseLayer *>(&layer)) {
        fillDenseLayer(protoLayer, *dense);
        return;
    }

    if (const auto *convolutional
        = dynamic_cast<const ConvolutionalLayer *>(&layer)) {
        fillConvolutionalLayer(protoLayer, *convolutional);
        return;
    }

    if (const auto *flatten = dynamic_cast<const FlattenLayer *>(&layer)) {
        fillFlattenLayer(protoLayer, *flatten);
        return;
    }

    if (const auto *hopfield = dynamic_cast<const HopfieldLayer *>(&layer)) {
        fillHopfieldLayer(protoLayer, *hopfield);
        return;
    }

    throw std::runtime_error("Cannot serialize unsupported layer type.");
}

nn::proto::Model toProto(const Model &model)
{
    nn::proto::Model protoModel;
    protoModel.set_format_version(modelFormatVersion);

    for (size_t index = 0; index < model.numLayers(); ++index) {
        fillLayer(*protoModel.add_layers(), model.getLayer(index));
    }

    return protoModel;
}

} // namespace

void saveModelToFile(const Model &model, const std::string &path)
{
    const nn::proto::Model protoModel = toProto(model);

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open model file for writing: " + path);
    }

    if (!protoModel.SerializeToOstream(&output)) {
        throw std::runtime_error("Could not serialize model file: " + path);
    }
}

} // namespace nn::serialization
