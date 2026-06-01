#include "serialization/ModelSerialization.h"

#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/HopfieldLayer.h"
#include "nn/model.pb.h"

#include <cstdint>
#include <fstream>
#include <memory>
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
        throw std::runtime_error(
            "Cannot serialize a layer without activation.");
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
    protoLayer.set_name(layer.getName());
    protoLayer.set_type(layer.getType().empty() ? fallbackType : layer.getType());
    protoLayer.set_info(layer.getInfo());
    protoLayer.set_activation(activationName(layer));

    if (const auto parameters = layer.parameters()) {
        fillParameters(*protoLayer.mutable_parameters(), *parameters);
    }
}

void fillDenseLayer(nn::proto::Layer &protoLayer, const DenseLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "DenseLayer");

    auto &dense = *protoLayer.mutable_dense();
    dense.set_input_size(static_cast<uint64_t>(layer.getInputShape().elementCount()));
    dense.set_output_size(static_cast<uint64_t>(layer.getOutputShape().elementCount()));
    fillShape(*dense.mutable_expected_input_shape(),
              layer.getInputShape());
    fillShape(*dense.mutable_expected_output_shape(),
              layer.getOutputShape());
}

void fillConvolutionalLayer(nn::proto::Layer &protoLayer,
                            const ConvolutionalLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "ConvolutionalLayer");

    auto &convolutional = *protoLayer.mutable_convolutional();
    convolutional.set_input_channels(
        static_cast<uint64_t>(layer.getInputChannels()));
    convolutional.set_input_length(static_cast<uint64_t>(layer.getInputLength()));
    convolutional.set_output_channels(
        static_cast<uint64_t>(layer.getOutputChannels()));
    convolutional.set_kernel_size(static_cast<uint64_t>(layer.getKernelSize()));
    convolutional.set_stride(static_cast<uint64_t>(layer.getStride()));
    convolutional.set_padding(static_cast<uint64_t>(layer.getPadding()));
}

void fillFlattenLayer(nn::proto::Layer &protoLayer, const FlattenLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "FlattenLayer");

    auto &flatten = *protoLayer.mutable_flatten();
    fillShape(*flatten.mutable_expected_input_shape(),
              layer.getInputShape());
}

void fillHopfieldLayer(nn::proto::Layer &protoLayer, const HopfieldLayer &layer)
{
    fillCommonLayer(protoLayer, layer, "HopfieldLayer");

    auto &hopfield = *protoLayer.mutable_hopfield();
    hopfield.set_size(static_cast<uint64_t>(layer.getInputShape().elementCount()));
    fillShape(*hopfield.mutable_expected_shape(), layer.getInputShape());
}

void fillLayer(nn::proto::Layer &protoLayer, const Layer &layer)
{
    if (const auto *dense = dynamic_cast<const DenseLayer *>(&layer)) {
        fillDenseLayer(protoLayer, *dense);
        return;
    }

    if (const auto *convolutional = dynamic_cast<const ConvolutionalLayer *>(
            &layer)) {
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

Shape shapeFromProto(const google::protobuf::RepeatedField<uint64_t> &protoShape)
{
    std::vector<size_t> dimensions;
    dimensions.reserve(static_cast<size_t>(protoShape.size()));

    for (const uint64_t dimension : protoShape) {
        if (dimension == 0) {
            throw std::runtime_error(
                "Tensor shape dimensions must be greater than zero.");
        }
        dimensions.push_back(static_cast<size_t>(dimension));
    }

    Shape shape{dimensions};
    if (!shape.isValid()) {
        throw std::runtime_error("Tensor shape is invalid.");
    }
    return shape;
}

Pattern tensorFromProto(const nn::proto::Tensor &protoTensor)
{
    const Shape shape = shapeFromProto(protoTensor.shape());
    if (shape.elementCount() != static_cast<size_t>(protoTensor.values_size())) {
        throw std::runtime_error(
            "Tensor value count does not match tensor shape.");
    }

    Pattern tensor(protoTensor.values().begin(), protoTensor.values().end());
    tensor.reshape(shape);
    return tensor;
}

Parameters parametersFromProto(const nn::proto::Parameters &protoParameters)
{
    return {.weights = tensorFromProto(protoParameters.weights()),
            .biases = tensorFromProto(protoParameters.biases())};
}

std::shared_ptr<ActivationFunction<Scalar>> activationFromName(
    const std::string &name)
{
    if (name == "identity") {
        return std::make_shared<IdentityActivation<Scalar>>();
    }
    if (name == "sigmoid") {
        return std::make_shared<SigmoidActivation<Scalar>>();
    }
    if (name == "relu") {
        return std::make_shared<ReLUActivation<Scalar>>();
    }
    if (name == "tanh") {
        return std::make_shared<TanhActivation<Scalar>>();
    }

    throw std::runtime_error("Unsupported activation: " + name);
}

DenseLayerRecipe denseRecipeFromProto(const nn::proto::Layer &protoLayer)
{
    const auto &protoDense = protoLayer.dense();

    DenseLayerRecipe recipe;
    recipe.name = protoLayer.name();
    recipe.type = protoLayer.type().empty() ? "DenseLayer" : protoLayer.type();
    recipe.info = protoLayer.info();
    recipe.activation = activationFromName(protoLayer.activation());

    const auto inputSize = static_cast<size_t>(protoDense.input_size());
    const auto outputSize = static_cast<size_t>(protoDense.output_size());

    recipe.inputShape = protoDense.expected_input_shape_size() > 0
                       ? shapeFromProto(protoDense.expected_input_shape())
                       : Shape{inputSize};
    recipe.outputShape = protoDense.expected_output_shape_size() > 0
                        ? shapeFromProto(protoDense.expected_output_shape())
                        : Shape{outputSize};

    if (inputSize > 0 && recipe.inputShape.elementCount() != inputSize) {
        throw std::runtime_error(
            "Dense input shape does not match serialized input size.");
    }

    if (outputSize > 0 && recipe.outputShape.elementCount() != outputSize) {
        throw std::runtime_error(
            "Dense output shape does not match serialized output size.");
    }

    return recipe;
}

std::unique_ptr<Layer> layerFromProto(const nn::proto::Layer &protoLayer)
{
    if (!protoLayer.has_dense()) {
        throw std::runtime_error(
            "Only dense layer deserialization is supported.");
    }

    auto layer = std::make_unique<DenseLayer>(denseRecipeFromProto(protoLayer));
    if (protoLayer.has_parameters()) {
        layer->setParameters(parametersFromProto(protoLayer.parameters()));
    }

    return layer;
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
