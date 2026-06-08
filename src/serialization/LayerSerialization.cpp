#include "serialization/LayerSerialization.h"

#include "base/ActivationFunction.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn::serialization {
namespace {

void fillShape(google::protobuf::RepeatedField<uint64_t> &protoShape,
               const std::vector<size_t> &shape)
{
    protoShape.Clear();
    for (const size_t dimension : shape) {
        protoShape.Add(static_cast<uint64_t>(dimension));
    }
}

size_t fieldAsSize(const LayerSnapshot &snapshot, const std::string &fieldName)
{
    for (const auto &[name, value] : snapshot.fields) {
        if (name == fieldName && std::holds_alternative<size_t>(value)) {
            return std::get<size_t>(value);
        }
    }

    throw std::runtime_error("Missing layer snapshot field: " + fieldName);
}

void fillTensor(nn::proto::Tensor &protoTensor, const Pattern &tensor)
{
    fillShape(*protoTensor.mutable_shape(), tensor.shape());
    protoTensor.clear_values();
    for (const Scalar value : tensor) {
        protoTensor.add_values(value);
    }
}

} // namespace

void fillLayer(nn::proto::Layer &protoLayer, const Layer &layer)
{
    const LayerSnapshot snapshot = layer.snapshot();
    auto &protoRecipe = *protoLayer.mutable_recipe();

    protoRecipe.set_name(snapshot.name);
    protoRecipe.set_type(snapshot.type);
    protoRecipe.set_info(snapshot.info);
    protoRecipe.set_activation(snapshot.activation);
    fillShape(*protoRecipe.mutable_input_shape(), snapshot.inputShape);
    fillShape(*protoRecipe.mutable_output_shape(), snapshot.outputShape);

    if (snapshot.type == "DenseLayer") {
        protoRecipe.mutable_dense();
    } else if (snapshot.type == "FlattenLayer") {
        protoRecipe.mutable_flatten();
    } else if (snapshot.type == "HopfieldLayer") {
        protoRecipe.mutable_hopfield();
    } else if (snapshot.type == "ConvolutionalLayer") {
        auto &convolutional = *protoRecipe.mutable_convolutional();
        convolutional.set_kernel_size(fieldAsSize(snapshot, "kernelSize"));
        convolutional.set_stride(fieldAsSize(snapshot, "stride"));
        convolutional.set_padding(fieldAsSize(snapshot, "padding"));
    }

    if (snapshot.parameters) {
        auto &protoParameters = *protoLayer.mutable_parameters();
        fillTensor(*protoParameters.mutable_weights(),
                   snapshot.parameters->weights);
        fillTensor(*protoParameters.mutable_biases(),
                   snapshot.parameters->biases);
    }
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

    throw std::runtime_error("Unsupported activation: " + name);
}

Shape shapeFromProto(const google::protobuf::RepeatedField<uint64_t> &protoShape)
{
    std::vector<size_t> dimensions;
    dimensions.reserve(static_cast<size_t>(protoShape.size()));
    for (const uint64_t dimension : protoShape) {
        dimensions.push_back(static_cast<size_t>(dimension));
    }
    return Shape{dimensions};
}

Pattern tensorFromProto(const nn::proto::Tensor &protoTensor)
{
    Pattern tensor(protoTensor.values().begin(), protoTensor.values().end());
    const Shape shape = shapeFromProto(protoTensor.shape());
    if (shape.elementCount() != tensor.size()) {
        throw std::runtime_error("Tensor shape does not match value count.");
    }
    tensor.reshape(shape);
    return tensor;
}

template<typename Recipe>
void fillCommonRecipeFields(Recipe &recipe,
                            const nn::proto::LayerRecipe &protoRecipe)
{
    recipe.name = protoRecipe.name();
    recipe.type = protoRecipe.type();
    recipe.info = protoRecipe.info();
    recipe.activation = activationFromName(protoRecipe.activation());
    recipe.inputShape = shapeFromProto(protoRecipe.input_shape());
    recipe.outputShape = shapeFromProto(protoRecipe.output_shape());
}

void assignParameters(Layer &layer, const nn::proto::Layer &protoLayer)
{
    if (!protoLayer.has_parameters()) {
        return;
    }

    layer.setParameters({
        .weights = tensorFromProto(protoLayer.parameters().weights()),
        .biases = tensorFromProto(protoLayer.parameters().biases()),
    });
}

std::unique_ptr<Layer> layerFromProto(const nn::proto::Layer &protoLayer)
{
    const auto &protoRecipe = protoLayer.recipe();

    if (protoRecipe.has_dense()) {
        DenseLayerRecipe recipe;
        fillCommonRecipeFields(recipe, protoRecipe);

        auto layer = std::make_unique<DenseLayer>(recipe);
        assignParameters(*layer, protoLayer);
        return layer;
    }

    if (protoRecipe.has_convolutional()) {
        ConvolutionalLayerRecipe recipe;
        fillCommonRecipeFields(recipe, protoRecipe);
        const auto &convolutional = protoRecipe.convolutional();
        recipe.kernelSize = static_cast<size_t>(convolutional.kernel_size());
        recipe.stride = static_cast<size_t>(convolutional.stride());
        recipe.padding = static_cast<size_t>(convolutional.padding());

        auto layer = std::make_unique<ConvolutionalLayer>(recipe);
        assignParameters(*layer, protoLayer);
        return layer;
    }

    throw std::runtime_error("Unsupported layer kind.");
}

} // namespace nn::serialization
