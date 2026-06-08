#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "nn/model.pb.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using namespace nn;

namespace {
std::unique_ptr<DenseLayer> makeDenseLayer()
{
    DenseLayerRecipe recipe;
    recipe.name = "serialize dense";
    recipe.type = "DenseLayer";
    recipe.info = "serialization fixture";
    recipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
    recipe.inputShape = {2};
    recipe.outputShape = {1};

    auto layer = std::make_unique<DenseLayer>(recipe);
    layer->setParameters({
        .weights = Pattern::matrix({{1.5F, -2.0F}}),
        .biases = {0.25F},
    });
    return layer;
}

std::unique_ptr<ConvolutionalLayer> makeConvolutionalLayer()
{
    ConvolutionalLayerRecipe recipe;
    recipe.name = "serialize convolution";
    recipe.type = "ConvolutionalLayer";
    recipe.info = "serialization fixture";
    recipe.activation = std::make_shared<IdentityActivation<Scalar>>();
    recipe.inputShape = {1, 4};
    recipe.outputShape = {1, 2};
    recipe.kernelSize = 3;
    recipe.stride = 1;
    recipe.padding = 0;

    Pattern weights = Pattern::withShape({1, 1, 3});
    weights.at({0, 0, 0}) = -1.0F;
    weights.at({0, 0, 1}) = 0.0F;
    weights.at({0, 0, 2}) = 1.0F;

    auto layer = std::make_unique<ConvolutionalLayer>(recipe);
    layer->setParameters({.weights = weights, .biases = {0.5F}});
    return layer;
}

std::filesystem::path modelPath(const std::string &filename)
{
    return std::filesystem::path{NN_RUNTIME_TEST_OUTPUT_DIR} / filename;
}

std::filesystem::path modelPath()
{
    return modelPath("nn-runtime-serialization-test.nn");
}

void writeProtoModel(const nn::proto::Model &protoModel,
                     const std::filesystem::path &path)
{
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output.good());
    REQUIRE(protoModel.SerializeToOstream(&output));
}

nn::proto::Layer &addDenseProtoLayer(nn::proto::Model &protoModel)
{
    auto &layer = *protoModel.add_layers();
    auto &recipe = *layer.mutable_recipe();
    recipe.set_name("dense fixture");
    recipe.set_type("DenseLayer");
    recipe.set_info("load failure fixture");
    recipe.set_activation("sigmoid");
    recipe.add_input_shape(2);
    recipe.add_output_shape(1);
    recipe.mutable_dense();

    auto &weights = *layer.mutable_parameters()->mutable_weights();
    weights.add_shape(1);
    weights.add_shape(2);
    weights.add_values(1.5F);
    weights.add_values(-2.0F);

    auto &biases = *layer.mutable_parameters()->mutable_biases();
    biases.add_shape(1);
    biases.add_values(0.25F);

    return layer;
}
} // namespace

TEST_CASE("model serializer creates a protobuf file", "[serialization]")
{
    const std::filesystem::path path = modelPath();
    std::filesystem::remove(path);

    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    REQUIRE(std::filesystem::exists(path));
}

TEST_CASE("model serializer writes the format version", "[serialization]")
{
    const std::filesystem::path path = modelPath();
    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    REQUIRE(protoModel.format_version() == 1);
}

TEST_CASE("model serializer writes one proto layer per model layer",
          "[serialization]")
{
    const std::filesystem::path path = modelPath();
    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    REQUIRE(protoModel.layers_size() == 1);
}

TEST_CASE("dense serialization writes recipe metadata", "[serialization]")
{
    const std::filesystem::path path = modelPath();
    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    const auto &recipe = protoModel.layers(0).recipe();
    REQUIRE(recipe.has_dense());
    REQUIRE(recipe.name() == "serialize dense");
    REQUIRE(recipe.type() == "DenseLayer");
    REQUIRE(recipe.info() == "serialization fixture");
    REQUIRE(recipe.activation() == "sigmoid");
}

TEST_CASE("dense serialization writes recipe input and output shapes",
          "[serialization]")
{
    const std::filesystem::path path = modelPath();
    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    const auto &recipe = protoModel.layers(0).recipe();
    REQUIRE(recipe.input_shape_size() == 1);
    REQUIRE(recipe.input_shape(0) == 2);
    REQUIRE(recipe.output_shape_size() == 1);
    REQUIRE(recipe.output_shape(0) == 1);
}

TEST_CASE("dense serialization writes weight tensor", "[serialization]")
{
    const std::filesystem::path path = modelPath();
    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    const auto &weights = protoModel.layers(0).parameters().weights();
    REQUIRE(weights.shape_size() == 2);
    REQUIRE(weights.shape(0) == 1);
    REQUIRE(weights.shape(1) == 2);
    REQUIRE(weights.values_size() == 2);
    REQUIRE(weights.values(0) == 1.5F);
    REQUIRE(weights.values(1) == -2.0F);
}

TEST_CASE("dense serialization writes bias tensor", "[serialization]")
{
    const std::filesystem::path path = modelPath();
    Model model;
    model.addLayer(makeDenseLayer());

    model.saveToFile(path.string());

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    const auto &biases = protoModel.layers(0).parameters().biases();
    REQUIRE(biases.shape_size() == 1);
    REQUIRE(biases.shape(0) == 1);
    REQUIRE(biases.values_size() == 1);
    REQUIRE(biases.values(0) == 0.25F);
}

TEST_CASE("model serializer loads saved dense model", "[serialization]")
{
    const std::filesystem::path path = modelPath("nn-runtime-save-load-test.nn");
    std::filesystem::remove(path);

    Model model;
    model.addLayer(makeDenseLayer());
    const Pattern before = model.infer({2.0F, -1.0F});

    model.saveToFile(path.string());
    Model loaded = Model::loadFromFile(path.string());
    const Pattern after = loaded.infer({2.0F, -1.0F});

    REQUIRE(loaded.numLayers() == 1);
    REQUIRE(after.size() == before.size());
    REQUIRE(after.at(0) == Catch::Approx(before.at(0)));
}

TEST_CASE("model serializer loads saved convolutional model", "[serialization]")
{
    const std::filesystem::path path = modelPath(
        "nn-runtime-convolution-save-load-test.nn");
    std::filesystem::remove(path);

    Pattern input = {0.0F, 1.0F, 2.0F, 3.0F};
    input.reshape({1, 4});

    Model model;
    model.addLayer(makeConvolutionalLayer());
    const Pattern before = model.infer(input);

    model.saveToFile(path.string());
    Model loaded = Model::loadFromFile(path.string());
    const Pattern after = loaded.infer(input);

    REQUIRE(after.shape() == before.shape());
    REQUIRE(after.at(0) == Catch::Approx(before.at(0)));
    REQUIRE(after.at(1) == Catch::Approx(before.at(1)));
}

TEST_CASE("model serializer rejects missing model files", "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("missing-model.nn");
    std::filesystem::remove(path);

    REQUIRE_THROWS_AS(Model::loadFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects malformed model files", "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("malformed-model.nn");
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output.good());
    output << "not a protobuf model";
    output.close();

    REQUIRE_THROWS_AS(Model::loadFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects unsupported format versions",
          "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("unsupported-version.nn");
    nn::proto::Model protoModel;
    protoModel.set_format_version(999);
    addDenseProtoLayer(protoModel);
    writeProtoModel(protoModel, path);

    REQUIRE_THROWS_AS(Model::loadFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects unsupported activation names",
          "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("unsupported-activation.nn");
    nn::proto::Model protoModel;
    protoModel.set_format_version(1);
    addDenseProtoLayer(protoModel).mutable_recipe()->set_activation("does_not_exist");
    writeProtoModel(protoModel, path);

    REQUIRE_THROWS_AS(Model::loadFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects invalid tensor value counts",
          "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("invalid-tensor.nn");
    nn::proto::Model protoModel;
    protoModel.set_format_version(1);
    auto &layer = addDenseProtoLayer(protoModel);
    layer.mutable_parameters()->mutable_weights()->clear_values();
    layer.mutable_parameters()->mutable_weights()->add_values(1.0F);
    writeProtoModel(protoModel, path);

    REQUIRE_THROWS_AS(Model::loadFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects unsupported layer kinds",
          "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("unsupported-layer.nn");
    nn::proto::Model protoModel;
    protoModel.set_format_version(1);

    auto &layer = *protoModel.add_layers();
    auto &recipe = *layer.mutable_recipe();
    recipe.set_name("flatten fixture");
    recipe.set_activation("identity");
    recipe.add_input_shape(2);
    recipe.add_output_shape(2);
    recipe.mutable_flatten();
    writeProtoModel(protoModel, path);

    REQUIRE_THROWS_AS(Model::loadFromFile(path.string()),
                      std::runtime_error);
}
