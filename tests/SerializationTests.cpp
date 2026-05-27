#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"
#include "nn/model.pb.h"
#include "serialization/ModelSerialization.h"

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
    recipe.inputSize = 2;
    recipe.outputSize = 1;

    auto layer = std::make_unique<DenseLayer>(recipe);
    layer->setParameters({
        .weights = Pattern::matrix({{1.5F, -2.0F}}),
        .biases = {0.25F},
    });
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
    layer.set_name("dense fixture");
    layer.set_type("DenseLayer");
    layer.set_info("load failure fixture");
    layer.set_activation("sigmoid");

    auto &dense = *layer.mutable_dense();
    dense.set_input_size(2);
    dense.set_output_size(1);
    dense.add_expected_input_shape(2);
    dense.add_expected_output_shape(1);

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

TEST_CASE("model serializer saves dense model as protobuf binary",
          "[serialization]")
{
    const std::filesystem::path path = modelPath();
    std::filesystem::remove(path);

    Model model;
    model.addLayer(makeDenseLayer());

    serialization::saveModelToFile(model, path.string());

    REQUIRE(std::filesystem::exists(path));

    nn::proto::Model protoModel;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    REQUIRE(protoModel.ParseFromIstream(&input));

    REQUIRE(protoModel.format_version() == 1);
    REQUIRE(protoModel.layers_size() == 1);

    const auto &layer = protoModel.layers(0);
    REQUIRE(layer.has_dense());
    REQUIRE(layer.name() == "serialize dense");
    REQUIRE(layer.type() == "DenseLayer");
    REQUIRE(layer.info() == "serialization fixture");
    REQUIRE(layer.activation() == "sigmoid");

    REQUIRE(layer.dense().input_size() == 2);
    REQUIRE(layer.dense().output_size() == 1);
    REQUIRE(layer.dense().expected_input_shape_size() == 1);
    REQUIRE(layer.dense().expected_input_shape(0) == 2);
    REQUIRE(layer.dense().expected_output_shape_size() == 1);
    REQUIRE(layer.dense().expected_output_shape(0) == 1);

    REQUIRE(layer.parameters().weights().shape_size() == 2);
    REQUIRE(layer.parameters().weights().shape(0) == 1);
    REQUIRE(layer.parameters().weights().shape(1) == 2);
    REQUIRE(layer.parameters().weights().values_size() == 2);
    REQUIRE(layer.parameters().weights().values(0) == 1.5F);
    REQUIRE(layer.parameters().weights().values(1) == -2.0F);

    REQUIRE(layer.parameters().biases().shape_size() == 1);
    REQUIRE(layer.parameters().biases().shape(0) == 1);
    REQUIRE(layer.parameters().biases().values_size() == 1);
    REQUIRE(layer.parameters().biases().values(0) == 0.25F);

    // Keep the file in the build tree so it can be inspected after the test.
}


TEST_CASE("model serializer loads saved dense model", "[serialization]")
{
    const std::filesystem::path path = modelPath("nn-runtime-save-load-test.nn");
    std::filesystem::remove(path);

    Model model;
    model.addLayer(makeDenseLayer());
    const Pattern before = model.infer({2.0F, -1.0F});

    serialization::saveModelToFile(model, path.string());
    Model loaded = serialization::loadModelFromFile(path.string());
    const Pattern after = loaded.infer({2.0F, -1.0F});

    REQUIRE(loaded.numLayers() == 1);
    REQUIRE(after.size() == before.size());
    REQUIRE(after.at(0) == Catch::Approx(before.at(0)));
}

TEST_CASE("model serializer rejects missing model files", "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("missing-model.nn");
    std::filesystem::remove(path);

    REQUIRE_THROWS_AS(serialization::loadModelFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects malformed model files", "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("malformed-model.nn");
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output.good());
    output << "not a protobuf model";
    output.close();

    REQUIRE_THROWS_AS(serialization::loadModelFromFile(path.string()),
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

    REQUIRE_THROWS_AS(serialization::loadModelFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects unsupported activation names",
          "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("unsupported-activation.nn");
    nn::proto::Model protoModel;
    protoModel.set_format_version(1);
    addDenseProtoLayer(protoModel).set_activation("does_not_exist");
    writeProtoModel(protoModel, path);

    REQUIRE_THROWS_AS(serialization::loadModelFromFile(path.string()),
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

    REQUIRE_THROWS_AS(serialization::loadModelFromFile(path.string()),
                      std::runtime_error);
}

TEST_CASE("model serializer rejects unsupported layer kinds",
          "[serialization][errors]")
{
    const std::filesystem::path path = modelPath("unsupported-layer.nn");
    nn::proto::Model protoModel;
    protoModel.set_format_version(1);

    auto &layer = *protoModel.add_layers();
    layer.set_name("flatten fixture");
    layer.set_activation("identity");
    layer.mutable_flatten()->add_expected_input_shape(2);
    writeProtoModel(protoModel, path);

    REQUIRE_THROWS_AS(serialization::loadModelFromFile(path.string()),
                      std::runtime_error);
}
