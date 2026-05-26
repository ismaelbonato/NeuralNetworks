#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"
#include "nn/model.pb.h"
#include "serialization/ModelSerialization.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <memory>

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

std::filesystem::path modelPath()
{
    return std::filesystem::path{NN_RUNTIME_TEST_OUTPUT_DIR}
           / "nn-runtime-serialization-test.nn";
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
