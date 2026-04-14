#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"
#include "networks/PerceptronNaturalSelection.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>

TEST_CASE("natural selection chooses the candidate with the lowest error", "[perceptron][natural-selection]")
{
    LayerConfig config{
        .learningRule = std::make_shared<PerceptronRule<Scalar>>(),
        .activation = std::make_shared<StepActivation<Scalar>>(),
        .inputSize = 1,
        .outputSize = 1,
        .name = "test natural perceptron",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .useBias = true,
        .initWeights = false,
    };

    auto layer = std::make_shared<DenseLayer>(config);
    PerceptronNaturalSelection network(layer);

    const Patterns candidateOutputs = {{0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}, {1.0F, 0.0F}};
    const Patterns labels = {{0.0F}, {1.0F}};

    REQUIRE(network.findClosestPerceptron(candidateOutputs, labels) == 1);
}
