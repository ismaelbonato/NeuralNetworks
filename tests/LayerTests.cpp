#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"
#include "training/GradientEngine.h"
#include "training/LayerParameterInitializer.h"
#include "training/LearningRule.h"
#include "training/Optimizer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize)
{
    DenseLayerRecipe denseRecipe;
    denseRecipe.name = "test dense layer";
    denseRecipe.type = "DenseLayer";
    denseRecipe.info = "deterministic test layer";
    denseRecipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
    denseRecipe.inputSize = inputSize;
    denseRecipe.outputSize = outputSize;

    return makeLayer<DenseLayer>(denseRecipe);
}

std::unique_ptr<FlattenLayer> makeFlattenLayer(const Shape &inputShape)
{
    FlattenLayerRecipe config;
    config.name = "test flatten layer";
    config.type = "FlattenLayer";
    config.info = "deterministic test layer";
    config.expectedInputShape = inputShape;

    return makeLayer<FlattenLayer>(config);
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}

LearningRuleOptimizer makeSgdOptimizer()
{
    return LearningRuleOptimizer{std::make_shared<SGDRule<Scalar>>()};
}

class UninitializedLayer : public DenseLayer
{
public:
    explicit UninitializedLayer(const DenseLayerRecipe &newRecipe)
        : DenseLayer(newRecipe)
    {}
};

class DelegatingLayer : public Layer
{
public:
    DelegatingLayer()
        : Layer(LayerRecipe{}, {2}, {2})
    {}

    size_t forwardCalls() const
    {
        return calls;
    }

protected:
    Pattern forward(const Pattern &input) const override
    {
        ++calls;
        return input + Pattern{1.0F, 2.0F};
    }

private:
    mutable size_t calls = 0;
};
}

TEST_CASE("base layer infer rejects shape mismatches before delegation",
          "[layer][infer][errors]")
{
    DelegatingLayer layer;

    REQUIRE_THROWS_AS(layer.infer({1.0F}), std::runtime_error);
    REQUIRE(layer.forwardCalls() == 0);
}

TEST_CASE("base layer infer delegates valid input to forward", "[layer][infer]")
{
    DelegatingLayer layer;

    const Pattern output = layer.infer({3.0F, 5.0F});

    REQUIRE(layer.forwardCalls() == 1);
    REQUIRE(output == Pattern{4.0F, 7.0F});
}

TEST_CASE("gradient engine rejects unsupported layer backpropagation",
          "[gradient][errors]")
{
    DelegatingLayer layer;
    const BackpropagationGradientEngine gradientEngine;

    REQUIRE_THROWS_AS(gradientEngine.backwardThroughLayer(layer,
                                                          Pattern{1.0F, 1.0F},
                                                          Pattern{1.0F, 1.0F}),
                      std::runtime_error);
}

TEST_CASE("dense layer adds recipe bias to pre-activation", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 1);
    const Parameters parameters{
        .weights = Pattern::matrix({{1.0F, 1.0F}}),
        .biases = {10.0F},
    };

    const Pattern output = layer->infer({1.0F, 1.0F}, parameters);

    requireClose(output.at(0), 0.9999938F);
}

TEST_CASE("skill parameter snapshots preserve weights and biases", "[skill][dense]")
{
    Skill source(makeDenseLayer(2, 2));
    source.setParameters({
        .weights = Pattern::matrix({{1.0F, 2.0F}, {3.0F, 4.0F}}),
        .biases = {0.5F, -0.5F},
    });

    Skill target(makeDenseLayer(2, 2));
    target.setParameters(source.getParameters());

    REQUIRE(target.getParameters().weights == source.getParameters().weights);
    REQUIRE(target.getParameters().biases == source.getParameters().biases);
}

TEST_CASE("parameter initializer initializes dense layer biases",
          "[layer][dense]")
{
    DenseLayerRecipe config;
    config.name = "bias init layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 2;

    auto skill = makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer
         = std::make_shared<ConstantInitializer<Scalar>>(0.25F)})
                     .intoSkill();

    REQUIRE(skill.getParameters().biases == Pattern{0.25F, 0.25F});
}

TEST_CASE("parameter initializer initializes dense layer parameters",
          "[layer][dense]")
{
    DenseLayerRecipe config;
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 2;

    auto skill = makeTrainableSkill<DenseLayer>(config).intoSkill();

    REQUIRE(skill.getParameters().weights.hasShape({2, 2}));
    REQUIRE(skill.getParameters().biases.shape() == std::vector<size_t>{2});
}

TEST_CASE("layer recipe derives flat sizes from explicit shapes", "[layer][shape]")
{
    DenseLayerRecipe config;
    config.name = "shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "shape-only test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

    auto skill = makeTrainableSkill<DenseLayer>(config).intoSkill();
    const auto &layer = skill.layer();

    REQUIRE(layer.getInputSize() == 2);
    REQUIRE(layer.getOutputSize() == 1);
    REQUIRE(layer.getInputShape().dimensions == std::vector<size_t>{2});
    REQUIRE(layer.getOutputShape().dimensions == std::vector<size_t>{1});
    REQUIRE(skill.getParameters().weights.hasShape({1, 2}));
}

TEST_CASE("layer recipe rejects inconsistent flat size and shape", "[layer][shape][errors]")
{
    DenseLayerRecipe config;
    config.name = "invalid shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "shape mismatch test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 3;
    config.outputSize = 1;
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

    REQUIRE_THROWS_AS(makeLayer<DenseLayer>(config), std::invalid_argument);
}

TEST_CASE("flatten layer reshapes explicit input shape to a vector", "[layer][flatten]")
{
    auto layer = makeFlattenLayer({2, 2});
    auto input = Pattern::withShape({2, 2});
    input.at({0, 0}) = 1.0F;
    input.at({0, 1}) = 2.0F;
    input.at({1, 0}) = 3.0F;
    input.at({1, 1}) = 4.0F;

    const Pattern output = layer->infer(input);

    REQUIRE(output.shape() == std::vector<size_t>{4});
    REQUIRE(output == Pattern{1.0F, 2.0F, 3.0F, 4.0F});
}

TEST_CASE("flatten layer restores previous activation shape during backward pass",
          "[layer][flatten]")
{
    auto layer = makeFlattenLayer({2, 2});
    const Pattern delta = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto previousActivation = Pattern::withShape({2, 2});

    const BackpropagationGradientEngine gradientEngine;
    const Pattern previousDelta = gradientEngine.backwardThroughLayer(*layer,
                                                                      delta,
                                                                      previousActivation);

    REQUIRE(previousDelta.shape() == std::vector<size_t>{2, 2});
    REQUIRE(previousDelta.at({0, 0}) == 1.0F);
    REQUIRE(previousDelta.at({1, 1}) == 4.0F);
}

TEST_CASE("parameter initializer initializes weights using configured scale",
          "[layer][dense]")
{
    DenseLayerRecipe config;
    config.name = "scaled init layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 2;

    auto skill = makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer
         = std::make_shared<UniformInitializer<Scalar>>(-0.25F, 0.25F)})
                     .intoSkill();

    for (const Scalar weight : skill.getParameters().weights) {
        REQUIRE(weight >= -0.25F);
        REQUIRE(weight <= 0.25F);
    }
}

TEST_CASE("optimizer step rejects mismatched activation and delta sizes",
          "[optimizer][errors]")
{
    Skill skill(makeDenseLayer(2, 2));
    skill.setParameters({
        .weights = Pattern::matrix({{0.0F, 0.0F}, {0.0F, 0.0F}}),
        .biases = {0.0F, 0.0F},
    });
    Model network;
    network.addSkill(std::move(skill));
    const auto optimizer = makeSgdOptimizer();

    REQUIRE_THROWS_AS(optimizer.step(network,
                                     {{1.0F}},
                                     {{1.0F, 1.0F}},
                                     0.1F),
                      std::runtime_error);
    REQUIRE_THROWS_AS(optimizer.step(network,
                                     {{1.0F, 1.0F}},
                                     {{1.0F}},
                                     0.1F),
                      std::runtime_error);
}
TEST_CASE("skill parameter setter rejects invalid weight and bias shapes",
          "[skill][errors]")
{
    Skill skill(makeDenseLayer(2, 2));

    REQUIRE_THROWS_AS(skill.setParameters(
                          {.weights = Pattern::vector(4, 0.0F),
                           .biases = {0.0F, 0.0F}}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(skill.setParameters(
                          {.weights = Pattern::matrix(1, 4, 0.0F),
                           .biases = {0.0F, 0.0F}}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(skill.setParameters(
                          {.weights = Pattern::matrix(2, 2, 0.0F),
                           .biases = {0.0F}}),
                      std::runtime_error);

    REQUIRE_NOTHROW(skill.setParameters(
        {.weights = Pattern::matrix(2, 2, 0.0F),
         .biases = {0.0F, 0.0F}}));
}

TEST_CASE("trainable skill factory initializes dense skill", "[skill][dense]")
{
    DenseLayerRecipe denseRecipe;
    denseRecipe.name = "trainable dense skill";
    denseRecipe.type = "DenseLayer";
    denseRecipe.info = "trainable skill factory";
    denseRecipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
    denseRecipe.inputSize = 2;
    denseRecipe.outputSize = 1;

    auto skill = makeTrainableSkill<DenseLayer>(
        denseRecipe,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
                     .intoSkill();

    REQUIRE(skill.hasParameters());
    REQUIRE(skill.getParameters().weights.hasShape({1, 2}));
    REQUIRE(skill.getParameters().biases.hasShape({1}));
    REQUIRE_NOTHROW(skill.requireInitialized());
}

TEST_CASE("skill performs through its runtime layer", "[skill][runtime]")
{
    auto layer = makeDenseLayer(2, 1);
    Skill skill(std::move(layer));
    skill.setParameters({
        .weights = Pattern::matrix({{1.0F, 1.0F}}),
        .biases = {0.0F},
    });

    const Pattern output = skill.perform({1.0F, 1.0F});

    requireClose(output.at(0), 0.880797F);
}

TEST_CASE("skill exposes parameter snapshots for parameterized layers",
          "[skill][parameters]")
{
    auto layer = makeDenseLayer(2, 1);
    Skill skill(std::move(layer));
    const Parameters parameters{
        .weights = Pattern::matrix({{1.0F, -1.0F}}),
        .biases = {0.5F},
    };

    skill.setParameters(parameters);

    REQUIRE(skill.hasParameters());
    REQUIRE(skill.parameters().has_value());
    REQUIRE(skill.getParameters().weights == parameters.weights);
    REQUIRE(skill.getParameters().biases == parameters.biases);
    REQUIRE_NOTHROW(skill.requireInitialized());
}

TEST_CASE("skill-owned parameters drive runtime execution",
          "[skill][parameters]")
{
    auto layer = makeDenseLayer(1, 1);
    Skill skill(std::move(layer));
    skill.setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    const Pattern output = skill.perform({1.0F});

    requireClose(output.at(0), 0.7310586F);
    REQUIRE(skill.getParameters().weights == Pattern::matrix({{2.0F}}));
    REQUIRE(skill.getParameters().biases == Pattern{-1.0F});
}

TEST_CASE("parameterized runtime layer still requires external parameters",
          "[skill][parameters]")
{
    auto layer = makeDenseLayer(1, 1);
    Skill skill(std::move(layer));
    skill.setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });
    const auto &wrappedLayer = dynamic_cast<const DenseLayer &>(skill.layer());

    const Pattern output = skill.perform({1.0F});

    requireClose(output.at(0), 0.7310586F);
    REQUIRE_THROWS_AS(wrappedLayer.infer({1.0F}), std::runtime_error);
}

TEST_CASE("skill initialization uses layer parameter contract",
          "[skill][parameters][initializer]")
{
    DenseLayerRecipe config;
    config.name = "contract initialized skill";
    config.type = "DenseLayer";
    config.info = "skill-owned initialization test";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 1;
    config.outputSize = 1;

    Skill skill(makeLayer<DenseLayer>(config));
    initializeSkillParameters(
        skill,
        {.weightInitializer = std::make_shared<ConstantInitializer<Scalar>>(2.0F),
         .biasInitializer = std::make_shared<ConstantInitializer<Scalar>>(-1.0F)});

    const Pattern output = skill.perform({1.0F});

    REQUIRE_THROWS_AS(skill.layer().infer({1.0F}), std::runtime_error);
    REQUIRE(skill.getParameters().weights == Pattern::matrix({{2.0F}}));
    REQUIRE(skill.getParameters().biases == Pattern{-1.0F});
    requireClose(output.at(0), 0.7310586F);
}

TEST_CASE("runtime-only skills do not expose parameters",
          "[skill][parameters]")
{
    auto layer = makeFlattenLayer({1, 2});
    Skill skill(std::move(layer));

    REQUIRE_FALSE(skill.hasParameters());
    REQUIRE_FALSE(skill.parameters().has_value());
    REQUIRE_NOTHROW(skill.requireInitialized());
    REQUIRE_THROWS_AS(skill.getParameters(), std::runtime_error);
    REQUIRE_THROWS_AS(skill.setParameters({}), std::runtime_error);
}

TEST_CASE("model can infer through added skills", "[model][skill]")
{
    DenseLayerRecipe config;
    config.name = "skill dense layer";
    config.type = "DenseLayer";
    config.info = "skill construction test";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 1;
    config.outputSize = 1;

    auto skill = makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()});
    Skill runtimeSkill = skill.intoSkill();
    runtimeSkill.setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    Model network;
    network.addSkill(std::move(runtimeSkill));

    const Pattern output = network.infer({1.0F});

    REQUIRE(network.numLayers() == 1);
    REQUIRE(&network.getSkill(0).layer() == &network.getLayer(0));
    requireClose(output.at(0), 0.7310586F);
}

TEST_CASE("layer guard rejects derived layers that skip initialization", "[layer][errors]")
{
    DenseLayerRecipe config;
    config.name = "uninitialized test layer";
    config.type = "TestLayer";
    config.info = "intentionally skips construction initialization";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;
    auto layer = std::make_unique<UninitializedLayer>(config);
    Skill skill(std::move(layer));

    REQUIRE_THROWS_AS(skill.requireInitialized(), std::runtime_error);
    REQUIRE_THROWS_AS(skill.perform({1.0F, 1.0F}), std::runtime_error);
    Model network;
    network.addSkill(std::move(skill));
    const auto optimizer = makeSgdOptimizer();
    REQUIRE_THROWS_AS(optimizer.step(network,
                                     {{1.0F, 1.0F}},
                                     {{1.0F}},
                                     0.1F),
                      std::runtime_error);
}
