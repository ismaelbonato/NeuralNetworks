#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "base/Model.h"
#include "layers/ConvolutionalLayer.h"
#include "training/FeedforwardTrainer.h"
#include "training/GradientEngine.h"
#include "training/LayerParameterInitializer.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <utility>

TEST_CASE("valid 1D convolution slides a kernel over a simple signal",
          "[convolution][1d]")
{
    Pattern signal
        = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    signal.reshape({1, signal.size()});

    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    ConvolutionalLayerRecipe config{};
    config.inputChannels = 1;
    config.inputLength = signal.size();
    config.outputChannels = 1;
    config.kernelSize = 3;
    config.stride = 1;
    config.padding = 0;
    config.name = "test convolutional layer";
    config.type = "ConvolutionalLayer";
    config.info = "deterministic test layer";
    config.activation = activation;

    const size_t outputLength = config.inputLength - config.kernelSize + 1;

    Model net;

    Pattern weights = Pattern::withShape({1, 1, 3});
    weights.at({0, 0, 0}) = -1.0F;
    weights.at({0, 0, 1}) = 0.0F;
    weights.at({0, 0, 2}) = 1.0F;

    auto skill = makeTrainableSkill<ConvolutionalLayer>(config).intoSkill();
    skill.setParameters({.weights = weights, .biases = {0.0F}});
    net.addSkill(std::move(skill));
    Pattern output = net.infer(signal);

    Pattern expected = {0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F};
    expected.reshape({1, outputLength});

    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output.at(i) == Catch::Approx(expected.at(i)).epsilon(0.001F));
    }
}

TEST_CASE("1D convolution backward spreads output deltas over input windows",
          "[convolution][1d][backward]")
{
    auto activation = std::make_shared<IdentityActivation<Scalar>>();

    ConvolutionalLayerRecipe config{};
    config.inputChannels = 1;
    config.inputLength = 4;
    config.outputChannels = 1;
    config.kernelSize = 2;
    config.stride = 1;
    config.padding = 0;
    config.name = "test convolutional layer";
    config.type = "ConvolutionalLayer";
    config.info = "deterministic backward test layer";
    config.activation = activation;

    Pattern weights = Pattern::withShape({1, 1, 2});
    weights.at({0, 0, 0}) = 2.0F;
    weights.at({0, 0, 1}) = 3.0F;

    auto layer = makeLayer<ConvolutionalLayer>(config);
    layer->setWeights(weights);
    layer->setBiases({0.0F});

    Pattern layerDelta = Pattern::withShape({1, 3});
    layerDelta.at({0, 0}) = 5.0F;
    layerDelta.at({0, 1}) = 7.0F;
    layerDelta.at({0, 2}) = 11.0F;

    const Pattern layerInput = Pattern::withShape({1, 4}, Scalar{0});
    const BackpropagationGradientEngine gradientEngine;
    const Pattern previousDelta = gradientEngine.backwardThroughLayer(*layer,
                                                                      layerDelta,
                                                                      layerInput);

    REQUIRE(previousDelta.at({0, 0}) == Catch::Approx(10.0F));
    REQUIRE(previousDelta.at({0, 1}) == Catch::Approx(29.0F));
    REQUIRE(previousDelta.at({0, 2}) == Catch::Approx(43.0F));
    REQUIRE(previousDelta.at({0, 3}) == Catch::Approx(33.0F));
}

TEST_CASE("training 1D convolution", "[convolution][1d]")
{
    Pattern signal
        = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    signal.reshape({1, signal.size()});

    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    ConvolutionalLayerRecipe config{};
    config.inputChannels = 1;
    config.inputLength = signal.size();
    config.outputChannels = 1;
    config.kernelSize = 3;
    config.stride = 1;
    config.padding = 0;
    config.name = "test convolutional layer";
    config.type = "ConvolutionalLayer";
    config.info = "deterministic test layer";
    config.activation = activation;

    const size_t outputLength = config.inputLength - config.kernelSize + 1;

    Model net;

    Pattern weights = Pattern::withShape({1, 1, 3});
    weights.at({0, 0, 0}) = -1.0F;
    weights.at({0, 0, 1}) = 0.0F;
    weights.at({0, 0, 2}) = 1.0F;

    net.addSkill(makeTrainableSkill<ConvolutionalLayer>(config).intoSkill());

    FeedforwardTrainer trainer;

    Pattern expected = {0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F,
                        0.880797F};
    expected.reshape({1, outputLength});

    trainer.learn(net, {signal}, {expected}, Scalar{0.1F}, 10000);

    Pattern output = net.infer(signal);

    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output.at(i) == Catch::Approx(expected.at(i)).epsilon(0.001F));
    }
}
