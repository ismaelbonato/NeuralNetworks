#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/ConvolutionalLayer.h"

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

    auto layer = std::make_unique<ConvolutionalLayer>(config);
    layer->setParameters({.weights = weights, .biases = {0.0F}});
    net.addLayer(std::move(layer));
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
