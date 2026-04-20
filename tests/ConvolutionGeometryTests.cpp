#include "base/ConvolutionGeometry.h"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

TEST_CASE("convolution geometry computes valid output and parameter shapes",
          "[convolution][shape]")
{
    ConvolutionGeometry geometry{
        .expectedInputShape = {3, 28, 28},
        .filterCount = 16,
        .kernelHeight = 5,
        .kernelWidth = 5,
        .strideHeight = 1,
        .strideWidth = 1,
        .paddingHeight = 0,
        .paddingWidth = 0,
    };

    REQUIRE(geometry.inputChannels() == 3);
    REQUIRE(geometry.expectedKernelShape().dimensions == std::vector<size_t>{16, 3, 5, 5});
    REQUIRE(geometry.expectedBiasShape().dimensions == std::vector<size_t>{16});
    REQUIRE(geometry.expectedOutputShape().dimensions == std::vector<size_t>{16, 24, 24});
}

TEST_CASE("convolution geometry accounts for padding and stride",
          "[convolution][shape]")
{
    ConvolutionGeometry geometry{
        .expectedInputShape = {1, 7, 7},
        .filterCount = 4,
        .kernelHeight = 3,
        .kernelWidth = 3,
        .strideHeight = 2,
        .strideWidth = 2,
        .paddingHeight = 1,
        .paddingWidth = 1,
    };

    REQUIRE(geometry.expectedOutputShape().dimensions == std::vector<size_t>{4, 4, 4});
}

TEST_CASE("convolution geometry rejects invalid contracts", "[convolution][shape][errors]")
{
    ConvolutionGeometry missingChannels{
        .expectedInputShape = {28, 28},
        .filterCount = 4,
        .kernelHeight = 3,
        .kernelWidth = 3,
    };

    ConvolutionGeometry zeroStride{
        .expectedInputShape = {1, 7, 7},
        .filterCount = 4,
        .kernelHeight = 3,
        .kernelWidth = 3,
        .strideHeight = 0,
        .strideWidth = 1,
    };

    ConvolutionGeometry tooLargeKernel{
        .expectedInputShape = {1, 2, 2},
        .filterCount = 4,
        .kernelHeight = 3,
        .kernelWidth = 3,
    };

    REQUIRE_FALSE(missingChannels.isValid());
    REQUIRE_THROWS_AS(missingChannels.expectedOutputShape(), std::runtime_error);
    REQUIRE_FALSE(zeroStride.isValid());
    REQUIRE_THROWS_AS(zeroStride.expectedOutputShape(), std::runtime_error);
    REQUIRE_THROWS_AS(tooLargeKernel.expectedOutputShape(), std::runtime_error);
}
