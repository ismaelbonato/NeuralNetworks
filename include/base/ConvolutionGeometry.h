#pragma once

#include "base/Tensor.h"

#include <cstddef>
#include <stdexcept>

struct ConvolutionGeometry
{
    Shape expectedInputShape;
    size_t filterCount = 0;
    size_t kernelHeight = 0;
    size_t kernelWidth = 0;
    size_t strideHeight = 1;
    size_t strideWidth = 1;
    size_t paddingHeight = 0;
    size_t paddingWidth = 0;

    bool isValid() const
    {
        return expectedInputShape.isValid()
               && expectedInputShape.dimensions.size() == 3
               && filterCount > 0
               && kernelHeight > 0
               && kernelWidth > 0
               && strideHeight > 0
               && strideWidth > 0;
    }

    size_t inputChannels() const
    {
        requireValid();
        return expectedInputShape.dimensions.at(0);
    }

    size_t inputHeight() const
    {
        requireValid();
        return expectedInputShape.dimensions.at(1);
    }

    size_t inputWidth() const
    {
        requireValid();
        return expectedInputShape.dimensions.at(2);
    }

    Shape expectedKernelShape() const
    {
        requireValid();
        return {filterCount, inputChannels(), kernelHeight, kernelWidth};
    }

    Shape expectedBiasShape() const
    {
        requireValid();
        return {filterCount};
    }

    Shape expectedOutputShape() const
    {
        requireValid();

        return {filterCount,
                outputDimension(inputHeight(), kernelHeight, paddingHeight, strideHeight),
                outputDimension(inputWidth(), kernelWidth, paddingWidth, strideWidth)};
    }

private:
    void requireValid() const
    {
        if (!isValid()) {
            throw std::runtime_error("Invalid convolution geometry.");
        }
    }

    static size_t outputDimension(const size_t input,
                                  const size_t kernel,
                                  const size_t padding,
                                  const size_t stride)
    {
        const size_t paddedInput = input + (padding * 2);
        if (kernel > paddedInput) {
            throw std::runtime_error("Convolution kernel is larger than padded input.");
        }

        return ((paddedInput - kernel) / stride) + 1;
    }
};
