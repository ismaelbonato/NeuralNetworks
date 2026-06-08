#include "base/ActivationFunction.h"
#include "base/Types.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace nn;

namespace
{
constexpr Scalar tolerance = 0.0001F;

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}
}

TEST_CASE("sigmoid activation returns expected value and derivative",
          "[activation]")
{
    SigmoidActivation<Scalar> sigmoid;

    requireClose(sigmoid(0.0F), 0.5F);
    requireClose(sigmoid.derivative(0.0F), 0.25F);
}

TEST_CASE("step activation returns expected values", "[activation]")
{
    StepActivation<Scalar> step;

    REQUIRE(step(-1.0F) == 0.0F);
    REQUIRE(step(0.0F) == 1.0F);
}

TEST_CASE("step activation derivative is zero", "[activation]")
{
    StepActivation<Scalar> step;

    REQUIRE(step.derivative(2.0F) == 0.0F);
}

TEST_CASE("polar step activation returns expected values", "[activation]")
{
    StepPolarActivation<Scalar> polarStep;

    REQUIRE(polarStep(-1.0F) == -1.0F);
    REQUIRE(polarStep(0.0F) == 1.0F);
}

TEST_CASE("polar step activation derivative is zero", "[activation]")
{
    StepPolarActivation<Scalar> polarStep;

    REQUIRE(polarStep.derivative(2.0F) == 0.0F);
}

TEST_CASE("relu activation returns expected values", "[activation]")
{
    ReLUActivation<Scalar> relu;

    REQUIRE(relu(-1.0F) == 0.0F);
    REQUIRE(relu(2.0F) == 2.0F);
}

TEST_CASE("relu activation returns expected derivatives", "[activation]")
{
    ReLUActivation<Scalar> relu;

    REQUIRE(relu.derivative(-1.0F) == 0.0F);
    REQUIRE(relu.derivative(2.0F) == 1.0F);
}

TEST_CASE("tanh activation returns expected value and derivative",
          "[activation]")
{
    TanhActivation<Scalar> tanh;

    requireClose(tanh(0.0F), 0.0F);
    requireClose(tanh.derivative(0.0F), 1.0F);
}
