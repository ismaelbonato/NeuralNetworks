#include "base/ActivationFunction.h"
#include "base/Types.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

namespace
{
constexpr Scalar tolerance = 0.0001F;

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}
}

TEST_CASE("common activation functions return expected values", "[activation]")
{
    SigmoidActivation<Scalar> sigmoid;
    StepActivation<Scalar> step;
    StepPolarActivation<Scalar> polarStep;
    ReLUActivation<Scalar> relu;
    TanhActivation<Scalar> tanh;

    requireClose(sigmoid(0.0F), 0.5F);
    requireClose(sigmoid.derivative(0.0F), 0.25F);

    REQUIRE(step(-1.0F) == 0.0F);
    REQUIRE(step(0.0F) == 1.0F);
    REQUIRE(step.derivative(2.0F) == 0.0F);

    REQUIRE(polarStep(-1.0F) == -1.0F);
    REQUIRE(polarStep(0.0F) == 1.0F);
    REQUIRE(polarStep.derivative(2.0F) == 0.0F);

    REQUIRE(relu(-1.0F) == 0.0F);
    REQUIRE(relu(2.0F) == 2.0F);
    REQUIRE(relu.derivative(-1.0F) == 0.0F);
    REQUIRE(relu.derivative(2.0F) == 1.0F);

    requireClose(tanh(0.0F), 0.0F);
    requireClose(tanh.derivative(0.0F), 1.0F);
}
