#include "base/LearningRule.h"
#include "base/Types.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("learning rules update weights according to their formulas", "[learning-rule]")
{
    SGDRule<Scalar> sgd;
    PerceptronRule<Scalar> perceptron;
    HebbianRule<Scalar> hebbian;

    REQUIRE(sgd.updateWeight(2.0F, 0.5F, 0.1F) == 1.95F);
    REQUIRE(perceptron.updateWeight(2.0F, 0.5F, 0.1F) == 2.05F);
    REQUIRE(hebbian.updateWeight(2.0F, 0.5F, 0.1F) == 2.5F);
}
