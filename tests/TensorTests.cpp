#include "base/Tensor.h"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

TEST_CASE("tensor elementwise operations reject mismatched sizes", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F};
    const Pattern b = {1.0F};

    REQUIRE_THROWS_AS(a + b, std::runtime_error);
    REQUIRE_THROWS_AS(a - b, std::runtime_error);
    REQUIRE_THROWS_AS(a * b, std::runtime_error);
}

TEST_CASE("tensor matrix vector multiplication rejects mismatched shapes", "[tensor]")
{
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F, 4.0F}};

    REQUIRE_THROWS_AS(matrix.matVecMul({1.0F}), std::runtime_error);
}

TEST_CASE("tensor transposed matrix vector multiplication rejects mismatched shapes", "[tensor]")
{
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F, 4.0F}};

    REQUIRE_THROWS_AS(matrix.matVecTransMul({1.0F}), std::runtime_error);
}

TEST_CASE("tensor computes dot product and matrix vector products", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F, 3.0F};
    const Pattern b = {4.0F, 5.0F, 6.0F};
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F, 4.0F}};

    REQUIRE(a.dot(b) == 32.0F);
    REQUIRE(matrix.matVecMul({5.0F, 6.0F}) == Pattern{17.0F, 39.0F});
    REQUIRE(matrix.matVecTransMul({5.0F, 6.0F}) == Pattern{23.0F, 34.0F});
}

TEST_CASE("tensor computes outer product", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F};
    const Pattern b = {3.0F, 4.0F, 5.0F};

    REQUIRE(a.outer(b) == Patterns{{3.0F, 4.0F, 5.0F}, {6.0F, 8.0F, 10.0F}});
}
