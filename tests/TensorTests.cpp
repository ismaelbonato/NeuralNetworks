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

TEST_CASE("tensor matrix vector multiplication rejects ragged matrices", "[tensor]")
{
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F}};

    REQUIRE_THROWS_AS(matrix.matVecMul({1.0F, 2.0F}), std::runtime_error);
}

TEST_CASE("tensor transposed matrix vector multiplication rejects ragged matrices", "[tensor]")
{
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F}};

    REQUIRE_THROWS_AS(matrix.matVecTransMul({1.0F, 2.0F}), std::runtime_error);
}

TEST_CASE("tensor maps values with a unary operation", "[tensor]")
{
    const Pattern values = {1.0F, 2.0F, 3.0F};

    REQUIRE(values.map([](Scalar value) { return value * value; }) ==
            Pattern{1.0F, 4.0F, 9.0F});
}

TEST_CASE("tensor recursively maps nested values with a unary operation", "[tensor]")
{
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F, 4.0F}};

    REQUIRE(matrix.mapValues([](Scalar value) { return value * 2.0F; }) ==
            Patterns{{2.0F, 4.0F}, {6.0F, 8.0F}});
}

TEST_CASE("tensor zips values with a binary operation", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F, 3.0F};
    const Pattern b = {4.0F, 5.0F, 6.0F};

    REQUIRE(a.zip(b, [](Scalar lhs, Scalar rhs) { return lhs + rhs; }) ==
            Pattern{5.0F, 7.0F, 9.0F});
    REQUIRE_THROWS_AS(a.zip(Pattern{1.0F}, [](Scalar lhs, Scalar rhs) {
                          return lhs + rhs;
                      }),
                      std::runtime_error);
}

TEST_CASE("tensor recursively zips nested values with a binary operation", "[tensor]")
{
    const Patterns a = {{1.0F, 2.0F}, {3.0F, 4.0F}};
    const Patterns b = {{5.0F, 6.0F}, {7.0F, 8.0F}};

    REQUIRE(a.zipValues(b, [](Scalar lhs, Scalar rhs) { return lhs + rhs; }) ==
            Patterns{{6.0F, 8.0F}, {10.0F, 12.0F}});
    REQUIRE_THROWS_AS(a.zipValues(Patterns{{1.0F, 2.0F}}, [](Scalar lhs, Scalar rhs) {
                          return lhs + rhs;
                      }),
                      std::runtime_error);
}

TEST_CASE("tensor generates values in vectors and nested matrices", "[tensor]")
{
    Pattern values(3);
    Scalar next = 1.0F;

    values.generate([&next]() { return next++; });

    REQUIRE(values == Pattern{1.0F, 2.0F, 3.0F});

    Patterns matrix(2, Pattern(2));
    next = 1.0F;

    matrix.generate([&next]() { return next++; });

    REQUIRE(matrix == Patterns{{1.0F, 2.0F}, {3.0F, 4.0F}});
}

TEST_CASE("tensor reports nested matrix shape", "[tensor]")
{
    const Patterns matrix = {{1.0F, 2.0F}, {3.0F, 4.0F}};
    const Patterns ragged = {{1.0F, 2.0F}, {3.0F}};

    REQUIRE(matrix.hasShape(2, 2));
    REQUIRE_FALSE(matrix.hasShape(1, 2));
    REQUIRE_FALSE(matrix.hasShape(2, 1));
    REQUIRE_FALSE(ragged.hasShape(2, 2));
}

TEST_CASE("tensor sets nested matrix diagonal", "[tensor]")
{
    Patterns matrix = {{1.0F, 2.0F}, {3.0F, 4.0F}};

    matrix.setDiagonal(0.0F);

    REQUIRE(matrix == Patterns{{0.0F, 2.0F}, {3.0F, 0.0F}});
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
