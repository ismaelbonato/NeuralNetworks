#include "base/Tensor.h"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

using namespace nn;

//todo:make sure one assert per test
TEST_CASE("tensor keeps value initializer lists as one-dimensional data",
          "[tensor]")
{
    const Pattern values = {1.0F, 2.0F, 3.0F};

    REQUIRE(values.shape() == std::vector<size_t>{3});
    REQUIRE(values.rank() == 1);
    REQUIRE(values.elementCount() == 3);
    REQUIRE(values == Pattern{1.0F, 2.0F, 3.0F});
}

TEST_CASE("tensor can be allocated from explicit shape", "[tensor]")
{
    const auto image = Tensor<Scalar>::withShape({28, 28, 3}, 1.0F);

    REQUIRE(image.shape() == std::vector<size_t>{28, 28, 3});
    REQUIRE(image.rank() == 3);
    REQUIRE(image.strides() == std::vector<size_t>{84, 3, 1});
    REQUIRE(image.elementCount() == 2352);
    REQUIRE(image.size() == 2352);
    REQUIRE(image.at(0) == 1.0F);
    REQUIRE(image.at(image.size() - 1) == 1.0F);
}

TEST_CASE("tensor can be allocated as a vector", "[tensor]")
{
    const auto vector = Tensor<Scalar>::vector(3, 2.0F);

    REQUIRE(vector.shape() == std::vector<size_t>{3});
    REQUIRE(vector.rank() == 1);
    REQUIRE(vector.strides() == std::vector<size_t>{1});
    REQUIRE(vector.elementCount() == 3);
    REQUIRE(vector == Pattern{2.0F, 2.0F, 2.0F});
}

TEST_CASE("tensor can be allocated as a matrix", "[tensor]")
{
    const auto matrix = Tensor<Scalar>::matrix(2, 3, 4.0F);

    REQUIRE(matrix.shape() == std::vector<size_t>{2, 3});
    REQUIRE(matrix.rank() == 2);
    REQUIRE(matrix.strides() == std::vector<size_t>{3, 1});
    REQUIRE(matrix.elementCount() == 6);
    REQUIRE(matrix.at({0, 0}) == 4.0F);
    REQUIRE(matrix.at({1, 2}) == 4.0F);
}

TEST_CASE("tensor can be allocated as a matrix from rows", "[tensor]")
{
    const auto matrix = Tensor<Scalar>::matrix(
        {{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

    REQUIRE(matrix.shape() == std::vector<size_t>{2, 3});
    REQUIRE(matrix.at({0, 0}) == 1.0F);
    REQUIRE(matrix.at({0, 2}) == 3.0F);
    REQUIRE(matrix.at({1, 0}) == 4.0F);
    REQUIRE(matrix.at({1, 2}) == 6.0F);
}

TEST_CASE("tensor matrix row factory rejects invalid rows", "[tensor]")
{
    REQUIRE_THROWS_AS(Tensor<Scalar>::matrix({}), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor<Scalar>::matrix({{}}), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor<Scalar>::matrix({{1.0F, 2.0F}, {3.0F}}),
                      std::runtime_error);
}

TEST_CASE("tensor vector and matrix factories reject empty dimensions",
          "[tensor]")
{
    REQUIRE_THROWS_AS(Tensor<Scalar>::vector(0), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor<Scalar>::matrix(0, 3), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor<Scalar>::matrix(2, 0), std::runtime_error);
}

TEST_CASE("tensor indexes shaped storage in row-major order", "[tensor]")
{
    auto tensor = Tensor<Scalar>::withShape({2, 3, 4});

    tensor.at({1, 2, 3}) = 42.0F;

    REQUIRE(tensor.offsetOf({0, 0, 0}) == 0);
    REQUIRE(tensor.offsetOf({0, 1, 0}) == 4);
    REQUIRE(tensor.offsetOf({1, 0, 0}) == 12);
    REQUIRE(tensor.offsetOf({1, 2, 3}) == 23);
    REQUIRE(tensor.at({1, 2, 3}) == 42.0F);
    REQUIRE(tensor.at(23) == 42.0F);
}

TEST_CASE("tensor rejects shaped indexes with wrong rank or bounds", "[tensor]")
{
    auto tensor = Tensor<Scalar>::withShape({2, 3, 4});

    REQUIRE_THROWS_AS(tensor.offsetOf({1, 2}), std::runtime_error);
    REQUIRE_THROWS_AS(tensor.offsetOf({2, 0, 0}), std::runtime_error);
    REQUIRE_THROWS_AS(tensor.at({0, 3, 0}), std::runtime_error);
}

TEST_CASE("tensor indexes shaped storage with dynamic index vectors", "[tensor]")
{
    auto tensor = Tensor<Scalar>::withShape({2, 3, 4});
    const std::vector<size_t> index{1, 2, 3};

    tensor.at(index) = 42.0F;

    REQUIRE(tensor.offsetOf(index) == 23);
    REQUIRE(tensor.at(index) == 42.0F);
}

TEST_CASE("tensor can reshape when element count matches", "[tensor]")
{
    Pattern values = {1.0F, 2.0F, 3.0F, 4.0F};

    values.reshape({2, 2});

    REQUIRE(values.shape() == std::vector<size_t>{2, 2});
    REQUIRE(values.rank() == 2);
    REQUIRE(values.elementCount() == 4);
}

TEST_CASE("tensor rejects invalid explicit shapes", "[tensor]")
{
    REQUIRE_THROWS_AS(Tensor<Scalar>::withShape({}), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor<Scalar>::withShape({28, 0, 3}), std::runtime_error);

    Pattern values = {1.0F, 2.0F, 3.0F, 4.0F};
    REQUIRE_THROWS_AS(values.reshape({3, 3}), std::runtime_error);
}

TEST_CASE("tensor elementwise operations reject mismatched sizes", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F};
    const Pattern b = {1.0F};

    REQUIRE_THROWS_AS(a + b, std::runtime_error);
    REQUIRE_THROWS_AS(a * b, std::runtime_error);
}

TEST_CASE("tensor elementwise operations reject mismatched shapes", "[tensor]")
{
    const auto matrix = Tensor<Scalar>::withShape({2, 2}, 1.0F);
    const auto vector = Tensor<Scalar>::withShape({4}, 1.0F);

    REQUIRE_THROWS_AS(matrix + vector, std::runtime_error);
    REQUIRE_THROWS_AS(matrix * vector, std::runtime_error);
}

TEST_CASE("tensor matrix vector multiplication uses explicit shape", "[tensor]")
{
    auto matrix = Tensor<Scalar>::withShape({2, 3});
    matrix.at({0, 0}) = 1.0F;
    matrix.at({0, 1}) = 2.0F;
    matrix.at({0, 2}) = 3.0F;
    matrix.at({1, 0}) = 4.0F;
    matrix.at({1, 1}) = 5.0F;
    matrix.at({1, 2}) = 6.0F;

    const Pattern vector = {7.0F, 8.0F, 9.0F};

    const Pattern result = matrix.matVec(vector);

    REQUIRE(result == Pattern{50.0F, 122.0F});
    REQUIRE(result.shape() == std::vector<size_t>{2});
}

TEST_CASE("tensor vector matrix multiplication uses input receiver convention",
          "[tensor]")
{
    auto matrix = Tensor<Scalar>::withShape({2, 3});
    matrix.at({0, 0}) = 1.0F;
    matrix.at({0, 1}) = 2.0F;
    matrix.at({0, 2}) = 3.0F;
    matrix.at({1, 0}) = 4.0F;
    matrix.at({1, 1}) = 5.0F;
    matrix.at({1, 2}) = 6.0F;

    const Pattern vector = {7.0F, 8.0F, 9.0F};

    const Pattern result = vector.matVec(matrix);

    REQUIRE(result == Pattern{50.0F, 122.0F});
    REQUIRE(result.shape() == std::vector<size_t>{2});
}

TEST_CASE("tensor matrix vector multiplication rejects invalid shapes",
          "[tensor]")
{
    const auto notMatrix = Tensor<Scalar>::withShape({2, 3, 4});
    const auto matrix = Tensor<Scalar>::withShape({2, 3});
    const auto notVector = Tensor<Scalar>::withShape({3, 1});
    const Pattern shortVector = {1.0F, 2.0F};

    REQUIRE_THROWS_AS(notMatrix.matVec(Pattern{1.0F, 2.0F, 3.0F}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(matrix.matVec(notVector), std::runtime_error);
    REQUIRE_THROWS_AS(matrix.matVec(shortVector), std::runtime_error);
}

TEST_CASE("tensor maps values with a unary operation", "[tensor]")
{
    const Pattern values = {1.0F, 2.0F, 3.0F};

    REQUIRE(values.map([](Scalar value) { return value * value; })
            == Pattern{1.0F, 4.0F, 9.0F});
}

TEST_CASE("tensor reports explicit multidimensional shape", "[tensor]")
{
    const auto tensor = Tensor<Scalar>::withShape({2, 3, 4});

    REQUIRE(tensor.hasShape({2, 3, 4}));
    REQUIRE_FALSE(tensor.hasShape({2, 12}));
    REQUIRE_FALSE(tensor.hasShape({24}));
}
