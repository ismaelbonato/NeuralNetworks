#include "base/Initializer.h"
#include "base/Tensor.h"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

TEST_CASE("initializers fill tensors through the shared strategy interface", "[initializer]")
{
    Pattern values = Pattern::vector(3, 1.0F);

    ZeroInitializer<Scalar>{}.fill(values);
    REQUIRE(values == Pattern{0.0F, 0.0F, 0.0F});

    ConstantInitializer<Scalar>{2.5F}.fill(values);
    REQUIRE(values == Pattern{2.5F, 2.5F, 2.5F});

    UniformInitializer<Scalar>{-0.25F, 0.25F}.fill(values);
    for (const Scalar value : values) {
        REQUIRE(value >= -0.25F);
        REQUIRE(value <= 0.25F);
    }
}

TEST_CASE("tensor keeps value initializer lists as one-dimensional data", "[tensor]")
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
    REQUIRE(image.front() == 1.0F);
    REQUIRE(image.back() == 1.0F);
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
    const auto matrix = Tensor<Scalar>::matrix({{1.0F, 2.0F, 3.0F},
                                                {4.0F, 5.0F, 6.0F}});

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

TEST_CASE("tensor vector and matrix factories reject empty dimensions", "[tensor]")
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
    REQUIRE_THROWS_AS(a - b, std::runtime_error);
    REQUIRE_THROWS_AS(a * b, std::runtime_error);
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

TEST_CASE("tensor matrix vector multiplication rejects invalid shapes", "[tensor]")
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

TEST_CASE("tensor transposed matrix vector multiplication uses explicit shape", "[tensor]")
{
    auto matrix = Tensor<Scalar>::withShape({2, 3});
    matrix.at({0, 0}) = 1.0F;
    matrix.at({0, 1}) = 2.0F;
    matrix.at({0, 2}) = 3.0F;
    matrix.at({1, 0}) = 4.0F;
    matrix.at({1, 1}) = 5.0F;
    matrix.at({1, 2}) = 6.0F;

    const Pattern vector = {7.0F, 8.0F};

    const Pattern result = matrix.transposedMatVec(vector);

    REQUIRE(result == Pattern{39.0F, 54.0F, 69.0F});
    REQUIRE(result.shape() == std::vector<size_t>{3});
}

TEST_CASE("tensor transposed matrix vector multiplication rejects invalid shapes", "[tensor]")
{
    const auto notMatrix = Tensor<Scalar>::withShape({2, 3, 4});
    const auto matrix = Tensor<Scalar>::withShape({2, 3});
    const auto notVector = Tensor<Scalar>::withShape({2, 1});
    const Pattern shortVector = {1.0F};

    REQUIRE_THROWS_AS(notMatrix.transposedMatVec(Pattern{1.0F, 2.0F}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(matrix.transposedMatVec(notVector), std::runtime_error);
    REQUIRE_THROWS_AS(matrix.transposedMatVec(shortVector), std::runtime_error);
}

TEST_CASE("tensor outer product uses explicit shape", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F};
    const Pattern b = {3.0F, 4.0F, 5.0F};

    const auto result = a.outer(b);

    REQUIRE(result.shape() == std::vector<size_t>{2, 3});
    REQUIRE(result.at({0, 0}) == 3.0F);
    REQUIRE(result.at({0, 1}) == 4.0F);
    REQUIRE(result.at({0, 2}) == 5.0F);
    REQUIRE(result.at({1, 0}) == 6.0F);
    REQUIRE(result.at({1, 1}) == 8.0F);
    REQUIRE(result.at({1, 2}) == 10.0F);
}

TEST_CASE("tensor outer product rejects non-vector shapes", "[tensor]")
{
    const auto matrix = Tensor<Scalar>::withShape({2, 3});
    const Pattern vector = {1.0F, 2.0F};

    REQUIRE_THROWS_AS(matrix.outer(vector), std::runtime_error);
    REQUIRE_THROWS_AS(vector.outer(matrix), std::runtime_error);
}

TEST_CASE("tensor maps values with a unary operation", "[tensor]")
{
    const Pattern values = {1.0F, 2.0F, 3.0F};

    REQUIRE(values.map([](Scalar value) { return value * value; }) ==
            Pattern{1.0F, 4.0F, 9.0F});
}

TEST_CASE("tensor recursively maps batch values with a unary operation", "[tensor]")
{
    const Batch batch = {{1.0F, 2.0F}, {3.0F, 4.0F}};

    REQUIRE(batch.mapValues([](Scalar value) { return value * 2.0F; }) ==
            Batch{{2.0F, 4.0F}, {6.0F, 8.0F}});
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

TEST_CASE("tensor recursively zips batch values with a binary operation", "[tensor]")
{
    const Batch a = {{1.0F, 2.0F}, {3.0F, 4.0F}};
    const Batch b = {{5.0F, 6.0F}, {7.0F, 8.0F}};

    REQUIRE(a.zipValues(b, [](Scalar lhs, Scalar rhs) { return lhs + rhs; }) ==
            Batch{{6.0F, 8.0F}, {10.0F, 12.0F}});
    REQUIRE_THROWS_AS(a.zipValues(Batch{{1.0F, 2.0F}}, [](Scalar lhs, Scalar rhs) {
                          return lhs + rhs;
                      }),
                      std::runtime_error);
}

TEST_CASE("tensor generates values in vectors and batches", "[tensor]")
{
    Pattern values(3);
    Scalar next = 1.0F;

    values.generate([&next]() { return next++; });

    REQUIRE(values == Pattern{1.0F, 2.0F, 3.0F});

    Batch batch(2, Pattern(2));
    next = 1.0F;

    batch.generate([&next]() { return next++; });

    REQUIRE(batch == Batch{{1.0F, 2.0F}, {3.0F, 4.0F}});
}

TEST_CASE("tensor reports explicit flat shape", "[tensor]")
{
    const auto tensor = Tensor<Scalar>::withShape({2, 3, 4});

    REQUIRE(tensor.hasShape({2, 3, 4}));
    REQUIRE_FALSE(tensor.hasShape({2, 12}));
    REQUIRE_FALSE(tensor.hasShape({24}));
}

TEST_CASE("tensor sets flat matrix diagonal", "[tensor]")
{
    auto matrix = Tensor<Scalar>::matrix({{1.0F, 2.0F}, {3.0F, 4.0F}});

    matrix.setDiagonal(0.0F);

    REQUIRE(matrix.at({0, 0}) == 0.0F);
    REQUIRE(matrix.at({0, 1}) == 2.0F);
    REQUIRE(matrix.at({1, 0}) == 3.0F);
    REQUIRE(matrix.at({1, 1}) == 0.0F);
}

TEST_CASE("tensor computes dot product", "[tensor]")
{
    const Pattern a = {1.0F, 2.0F, 3.0F};
    const Pattern b = {4.0F, 5.0F, 6.0F};

    REQUIRE(a.dot(b) == 32.0F);
}
