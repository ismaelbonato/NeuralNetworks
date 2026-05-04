#pragma once

#include "base/Types.h"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Shape
{
    std::vector<size_t> dimensions;

    Shape() = default;

    Shape(std::initializer_list<size_t> newDimensions)
        : dimensions(newDimensions)
    {}

    explicit Shape(const std::vector<size_t> &newDimensions)
        : dimensions(newDimensions)
    {}

    bool empty() const { return dimensions.empty(); }

    bool isValid() const
    {
        return !dimensions.empty()
               && std::all_of(dimensions.begin(),
                              dimensions.end(),
                              [](const size_t dimension) {
                                  return dimension > 0;
                              });
    }

    size_t elementCount() const
    {
        if (dimensions.empty()) {
            return 0;
        }

        size_t count = 1;
        for (const size_t dimension : dimensions) {
            if (dimension == 0) {
                return 0;
            }
            count *= dimension;
        }

        return count;
    }
};

template<typename T>
class Tensor;

template<typename T>
bool operator==(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }

    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T>
bool operator!=(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
    return !(lhs == rhs);
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &item)
{
    os << "{";
    for (auto beg = item.begin(); beg != item.end(); ++beg) {
        os << *beg;
        if ((beg + 1) != item.end())
            os << ", ";
    }
    os << "}";
    return os;
}
//  it is elementwise multiplication
template<typename T>
Tensor<T> operator*(const Tensor<T> &a, const Tensor<T> &b)
{
    if (!a.hasSameShapeAs(b))
        throw std::runtime_error("Size mismatch in elementwise_mul.");

    Tensor<T> result(b.size());
    result.dimensions = a.dimensions;
    result.updateStrides();

    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }

    return result;
}

// element Wise Sum
template<typename T>
Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b)
{
    if (!a.hasSameShapeAs(b))
        throw std::runtime_error("Size mismatch in elementwise_sum.");

    Tensor<T> result(a.size());
    result.dimensions = a.dimensions;
    result.updateStrides();

    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// element Wise subtraction
template<typename T>
Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b)
{
    if (!a.hasSameShapeAs(b))
        throw std::runtime_error("Size mismatch in elementwise_sum.");

    Tensor<T> result(a.size());
    result.dimensions = a.dimensions;
    result.updateStrides();

    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

template<typename T>
class Tensor
{
    friend bool operator== <T>(const Tensor<T> &lhs, const Tensor<T> &rhs);
    friend bool operator!= <T>(const Tensor<T> &lhs, const Tensor<T> &rhs);
    friend std::ostream &operator<< <T>(std::ostream &os, const Tensor<T> &item);
    // it is not matrix multiplication, it is elementwise
    friend Tensor<T> operator* <T>(const Tensor<T> &a, const Tensor<T> &b);
    friend Tensor<T> operator+ <T>(const Tensor<T> &a, const Tensor<T> &b);
    friend Tensor<T> operator- <T>(const Tensor<T> &a, const Tensor<T> &b);

public:
    using value_type = T;

    Tensor() = default;
    explicit Tensor(size_t size, const T &value = T{});

    template<typename IT>
    Tensor(IT begin, IT end);

    Tensor(const std::initializer_list<T> &init);

    ~Tensor() = default;

    static Tensor<T> withShape(const Shape &shape, const T &value = T{});
    static Tensor<T> vector(const size_t size, const T &value = T{});
    static Tensor<T> matrix(const size_t rows,
                            const size_t cols,
                            const T &value = T{});
    static Tensor<T> matrix(
        const std::initializer_list<std::initializer_list<T>> rows);

    const std::vector<size_t> &shape() const;
    const std::vector<size_t> &strides() const;
    size_t rank() const;
    size_t elementCount() const;
    void reshape(const Shape &shape);

    template<typename UnaryOperation>
    Tensor<T> map(UnaryOperation operation) const;

    template<typename UnaryOperation>
    Tensor<T> mapValues(UnaryOperation operation) const;

    template<typename BinaryOperation>
    Tensor<T> zip(const Tensor<T> &other, BinaryOperation operation) const;

    template<typename BinaryOperation>
    Tensor<T> zipValues(const Tensor<T> &other, BinaryOperation operation) const;

    bool hasShape(const Shape &shape) const;
    bool hasSameShapeAs(const Tensor<T> &other) const;

    template<typename Value>
    void setDiagonal(const Value &value);

    template<typename Generator>
    void generate(Generator generator);

    T dot(const Tensor<T> &b) const;
    Tensor<T> conv1D(const Tensor<T> &kernel,
                     size_t stride = 1,
                     size_t padding = 0) const;
    Tensor<T> matVec(const Tensor<T> &b) const;
    Tensor<T> transposedMatVec(const Tensor<T> &b) const;
    Tensor<T> outer(const Tensor<T> &b) const;

    void outerInto(const Tensor<T> &b, Tensor<T> &result) const;

    // element wise
    Tensor<T> mul(const Tensor<T> &b);

    typename std::vector<T>::iterator begin();
    typename std::vector<T>::iterator end();
    typename std::vector<T>::const_iterator begin() const;
    typename std::vector<T>::const_iterator end() const;
    size_t size() const;

    T &operator[](size_t index);
    const T &operator[](size_t index) const;

    void push_back(const T &t);
    void emplace_back(const T &t);
    void reserve(const size_t size);

    T &at(const size_t i);
    const T &at(const size_t i) const;
    T &at(const std::initializer_list<size_t> indices);
    const T &at(const std::initializer_list<size_t> indices) const;
    T &at(const std::vector<size_t> &indices);
    const T &at(const std::vector<size_t> &indices) const;

    size_t offsetOf(const std::initializer_list<size_t> indices) const;
    size_t offsetOf(const std::vector<size_t> &indices) const;

    T &back();
    T &front();
    const T &back() const;
    const T &front() const;
    bool empty() const;

    void resize(const size_t t);
    void resize(const size_t t, const T &s);

protected:
    void updateStrides();
    size_t conv1DOutputLength(const Tensor<T> &kernel,
                              size_t stride,
                              size_t padding) const;
    T conv1DWindowSum(const Tensor<T> &kernel,
                      size_t outputChannel,
                      size_t outputIndex,
                      size_t stride,
                      size_t padding) const;

    std::vector<T> data; // Store the tensor data
    std::vector<size_t> dimensions;
    std::vector<size_t> dimensionStrides;
};

template<typename T>
Tensor<T>::Tensor(size_t size, const T &value)
    : data(size, value)
    , dimensions{size}
{
    updateStrides();
}

template<typename T>
template<typename IT>
Tensor<T>::Tensor(IT begin, IT end)
    : data(begin, end)
    , dimensions{data.size()}
{
    updateStrides();
}

template<typename T>
Tensor<T>::Tensor(const std::initializer_list<T> &init)
    : data(init)
    , dimensions{init.size()}
{
    updateStrides();
}

template<typename T>
Tensor<T> Tensor<T>::withShape(const Shape &shape, const T &value)
{
    if (shape.dimensions.empty()) {
        throw std::runtime_error("Tensor shape cannot be empty.");
    }

    size_t count = 1;
    for (const size_t dimension : shape.dimensions) {
        if (dimension == 0) {
            throw std::runtime_error(
                "Tensor shape dimensions must be greater than zero.");
        }
        count *= dimension;
    }

    Tensor<T> result(count, value);
    result.dimensions = shape.dimensions;
    result.updateStrides();
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::vector(const size_t size, const T &value)
{
    return withShape({size}, value);
}

template<typename T>
Tensor<T> Tensor<T>::matrix(const size_t rows, const size_t cols, const T &value)
{
    return withShape({rows, cols}, value);
}

template<typename T>
Tensor<T> Tensor<T>::matrix(
    const std::initializer_list<std::initializer_list<T>> rows)
{
    if (rows.size() == 0) {
        throw std::runtime_error("Tensor matrix rows cannot be empty.");
    }

    const size_t cols = rows.begin()->size();
    if (cols == 0) {
        throw std::runtime_error("Tensor matrix columns cannot be empty.");
    }

    Tensor<T> result = matrix(rows.size(), cols);
    size_t rowIndex = 0;
    for (const auto &row : rows) {
        if (row.size() != cols) {
            throw std::runtime_error(
                "Tensor matrix rows must have the same size.");
        }

        size_t colIndex = 0;
        for (const auto &value : row) {
            result.at({rowIndex, colIndex}) = value;
            ++colIndex;
        }
        ++rowIndex;
    }

    return result;
}

template<typename T>
const std::vector<size_t> &Tensor<T>::shape() const
{
    return dimensions;
}

template<typename T>
const std::vector<size_t> &Tensor<T>::strides() const
{
    return dimensionStrides;
}

template<typename T>
size_t Tensor<T>::rank() const
{
    return dimensions.size();
}

template<typename T>
size_t Tensor<T>::elementCount() const
{
    return data.size();
}

template<typename T>
void Tensor<T>::reshape(const Shape &shape)
{
    if (shape.dimensions.empty()) {
        throw std::runtime_error("Tensor shape cannot be empty.");
    }

    size_t count = 1;
    for (const size_t dimension : shape.dimensions) {
        if (dimension == 0) {
            throw std::runtime_error(
                "Tensor shape dimensions must be greater than zero.");
        }
        count *= dimension;
    }

    if (count != data.size()) {
        throw std::runtime_error("Tensor shape does not match element count.");
    }

    dimensions = shape.dimensions;
    updateStrides();
}

template<typename T>
template<typename UnaryOperation>
Tensor<T> Tensor<T>::map(UnaryOperation operation) const
{
    Tensor<T> result(size());
    result.dimensions = dimensions;
    result.updateStrides();
    for (size_t i = 0; i < size(); ++i) {
        result[i] = operation(data[i]);
    }
    return result;
}

template<typename T>
template<typename UnaryOperation>
Tensor<T> Tensor<T>::mapValues(UnaryOperation operation) const
{
    Tensor<T> result(size());
    result.dimensions = dimensions;
    result.updateStrides();
    for (size_t i = 0; i < size(); ++i) {
        if constexpr (requires { data[i].mapValues(operation); }) {
            result[i] = data[i].mapValues(operation);
        } else {
            result[i] = operation(data[i]);
        }
    }
    return result;
}

template<typename T>
template<typename BinaryOperation>
Tensor<T> Tensor<T>::zip(const Tensor<T> &other, BinaryOperation operation) const
{
    if (!hasSameShapeAs(other)) {
        throw std::runtime_error("Size mismatch in tensor zip.");
    }

    Tensor<T> result(size());
    result.dimensions = dimensions;
    result.updateStrides();
    for (size_t i = 0; i < size(); ++i) {
        result[i] = operation(data[i], other[i]);
    }
    return result;
}

template<typename T>
template<typename BinaryOperation>
Tensor<T> Tensor<T>::zipValues(const Tensor<T> &other,
                               BinaryOperation operation) const
{
    if (!hasSameShapeAs(other)) {
        throw std::runtime_error("Size mismatch in tensor zip values.");
    }

    Tensor<T> result(size());
    result.dimensions = dimensions;
    result.updateStrides();
    for (size_t i = 0; i < size(); ++i) {
        if constexpr (requires { data[i].zipValues(other[i], operation); }) {
            result[i] = data[i].zipValues(other[i], operation);
        } else {
            result[i] = operation(data[i], other[i]);
        }
    }
    return result;
}

template<typename T>
bool Tensor<T>::hasShape(const Shape &shape) const
{
    return dimensions == shape.dimensions;
}

template<typename T>
bool Tensor<T>::hasSameShapeAs(const Tensor<T> &other) const
{
    return dimensions == other.dimensions;
}

template<typename T>
template<typename Value>
void Tensor<T>::setDiagonal(const Value &value)
{
    if (rank() != 2) {
        throw std::runtime_error("Tensor diagonal requires a rank-2 tensor.");
    }

    const size_t diagonalSize = dimensions.at(0) < dimensions.at(1)
                                    ? dimensions.at(0)
                                    : dimensions.at(1);
    for (size_t i = 0; i < diagonalSize; ++i) {
        at({i, i}) = value;
    }
}

template<typename T>
template<typename Generator>
void Tensor<T>::generate(Generator generator)
{
    for (auto &item : data) {
        if constexpr (requires { item.generate(generator); }) {
            item.generate(generator);
        } else {
            item = generator();
        }
    }
}

template<typename T>
T Tensor<T>::dot(const Tensor<T> &b) const
{
    if (this->size() != b.size()) {
        throw std::runtime_error("Size mismatch in dot product.");
    }

    T result = T{};
    for (size_t i = 0; i < this->size(); ++i) {
        result += data[i] * b[i];
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::conv1D(const Tensor<T> &kernel,
                            size_t stride,
                            size_t padding) const
{
    const size_t outputLength = conv1DOutputLength(kernel, stride, padding);
    const size_t outputChannels = kernel.dimensions.at(0);
    Tensor<T> result = Tensor<T>::withShape({outputChannels, outputLength});

    for (size_t outputChannel = 0; outputChannel < outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < outputLength; ++outputIndex) {
            result.at({outputChannel, outputIndex})
                = conv1DWindowSum(kernel,
                                  outputChannel,
                                  outputIndex,
                                  stride,
                                  padding);
        }
    }

    return result;
}

template<typename T>
size_t Tensor<T>::conv1DOutputLength(const Tensor<T> &kernel,
                                     size_t stride,
                                     size_t padding) const
{
    if (rank() != 2) {
        throw std::runtime_error("1D convolution requires a rank-2 tensor.");
    }
    if (kernel.rank() != 3) {
        throw std::runtime_error(
            "1D convolution kernel requires a rank-3 tensor.");
    }
    if (stride == 0) {
        throw std::runtime_error(
            "1D convolution stride must be greater than zero.");
    }

    const size_t inputChannels = dimensions.at(0);
    const size_t inputLength = dimensions.at(1);
    const size_t kernelChannels = kernel.dimensions.at(1);
    const size_t kernelSize = kernel.dimensions.at(2);
    const size_t paddedInputLength = inputLength + (2 * padding);

    if (inputChannels != kernelChannels) {
        throw std::runtime_error(
            "1D convolution input channels must match kernel channels.");
    }
    if (kernelSize > paddedInputLength) {
        throw std::runtime_error(
            "1D convolution kernel is larger than padded input.");
    }

    return ((paddedInputLength - kernelSize) / stride) + 1;
}

template<typename T>
T Tensor<T>::conv1DWindowSum(const Tensor<T> &kernel,
                             size_t outputChannel,
                             size_t outputIndex,
                             size_t stride,
                             size_t padding) const
{
    const size_t inputChannels = dimensions.at(0);
    const size_t inputLength = dimensions.at(1);
    const size_t kernelSize = kernel.dimensions.at(2);

    T sum = T{};
    for (size_t inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
        for (size_t kernelIndex = 0; kernelIndex < kernelSize; ++kernelIndex) {
            const size_t paddedInputIndex = (outputIndex * stride)
                                            + kernelIndex;
            if (paddedInputIndex < padding) {
                continue;
            }

            const size_t inputIndex = paddedInputIndex - padding;
            if (inputIndex >= inputLength) {
                continue;
            }

            sum += at({inputChannel, inputIndex})
                   * kernel.at({outputChannel, inputChannel, kernelIndex});
        }
    }

    return sum;
}

template<typename T>
Tensor<T> Tensor<T>::matVec(const Tensor<T> &b) const
{
    if (rank() == 1 && b.rank() == 2) {
        const size_t rows = b.dimensions.at(0);
        const size_t cols = b.dimensions.at(1);
        if (size() != cols) {
            throw std::runtime_error("Matrix columns must match vector size.");
        }

        Tensor<T> result = Tensor<T>::withShape({rows});
        for (size_t row = 0; row < rows; ++row) {
            T sum = T{};
            for (size_t col = 0; col < cols; ++col) {
                sum += b.at({row, col}) * at(col);
            }
            result[row] = sum;
        }

        return result;
    }

    if (rank() != 2) {
        throw std::runtime_error(
            "Matrix-vector multiplication requires a rank-2 matrix.");
    }
    if (b.rank() != 1) {
        throw std::runtime_error(
            "Matrix-vector multiplication requires a rank-1 vector.");
    }

    const size_t rows = dimensions.at(0);
    const size_t cols = dimensions.at(1);
    if (b.size() != cols) {
        throw std::runtime_error("Matrix columns must match vector size.");
    }

    Tensor<T> result = Tensor<T>::withShape({rows});
    for (size_t row = 0; row < rows; ++row) {
        T sum = T{};
        for (size_t col = 0; col < cols; ++col) {
            sum += at({row, col}) * b[col];
        }
        result[row] = sum;
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::transposedMatVec(const Tensor<T> &b) const
{
    if (rank() != 2) {
        throw std::runtime_error("Transposed matrix-vector multiplication "
                                 "requires a rank-2 matrix.");
    }
    if (b.rank() != 1) {
        throw std::runtime_error("Transposed matrix-vector multiplication "
                                 "requires a rank-1 vector.");
    }

    const size_t rows = dimensions.at(0);
    const size_t cols = dimensions.at(1);
    if (b.size() != rows) {
        throw std::runtime_error("Matrix rows must match vector size.");
    }

    Tensor<T> result = Tensor<T>::withShape({cols});
    for (size_t col = 0; col < cols; ++col) {
        T sum = T{};
        for (size_t row = 0; row < rows; ++row) {
            sum += at({row, col}) * b[row];
        }
        result[col] = sum;
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::outer(const Tensor<T> &b) const
{
    if (rank() != 1 || b.rank() != 1) {
        throw std::runtime_error("Outer product requires rank-1 tensors.");
    }

    const size_t rows = size();
    const size_t cols = b.size();

    Tensor<T> result = Tensor<T>::withShape({rows, cols});
    for (size_t row = 0; row < rows; ++row) {
        const T lhs = data[row];
        const size_t base = row * cols;
        for (size_t col = 0; col < cols; ++col) {
            result[base + col] = lhs * b[col];
        }
    }

    return result;
}

template<typename T>
void Tensor<T>::outerInto(const Tensor<T> &b, Tensor<T> &result) const
{
    if (rank() != 1 || b.rank() != 1) {
        throw std::runtime_error("Outer product requires rank-1 tensors.");
    }
    if (!result.hasShape({size(), b.size()})) {
        throw std::runtime_error(
            "Outer product result tensor has incorrect shape.");
    }

    const size_t rows = size();
    const size_t cols = b.size();

    for (size_t row = 0; row < rows; ++row) {
        const T lhs = data[row];
        const size_t base = row * cols;
        for (size_t col = 0; col < cols; ++col) {
            result[base + col] = lhs * b[col];
        }
    }
}

template<typename T>
typename std::vector<T>::iterator Tensor<T>::begin()
{
    return data.begin();
}

template<typename T>
typename std::vector<T>::iterator Tensor<T>::end()
{
    return data.end();
}

template<typename T>
typename std::vector<T>::const_iterator Tensor<T>::begin() const
{
    return data.begin();
}

template<typename T>
typename std::vector<T>::const_iterator Tensor<T>::end() const
{
    return data.end();
}

template<typename T>
size_t Tensor<T>::size() const
{
    return data.size();
}

template<typename T>
T &Tensor<T>::operator[](size_t index)
{
    return data[index];
}

template<typename T>
const T &Tensor<T>::operator[](size_t index) const
{
    return data[index];
}

template<typename T>
void Tensor<T>::push_back(const T &t)
{
    data.push_back(t);
    dimensions = {data.size()};
    updateStrides();
}

template<typename T>
void Tensor<T>::emplace_back(const T &t)
{
    data.emplace_back(t);
    dimensions = {data.size()};
    updateStrides();
}

template<typename T>
void Tensor<T>::reserve(const size_t size)
{
    data.reserve(size);
}

template<typename T>
T &Tensor<T>::at(const size_t i)
{
    return data.at(i);
}

template<typename T>
const T &Tensor<T>::at(const size_t i) const
{
    return data.at(i);
}

template<typename T>
T &Tensor<T>::at(const std::initializer_list<size_t> indices)
{
    return data.at(offsetOf(indices));
}

template<typename T>
const T &Tensor<T>::at(const std::initializer_list<size_t> indices) const
{
    return data.at(offsetOf(indices));
}

template<typename T>
T &Tensor<T>::at(const std::vector<size_t> &indices)
{
    return data.at(offsetOf(indices));
}

template<typename T>
const T &Tensor<T>::at(const std::vector<size_t> &indices) const
{
    return data.at(offsetOf(indices));
}

template<typename T>
size_t Tensor<T>::offsetOf(const std::initializer_list<size_t> indices) const
{
    return offsetOf(std::vector<size_t>(indices));
}

template<typename T>
size_t Tensor<T>::offsetOf(const std::vector<size_t> &indices) const
{
    if (indices.size() != dimensions.size()) {
        throw std::runtime_error(
            "Tensor index rank does not match tensor rank.");
    }

    size_t offset = 0;
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        const size_t index = indices.at(axis);
        if (index >= dimensions.at(axis)) {
            throw std::runtime_error("Tensor index is out of bounds.");
        }

        offset += index * dimensionStrides.at(axis);
    }

    return offset;
}

template<typename T>
T &Tensor<T>::back()
{
    return data.back();
}

template<typename T>
T &Tensor<T>::front()
{
    return data.front();
}

template<typename T>
const T &Tensor<T>::back() const
{
    return data.back();
}

template<typename T>
const T &Tensor<T>::front() const
{
    return data.front();
}

template<typename T>
bool Tensor<T>::empty() const
{
    return data.empty();
}

template<typename T>
void Tensor<T>::resize(const size_t t)
{
    data.resize(t, T{});
    dimensions = {t};
    updateStrides();
}

template<typename T>
void Tensor<T>::resize(const size_t t, const T &s)
{
    data.resize(t, s);
    dimensions = {t};
    updateStrides();
}

template<typename T>
void Tensor<T>::updateStrides()
{
    dimensionStrides = std::vector<size_t>(dimensions.size(), 1);
    if (dimensions.empty()) {
        return;
    }

    for (size_t i = dimensions.size() - 1; i > 0; --i) {
        dimensionStrides[i - 1] = dimensionStrides[i] * dimensions[i];
    }
}
