#pragma once

#include "base/Types.h"
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

template<typename T>
class Tensor;

/*
template<typename T>
class TensorTransposed : public Tensor<T> 
{
public:
    TensorTransposed(Tensor<T> &t) : Tensor<T>(t) {}
};

template<typename T>
TensorTransposed<T> transpose(Tensor<T> &t)
{
    return TensorTransposed<T>(t);
}
*/

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
    if (a.size() != b.size())
        throw std::runtime_error("Size mismatch in elementwise_mul.");

    Tensor<T> result(b.size());

    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }

    return result;
}

// element Wise Sum
template<typename T>
Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Size mismatch in elementwise_sum.");

    Tensor<T> result(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// element Wise subtraction
template<typename T>
Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Size mismatch in elementwise_sum.");

    Tensor<T> result(a.size());

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


    //  friend Tensor<T> operator* <T>(const TensorTransposed<T> &a,
    //                                 const Tensor<T> &b);
    //  friend Tensor<T> operator* <T>(const Tensor<Tensor<T>> &a,
    //                                 const Tensor<T> &b);

public:
    Tensor() = default;

    Tensor(size_t size, const T &value = T{})
        : data(size, value)
    {}

    template<typename IT>
    Tensor(IT begin, IT end)
        : data(begin, end)
    {}

    Tensor(const std::initializer_list<T> &init)
        : data(init)
    {}

    ~Tensor() = default;

    T dot(const Tensor<T> &b) const
    {
        if (this->size() != b.size())
            throw std::runtime_error("Size mismatch in dot product.");

        T result = T{};
        for (size_t i = 0; i < this->size(); ++i) {
            result += data[i] * b[i];
        }
        return result;
    }

    T matVecTransMul(const T &b) const
    {
        if (data.empty() || data.size() != b.size())
            throw std::runtime_error(
                "Matrix and vector size mismatch in matvec_mul.");

        auto dataSize = data.at(0).size();

        T result(dataSize);
        // it makes the transposition and dot product at the same time.
        for (size_t i = 0; i < dataSize; ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result[i] += data[j][i] * b[j];
            }
        }
        return result;
    }

    T matVecMul(const T &b) const
    {
        if (data.empty() || data.at(0).size() != b.size())
        throw std::runtime_error(
            "Matrix and vector size mismatch in matvec_mul.");

        auto dataSize = data.size();

        T result(dataSize);
        for (size_t i = 0; i < dataSize; ++i) {
                result[i] = data[i].dot(b);
        }
        return result;
    }
    

    Tensor<Tensor<T>> outer(const Tensor<T> &b) const
    {
        // Outer product: result[i][j] = this[i] * b[j]
        Tensor<Tensor<T>> result(this->size(), Tensor<T>(b.size(), T{}));
        for (size_t i = 0; i < this->size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result[i][j] = data[i] * b[j];
            }
        }
        return result;
    }

    // element wise
    Tensor<T> mul(const Tensor<T> &b) { return operator* <T>(data, b); }


    //Matrix Multiplication
    Tensor<T> matMul(const Tensor<T> &b)
    {
        // Matrix multiplication for 2D tensors (Tensor<Tensor<T>>)
        // 'this' is assumed to be a matrix (Tensor<Tensor<T>>)
        // 'b' is also a matrix (Tensor<Tensor<T>>)
        // Result is a matrix of size (rows of this) x (cols of b)
        if (this->size() == 0 || b.size() == 0)
            throw std::runtime_error("Empty matrices in matMul.");

        size_t rows = this->size();
        size_t cols = b[0].size();
        size_t inner = (*this)[0].size();

        // Check dimensions
        for (size_t i = 0; i < rows; ++i) {
            if ((*this)[i].size() != inner)
                throw std::runtime_error(
                    "Inconsistent row size in lhs matrix.");
        }
        for (size_t i = 0; i < b.size(); ++i) {
            if (b[i].size() != cols)
                throw std::runtime_error(
                    "Inconsistent row size in rhs matrix.");
        }
        if (inner != b.size())
            throw std::runtime_error(
                "Matrix size mismatch for multiplication.");

        Tensor<Tensor<T>> result(rows, Tensor<T>(cols, T{}));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                for (size_t k = 0; k < inner; ++k) {
                    result[i][j] += (*this)[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    typename std::vector<T>::iterator begin() { return data.begin(); }
    typename std::vector<T>::iterator end() { return data.end(); }
    typename std::vector<T>::const_iterator begin() const
    {
        return data.begin();
    }

    typename std::vector<T>::const_iterator end() const { return data.end(); }

    size_t size() const { return data.size(); }

    inline T &operator[](size_t index) { return data[index]; }

    inline const T &operator[](size_t index) const { return data[index]; }

    void push_back(const T &t) { return data.push_back(t); }

    void emplace_back(const T &t) { data.emplace_back(t); }

    void reserve(const size_t size) { return data.reserve(size); }

    T &at(const size_t i) { return data.at(i); }
    const T &at(const size_t i) const { return data.at(i); }

    T &back() { return data.back(); }
    T &front() { return data.front(); }
    const T &back() const { return data.back(); }
    const T &front() const { return data.front(); }

    bool empty() const { return data.empty(); }

    inline void resize(const size_t t) { data.resize(t, T{}); }

    inline void resize(const size_t t, const T s) { data.resize(t, s); }

protected:
    std::vector<T> data; // Store the tensor data
};