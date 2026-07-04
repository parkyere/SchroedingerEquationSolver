#pragma once

// Hand-rolled complex number type (purist reinvention boundary: we do not use
// std::complex -- see docs/ARCHITECTURE.md). The TDSE state, FFT twiddle
// factors, and split-operator phase factors are all built on this.

#include <cmath>

namespace ses {

template <typename T>
struct Complex {
    T re{};
    T im{};
};

template <typename T>
constexpr Complex<T> operator+(Complex<T> a, Complex<T> b) {
    return {a.re + b.re, a.im + b.im};
}

template <typename T>
constexpr Complex<T> operator-(Complex<T> a, Complex<T> b) {
    return {a.re - b.re, a.im - b.im};
}

// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
template <typename T>
constexpr Complex<T> operator*(Complex<T> a, Complex<T> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template <typename T>
constexpr Complex<T> operator*(T s, Complex<T> z) {
    return {s * z.re, s * z.im};
}

template <typename T>
constexpr Complex<T> operator*(Complex<T> z, T s) {
    return {z.re * s, z.im * s};
}

template <typename T>
constexpr Complex<T> conj(Complex<T> z) {
    return {z.re, -z.im};
}

// |z|^2 — the probability-density operation.
template <typename T>
constexpr T norm_sq(Complex<T> z) {
    return z.re * z.re + z.im * z.im;
}

template <typename T>
T abs(Complex<T> z) {
    return std::sqrt(norm_sq(z));
}

// a / b = a * conj(b) / |b|^2
template <typename T>
constexpr Complex<T> operator/(Complex<T> a, Complex<T> b) {
    const T d = norm_sq(b);
    return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
}

}  // namespace ses
