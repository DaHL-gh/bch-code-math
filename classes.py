from __future__ import annotations
from calendar import c
from math import log
from pprint import pprint
from typing import List, Optional, AnyStr

import numpy as np


class ExpandedField:
    def __init__(self, expanding_poly: FieldPoly) -> None:
        self.expanding_poly = expanding_poly
        self.q = expanding_poly.q
        self.m = len(self.expanding_poly.coefs) - 1

        self.xi_powers = []
        self.calculate_basic_xi_powers()

    def calculate_basic_xi_powers(self):
        max_pow_decomp = -self.expanding_poly
        max_pow_decomp.letter = r"\xi"
        max_pow_decomp.coefs = max_pow_decomp.coefs[:-1]
        basic_power_count = len(max_pow_decomp.coefs)

        for i in range(basic_power_count):
            array = [0] * (i + 1)
            array[i] = 1
            self.xi_powers.append(FieldPoly(self.q, array, letter=r"\xi"))
        self.xi_powers.append(max_pow_decomp)

    def calculate_xi_powers(self):
        just_xi = FieldPoly(self.q, [0, 1])
        last_xi = self.xi_powers[-1]

        while True:
            last_xi *= just_xi
            last_xi = last_xi.simplify_power(self.xi_powers)

            if last_xi.coefs == [1]:
                break

            self.xi_powers.append(last_xi)

    def get_check_matrix(self, n, d) -> FieldMatrix:
        matrix = np.zeros((d - 1, n), dtype=object)
        for y in range(d - 1):
            for x in range(n):
                matrix[y][x] = FieldPoly(
                    self.q, [0] * (x * (y + 1) % len(self.xi_powers)) + [1], r"\xi"
                )

        return FieldMatrix(self.q, matrix)

    def get_gilberts_matrix(self, sindrom: FieldMatrix, r: int) -> FieldMatrix:
        gilberts_matrix = np.zeros((r, r), dtype=object)

        for y in range(r):
            for x in range(r):
                gilberts_matrix[y][x] = sindrom.data[0][(r - (x + 1)) + y]

        return FieldMatrix(sindrom.q, gilberts_matrix)

    def get_minimal_poly(self, xi: FieldPoly):
        orbit = []
        current = xi.copy()

        while True:
            orbit.append(current)
            current = current**2
            if current == xi:
                break

        # Построение (x - a0)(x - a1)...(x - a_{k-1})
        # В GF(2) вычитание — это сложение, поэтому (x - a) = (x + a)
        min_poly = FieldPoly(2, [1])  # Многочлен 1 (константа)

        for root in orbit:
            # Умножаем на (x + root)
            # Коэффициенты root.coefs представляют элемент поля как вектор над GF(2)
            # Например, если root = a^2 + a + 1, то root.coefs = [1, 1, 1] (для GF(8))
            term_coeffs = root.coefs.copy()
            term_coeffs.append(1)  # x имеет коэффициент 1
            term_poly = FieldPoly(2, term_coeffs)
            min_poly *= term_poly

        return min_poly


class FieldMatrix:
    def __init__(self, q, data: np.ndarray) -> None:
        self.data = data
        if len(self.data.shape) != 2:
            raise ValueError(
                f"Данные должны иметь форму матрицы, а не {self.data.shape}"
            )

        self.q = q

    def transpose(self) -> FieldMatrix:
        return FieldMatrix(self.q, np.transpose(self.data, axes=(1, 0)))

    def determinant(self) -> FieldPoly:
        return self.__determinant(self.q, self.data)

    def simplify_power(self, power_table) -> FieldMatrix:
        new_matrix = self.copy()
        for y in range(self.data.shape[0]):
            for x in range(self.data.shape[1]):
                new_matrix.data[y][x] = new_matrix.data[y][x].simplify_power(
                    power_table
                )
        return new_matrix

    def copy(self) -> FieldMatrix:
        return FieldMatrix(self.q, self.data.copy())

    @staticmethod
    def __determinant(q, matrix) -> FieldPoly:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix should be square")

        n = matrix.shape[0]

        # 1x1
        if n == 1:
            return matrix[0, 0]

        # 2x2
        if n == 2:
            return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

        # else
        det = FieldPoly(q)
        for col in range(n):
            minor = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)

            print(
                matrix[0, col].__class__, FieldMatrix.__determinant(q, minor).__class__
            )
            cofactor = matrix[0, col] * FieldMatrix.__determinant(q, minor)

            if col % 2 != 0:
                cofactor = -cofactor

            det += cofactor

        return det

    def inverse_gauss_jordan(self):
        matrix = self.data
        n = matrix.shape[0]
        A = matrix.copy()

        eye = np.zeros((n, n))
        for i in range(n):
            eye[i][i] = FieldPoly(self.q, [1])

        for i in range(n):
            pivot = A[i, i]
            inv_pivot = self.get_invert_poly(pivot)

            for j in range(n):
                A[i, j] = A[i, j] * inv_pivot
                eye[i, j] = eye[i, j] * inv_pivot

            for k in range(n):
                if k != i:
                    for j in range(n):
                        A[k, j] -= A[k, i] * A[i, j]
                        eye[k, j] -= A[k, i] * eye[i, j]
        return eye

    def __mul__(self, other) -> FieldMatrix:
        A = self.data
        B = other.data

        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"{A.shape} and {B.shape} matrices could not be mulitiplied"
            )

        width = B.shape[1]
        height = A.shape[0]
        depth = A.shape[1]

        result_matrix = np.ndarray((height, width), dtype=object)

        for x in range(width):
            for y in range(height):
                result_matrix[y][x] = FieldPoly(self.q)
                for z in range(depth):
                    result_matrix[y][x] += A[y][z] * B[z][x]

        return FieldMatrix(q, result_matrix)

    def __neg__(self) -> FieldMatrix:
        return FieldMatrix(self.q, -self.data)

    def __repr__(self) -> str:
        out = r"\begin{pmatrix}" + "\n"
        y, x = self.data.shape
        for j in range(y):
            values = []
            for i in range(x):
                values.append(str(self.data[j][i]))
            out += " & ".join(values) + r" \\" + "\n"

        out += r"\end{pmatrix}" + "\n"

        return out


class FieldPoly:
    def __init__(self, q: int, coefs: Optional[List] = None, letter: AnyStr = "x"):
        if coefs is None:
            self.coefs = [0]
        else:
            self.coefs = coefs

        self.q = q
        self.letter = letter

    def simplify_power(self, power_table: list[FieldPoly]) -> FieldPoly:
        new_poly = self.copy()
        temp_poly = FieldPoly(self.q, [0] * len(power_table))

        for power in range(len(power_table) - 1, -1, -1):
            temp_poly.coefs[power] = 1

            if power < len(new_poly.coefs):
                new_poly += (power_table[power] + temp_poly) * new_poly.coefs[power]

            temp_poly.coefs[power] = 0

        return new_poly

    def __add__(self, other: FieldPoly | int) -> FieldPoly:
        if isinstance(other, int):
            new_poly = self.copy()
            new_poly.coefs[0] += other
            new_poly.coefs[0] %= self.q

            return new_poly
        else:
            if len(self.coefs) > len(other.coefs):
                min, max = other.coefs, self.coefs
            else:
                min, max = self.coefs, other.coefs

            new_coefs = [0] * len(max)
            for i in range(len(max)):
                new_coefs[i] = max[i]
                if i < len(min):
                    new_coefs[i] += min[i]
                new_coefs[i] %= self.q

            self.__truncate_coefs(new_coefs)
            return FieldPoly(self.q, new_coefs, letter=self.letter)

    def __neg__(self) -> FieldPoly:
        new_coefs = self.coefs.copy()
        for i, coeff in enumerate(new_coefs):
            new_coefs[i] = -coeff % self.q

        return FieldPoly(self.q, new_coefs, letter=self.letter)

    def __sub__(self, other: FieldPoly) -> FieldPoly:
        return self + -other

    def __mul__(self, other: FieldPoly | int) -> FieldPoly:
        if isinstance(other, int):
            new_coefs = self.coefs.copy()
            for i in range(len(new_coefs)):
                new_coefs[i] = new_coefs[i] * other % self.q

        else:
            a = self.coefs
            q = self.q
            b = other.coefs

            new_coefs = [0] * (len(a) + len(b) - 1)
            for i in range(len(a)):
                for j in range(len(b)):
                    index = (i + j) % 31
                    new_coefs[index] = (new_coefs[index] + a[i] * b[j]) % q

        self.__truncate_coefs(new_coefs)
        return FieldPoly(self.q, new_coefs, letter=self.letter)

    def __pow__(self, p: int) -> FieldPoly:
        if p < 0:
            raise ValueError("power must be >= 0")

        new_poly = FieldPoly(self.q, [1])
        for i in range(p):
            new_poly *= self
        return new_poly

    def __eq__(self, other: FieldPoly | int) -> bool:
        if isinstance(other, int):
            return len(self.coefs) == 1 and other == self.coefs[0]
        else:
            a = self.coefs
            b = other.coefs

            if len(a) != len(b):
                return False

            return all(a[i] == b[i] for i in range(len(a)))

    def __hash__(self):
        return tuple(self.coefs).__hash__()

    @staticmethod
    def __truncate_coefs(coefs: list[int]):
        while len(coefs) > 1 and coefs[-1] == 0:
            coefs.pop()
        return coefs

    def __repr__(self) -> str:
        terms = []
        for power, coeff in reversed(list(enumerate(self.coefs))):
            if coeff:
                coeff_pref = coeff if coeff > 1 else ""
                if power == 0:
                    terms.append(f"{coeff}")
                elif power == 1:
                    terms.append(f"{coeff_pref}{self.letter}")
                else:
                    terms.append(f"{coeff_pref}{self.letter}^{{{power}}}")
        result = " + ".join(terms) if terms else "0"

        return result

    def calculate(self, x: FieldPoly) -> FieldPoly:
        ans = FieldPoly(self.q)
        x_rem_pow = FieldPoly(self.q, [1])
        for pow, coef in enumerate(self.coefs):
            ans += x_rem_pow * coef
            x_rem_pow *= x

        return ans

    def div(self, other: FieldPoly) -> tuple:
        q, r = np.polydiv(self.coefs[::-1], other.coefs[::-1])
        q = q.tolist()[::-1]
        r = r.tolist()[::-1]
        return q, r

    def copy(self) -> FieldPoly:
        return FieldPoly(self.q, self.coefs.copy(), self.letter)


if __name__ == "__main__":
    # # word = "12201010"
    # # t = 2
    # q = 3
    # init_poly = [2, 1, 1]

    word = "0101110101011001110001010111110"
    # word = "0101110101011001010001010011110"
    t = 2
    q = 2
    init_poly = [1, 1, 0, 1, 1, 1]

    n = len(word)
    d = 2 * t + 1
    expanding_poly = FieldPoly(q, init_poly)
    field = ExpandedField(expanding_poly)
    field.calculate_xi_powers()
    #
    # check_m = field.get_check_matrix(d=d, n=n)
    #
    # print(check_m.transpose())
    # print(check_m.simplify_power(field.xi_powers).transpose())
    #
    # word_array = np.array([[FieldPoly(q, [int(s)])] for s in word], dtype=object)
    # word_vector = FieldMatrix(q, word_array)
    # print(word_vector)
    # sindrom = check_m * word_vector
    # print(sindrom)
    # sindrom = sindrom.simplify_power(field.xi_powers)
    # print(sindrom)
    #
    # gilberts_matrix = field.get_gilberts_matrix(sindrom.transpose(), 2)
    # print(gilberts_matrix)
    # print(gilberts_matrix.determinant().simplify_power(field.xi_powers))

    # p1 = FieldPoly(q, [1, 2, 1, 3, 1])
    # p2 = FieldPoly(q, [1, 1])
    # div = p1 % p2
    # r = p1 % p2

    # x_l = (
    #     (240, 120, 60, 30, 15),
    #     (360, 300, 270, 255, 180, 150, 135, 90, 75, 45),
    #     (315, 285, 270, 225, 210, 180, 105, 90, 60, 0),
    #     (450, 435, 405, 345, 225),
    #     (0, )
    # )
    # y_l = (0, 0, 105, 0, 465)
    # coefs = [1]
    # for x, y in zip(x_l, y_l):
    #     p1 = FieldPoly(q, [0] * (max(x)+1))
    #     for i in x:
    #         p1.coefs[i] = 1
    #     #                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
    #     x = 75
    #     p2 = FieldPoly(q, [0] * y + [1])
    #     coefs.insert(0, (p2 * p1).simplify_power(field.xi_powers))
    # print(coefs)

    # 1 = x^5 + x^4 + x^3 + x + 1
    m1 = FieldPoly(q, [1, 1, 0, 1, 1, 1])
    # 2 = x^5 + x^2 + 1
    m2 = FieldPoly(q, [1, 0, 1, 0, 0, 1])
    # 3 = x^5 + x^4 + x^3 + x^2 + 1
    m3 = FieldPoly(q, [1, 0, 1, 1, 1, 1])
    # 4 = x^5 + x^3 + x^2 + x + 1
    m4 = FieldPoly(q, [1, 1, 1, 1, 0, 1])
    # 5 = x^5 + x^3 + 1
    m5 = FieldPoly(q, [1, 0, 0, 1, 0, 1])
    # 6 = x^5 + x^4 + x^2 + x + 1
    m6 = FieldPoly(q, [1, 1, 1, 0, 1, 1])
    for xi_pow in range(1, 31):
        for i, poly in enumerate([m1, m2, m3, m4, m5, m6]):
            xi = FieldPoly(q, [0] * xi_pow + [1])
            result = poly.calculate(xi)
            simplified_res = result.simplify_power(field.xi_powers)
            if simplified_res == 0:
                print(xi_pow, i, poly, xi)
                break
        else:
            print(xi_pow, "skip")

    print(" - Минимальный многочлен")
    g = m1 * m2
    print(m1, m2)
    print(g)

    corrected_word = "0101110101011001010001010011110"
    word_poly = FieldPoly(q, [int(s) for s in corrected_word])
    print(word_poly)

    word_array = np.array(
        [[FieldPoly(q, [int(s)])] for s in corrected_word], dtype=object
    )
    word_vector = FieldMatrix(q, word_array)
    check_m = field.get_check_matrix(d=d, n=n)
    sindrom = check_m * word_vector
    sindrom = sindrom.simplify_power(field.xi_powers)
    print(" - Синдром\n", sindrom)

    q = FieldPoly(q, word_poly.div(g)[0]) * 1

    print("".join(map(str, map(int, q.coefs))))
