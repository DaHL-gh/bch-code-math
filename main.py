import enum
import itertools
import numpy as np


class InFieldOperator:
    def __init__(self, q, m):
        self.q = q
        self.m = m
        self.xi_powers = []

    def calculate_xi_powers(self, initial_polynom: list[int]):
        xi_powers = self.generate_table(initial_polynom)

        max_xi_deg = len(xi_powers) - 2
        power_index = len(xi_powers)

        one_poly = np.zeros(max_xi_deg + 1)
        one_poly[0] = 1

        while True:
            current_poly = np.astype(xi_powers[power_index - 1], np.int16, copy=True)

            # умножение на xi
            highest_coeff = current_poly[max_xi_deg]
            for i in range(max_xi_deg, 0, -1):
                current_poly[i] = current_poly[i - 1]
            current_poly[0] = 0

            # упрощение степеней
            reduced_poly = current_poly + xi_powers[max_xi_deg + 1] * highest_coeff

            # приведение к значениям поля
            reduced_poly %= self.q

            # xi^power_index == 1
            if np.array_equal(reduced_poly, one_poly):
                break

            xi_powers.append(reduced_poly)
            power_index += 1

        self.xi_powers = xi_powers

    def generate_table(self, initial_polynom: list[int]):
        table = []

        for i in range(len(initial_polynom) - 1):
            to_append = np.zeros(len(initial_polynom) - 1, dtype=np.int16)
            to_append[i] = 1
            table.append(to_append)
        table.append(self.negate_poly(np.array(initial_polynom[:-1], dtype=np.int16)))

        return table

    def negate_poly(self, A: np.ndarray) -> np.ndarray:
        return (-A) % self.q

    def get_invert_poly(self, poly: np.ndarray):
        for i in range(len(self.xi_powers)):
            if all(poly == self.xi_powers[i]):
                return self.xi_powers[-i % len(self.xi_powers)]
        raise ValueError(
            f"No invert poly found for specified poly:  {polynom_to_latex_str(poly)}"
        )

    def power_poly(self, poly: np.ndarray, p):
        for i in range(len(self.xi_powers)):
            if all(poly == self.xi_powers[i]):
                return self.xi_powers[(i * p) % len(self.xi_powers)]
        else:
            return np.zeros(self.m)

    def sum_poly(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A + B) % self.q

    def is_zero_poly(self, poly: np.ndarray):
        return all(poly == 0)

    def multiply_poly(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        m = len(B)
        new_poly = np.zeros(n, dtype=np.int16)

        for i in range(n):
            for j in range(m):
                new_poly += np.astype(
                    self.xi_powers[(i + j) % len(self.xi_powers)] * A[i] * B[j],
                    np.uint16,
                )

        new_poly %= self.q

        return new_poly

    def negate_matrix(self, A: np.ndarray) -> np.ndarray:
        negate_vectorized = np.vectorize(self.negate_poly)
        return negate_vectorized(A)

    def multiply_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"{A.shape} and {B.shape} matrices could not be mulitiplied"
            )

        width = B.shape[1]
        height = A.shape[0]
        depth = A.shape[1]

        result_matrix = np.zeros((A.shape[0], B.shape[1], self.m))

        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    result_matrix[y][x] = self.sum_poly(
                        self.multiply_poly(A[y][z], B[z][x]), result_matrix[y][x]
                    )

        return result_matrix

    @staticmethod
    def transpose_matrix(A: np.ndarray):
        return np.transpose(A, axes=(1, 0, 2))

    def matrix_determinant(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix should be square")

        n = matrix.shape[0]

        # 1x1
        if n == 1:
            return matrix[0, 0]

        # 2x2
        if n == 2:
            return self.sum_poly(
                self.multiply_poly(matrix[0, 0], matrix[1, 1]),
                self.negate_poly(self.multiply_poly(matrix[0, 1], matrix[1, 0])),
            )
            # return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

        # else
        det = np.zeros(self.m)
        for col in range(n):
            minor = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
            cofactor = matrix[0, col] * self.matrix_determinant(minor)

            if col % 2 != 0:
                self.negate_poly(cofactor)

            det = self.sum_poly(det, cofactor)

        return det

    def get_identity_matrix(self, n):
        eye = np.zeros((n, n, self.m), dtype=object)
        for i in range(n):
            eye[i][i][0] = 1
        return eye

    def matrix_inverse_gauss_jordan(self, matrix):
        n = matrix.shape[0]
        A = matrix.copy()
        eye = self.get_identity_matrix(n)

        for i in range(n):
            pivot = A[i, i]
            inv_pivot = self.get_invert_poly(pivot)

            for j in range(n):
                A[i, j] = self.multiply_poly(A[i, j], inv_pivot)
                eye[i, j] = self.multiply_poly(eye[i, j], inv_pivot)

            for k in range(n):
                if k != i:
                    neg_factor = self.negate_poly(A[k, i])
                    for j in range(n):
                        A[k, j] = self.sum_poly(
                            A[k, j], self.multiply_poly(neg_factor, A[i, j])
                        )
                        eye[k, j] = self.sum_poly(
                            eye[k, j], self.multiply_poly(neg_factor, eye[i, j])
                        )
        return eye

    def get_check_matrix(self, n, d) -> np.ndarray:
        matrix = np.zeros((d - 1, n, self.m), dtype=np.uint16)
        for y in range(d - 1):
            for x in range(n):
                matrix[y][x] = self.xi_powers[(x * (y + 1)) % len(self.xi_powers)]

        return matrix

    def get_gilberts_matrix(self, S: np.ndarray, r: int) -> np.ndarray:
        gilberts_matrix = np.zeros((r, r, self.m))

        for y in range(r):
            for x in range(r):
                gilberts_matrix[y][x] = S[0][r - (x + 1) + y]

        return gilberts_matrix

    def convert_word_to_vector(self, word: str):
        vector = np.zeros((1, len(word), self.m), dtype=np.uint16)

        for i, symbol in enumerate(word):
            vector[0][i][0] = int(symbol)

        return vector


def print_polynoms_in_latex(polynoms: list[np.ndarray]):
    for i, coeffs in enumerate(polynoms):
        left = "\\xi^{" + str(i) + "}"
        right = polynom_to_latex_str(coeffs)
        print(f"${left} = {right}$")


def polynom_to_latex_str(polynom: np.ndarray, is_inside_mk=False) -> str:
    terms = []
    for power, coeff in reversed(list(enumerate(polynom))):
        if coeff:
            coeff_pref = coeff if coeff > 1 else ""
            if power == 0:
                terms.append(f"{coeff}")
            elif power == 1:
                terms.append(f"{coeff_pref}\\xi")
            else:
                terms.append(f"{coeff_pref}\\xi^{power}")
    result = " + ".join(terms) if terms else "0"
    if is_inside_mk:
        return "$" + result + "$"
    return result


def print_poly_matrix_in_latex(matrix: np.ndarray, is_inside_dm=True):
    if is_inside_dm:
        print("$$")
    print(r"\begin{pmatrix}")
    y, x, _ = matrix.shape
    for j in range(y):
        values = []
        for i in range(x):
            values.append(polynom_to_latex_str(matrix[j][i]))
        print(" & ".join(values) + r" \\")
    print(r"\end{pmatrix}")
    if is_inside_dm:
        print("$$")


def pipeline(initial_poly, q, t, word):
    print(f"Дано $f(x) = {polynom_to_latex_str(initial_poly)}$")
    n = len(word)
    d = 2 * t + 1
    m = len(initial_poly) - 1
    print(f"q={q}, m={m}, t={t}, d={d}, n={n}")
    print(f"Слово для расшифровки ${word}$")

    print(r" - Степени \xi")
    operator = InFieldOperator(q=q, m=m)
    operator.calculate_xi_powers(initial_polynom)

    for i, coeffs in enumerate(operator.xi_powers):
        left = "\\xi^{" + str(i) + "}"
        right = polynom_to_latex_str(coeffs)
        print(f"${left} = {right}$")

    print(" - Проверочная матрица $H=$")
    check_matrix = operator.get_check_matrix(n=n, d=d)
    print_poly_matrix_in_latex(check_matrix)

    print(" - Синдром")

    vector = operator.convert_word_to_vector(word)
    vector_t = operator.transpose_matrix(vector)

    sindrom = operator.multiply_matrices(check_matrix, vector_t)
    sindrom_t = operator.transpose_matrix(sindrom)

    print_poly_matrix_in_latex(sindrom)

    r = t
    while r > 0:
        g_m = operator.get_gilberts_matrix(sindrom_t, r)

        if any(operator.matrix_determinant(g_m) != 0):
            break

        r -= 1
    print(f" - Ошибок {r}")
    if r == 0:
        print("В слове нет ошибок")
        return

    print(f" - Матрица синдромов размера {r}")
    print_poly_matrix_in_latex(g_m)
    print(f"Её определитель ${polynom_to_latex_str(operator.matrix_determinant(g_m))}$")
    inverse_g_m = operator.matrix_inverse_gauss_jordan(g_m)

    print(r" - Локаторы ошибок $\tau$")
    some_sindromes = np.zeros((r, 1, m), dtype=np.uint16)
    for i in range(r):
        some_sindromes[i][0] = sindrom[i + r][0]
    neg_some_sindrom = operator.negate_matrix(some_sindromes)

    print("$$")
    print_poly_matrix_in_latex(neg_some_sindrom, False)
    print(r"\times")
    print_poly_matrix_in_latex(inverse_g_m, False)
    print("=")
    tau = operator.multiply_matrices(inverse_g_m, neg_some_sindrom)
    print_poly_matrix_in_latex(tau, False)
    print("$$")

    print(r" - Вычисления локатора для каждого $\xi$")

    def get_locator(tau):
        def locator(x):
            loc = np.zeros(m)
            loc[0] = 1
            for i in range(len(tau)):
                loc = operator.sum_poly(
                    loc,
                    operator.multiply_poly(tau[i][0], operator.power_poly(x, i + 1)),
                )
            return loc

        return locator

    locator = get_locator(tau)

    wrong_bits = []
    for i in range(len(operator.xi_powers)):
        locator_result = locator(operator.xi_powers[i])
        print(rf"$\sigma(\xi^{{{i}}})={polynom_to_latex_str(locator_result)}$")
        if all(locator_result == 0):
            wrong_bits.append(-i % len(operator.xi_powers))

    print(" - Найденные позиции ошибок:", *wrong_bits)

    print(" - Вектор ошибки")
    error_vec = np.zeros((1, n, m), dtype=np.uint16)
    if q == 2:
        for i in wrong_bits:
            error_vec[0][i][0] = 1
    else:
        possible_bits = [i + 1 for i in range(q - 1)]
        for vec_bits in itertools.product(possible_bits, repeat=len(wrong_bits)):
            for i in range(len(vec_bits)):
                error_vec[0][wrong_bits[i]][0] = vec_bits[i]

            error_vec_t = operator.transpose_matrix(error_vec)

            mb_sindrome = operator.multiply_matrices(check_matrix, error_vec_t)
            if (mb_sindrome == sindrom).all():
                break
        else:
            raise ValueError(
                "Ни одна комбинация вектора ошибки не дает синдром (такого не может быть(ошибка вычислений))"
            )
    for i in range(error_vec.shape[1]):
        print(error_vec[0][i][0], end="")
    print()

    
    print(" - Восстановленное слово")
    fixed_word = operator.sum_poly(vector, operator.negate_matrix(error_vec))

    for i in range(fixed_word.shape[1]):
        print(fixed_word[0][i][0], end="")

    print(" - ")


if __name__ == "__main__":
    initial_polynom = [1, 1, 0, 1, 1, 1]
    word = "0101110101011001110001010111110"
    # word = "0101110101011001010001010011110"
    q=2
    t=2

    # initial_polynom = [2, 1, 1]
    # word = "12201010"
    # q = 3
    # t = 2

    pipeline(initial_polynom, q=q, t=t, word=word)
