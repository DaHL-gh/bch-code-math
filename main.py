import numpy as np


def generate_table(initial_polynom: list[int]):
    table = []

    for i in range(len(initial_polynom) - 1):
        to_append = np.zeros(len(initial_polynom) - 1, dtype=np.uint16)
        to_append[i] = 1
        table.append(to_append)
    table.append(np.array(initial_polynom[:-1], dtype=np.uint16))

    return table


def calculate_all_xi_powers(
    initial_powers: list[np.ndarray], q: int
) -> list[np.ndarray]:
    xi_powers = initial_powers.copy()

    max_xi_deg = len(xi_powers) - 2
    power_index = len(xi_powers)

    one_poly = np.zeros(max_xi_deg + 1)
    one_poly[0] = 1

    while True:
        current_poly = np.astype(xi_powers[power_index - 1], np.uint16, copy=True)

        # умножение на xi
        highest_coeff = current_poly[max_xi_deg]
        for i in range(max_xi_deg, 0, -1):
            current_poly[i] = current_poly[i - 1]
        current_poly[0] = 0

        # упрощение степеней
        reduced_poly = current_poly + xi_powers[max_xi_deg + 1] * highest_coeff

        # приведение к значениям поля
        for i in range(len(reduced_poly)):
            reduced_poly[i] %= q

        # xi^power_index == 1
        if np.array_equal(reduced_poly, one_poly):
            break

        xi_powers.append(reduced_poly)
        power_index += 1

    return xi_powers


def multiply_poly(
    powers_table: list[np.ndarray], A: np.ndarray, B: np.ndarray, q: int
) -> np.ndarray:
    n = len(A)
    m = len(B)
    new_poly = np.zeros(n, dtype=np.uint16)

    for i in range(n):
        for j in range(m):
            new_poly += powers_table[(i + j) % len(powers_table)] * A[i] * B[j]

    for i in range(len(new_poly)):
        new_poly[i] %= q

    return new_poly


def multiply_matrices(
    powers_table: list[np.ndarray], A: np.ndarray, B: np.ndarray, q: int, m
) -> np.ndarray:
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"{A.shape} and {B.shape} matrices could not be mulitiplied")

    width = B.shape[1]
    height = A.shape[0]
    depth = A.shape[1]

    result_matrix = np.zeros((A.shape[0], B.shape[1], m))

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                result_matrix[y][x] = sum_poly(
                    multiply_poly(powers_table, A[y][z], B[z][x], q),
                    result_matrix[y][x],
                    q,
                )

    return result_matrix


def transpose_matrix(A: np.ndarray):
    return np.transpose(A, axes=(1, 0, 2))


def sum_poly(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    new_poly = A + B

    for i in range(len(new_poly)):
        new_poly[i] %= q

    return np.astype(new_poly, np.uint16)


def print_polynoms_in_latex(polynoms: list[np.ndarray]):
    for i, coeffs in enumerate(polynoms):
        left = "\\xi^{" + str(i) + "}"
        right = polynom_to_latex_str(coeffs)
        print(f"${left} = {right}$")


def polynom_to_latex_str(polynom: np.ndarray) -> str:
    terms = []
    for power, coeff in reversed(list(enumerate(polynom))):
        if coeff:
            if power == 0:
                terms.append("1")
            elif power == 1:
                terms.append("\\xi")
            else:
                terms.append(f"\\xi^{power}")
    return " + ".join(terms) if terms else "0"


def print_poly_matrix_in_latex(matrix: np.ndarray):
    print(r"\begin{pmatrix}")
    x, y, _ = matrix.shape
    for j in range(y):
        values = []
        for i in range(x):
            values.append(polynom_to_latex_str(matrix[i][j]))
        print(" & ".join(values) + r" \\")
    print(r"\end{pmatrix}")


def convert_word_to_vector(word: str, m: int):
    vector = np.ndarray((1, len(word), m), dtype=np.uint16)

    for i, symbol in enumerate(word):
        poly = np.zeros(m)
        poly[0] = int(symbol)
        vector[0][i] = poly

    return vector


def get_check_matrix(powers_table: list[np.ndarray], n, d, m) -> np.ndarray:
    matrix = np.ones((d - 1, n, m), dtype=np.uint16)
    for y in range(d - 1):
        for x in range(n - 1):
            matrix[y][x + 1] = powers_table[(x * y) % len(powers_table)]

    return matrix


if __name__ == "__main__":
    initial_polynom = [1, 1, 0, 1, 1, 1]
    initial_powers_table = generate_table(initial_polynom)

    powers_table = calculate_all_xi_powers(initial_powers_table, q=2)

    # print_polynoms_in_latex(full_powers_table)
    f = np.array([0, 0, 1, 0, 0], dtype=np.uint16)
    s = np.array([0, 0, 0, 0, 1], dtype=np.uint16)

    # print(multiply_poly(powers_table, f, s, 2))

    # print_check_matrix_in_latex(powers_table, n=31, d=5)

    vector = convert_word_to_vector("0101110101011001110001010111110", m=5)
    vector_t = transpose_matrix(vector)

    check_matrix = get_check_matrix(powers_table, n=31, d=5, m=5)

    sindrom_t = multiply_matrices(powers_table, check_matrix, vector_t, q=2, m=5)
    sindrom = transpose_matrix(sindrom_t)

    # print_poly_matrix_in_latex(sindrom)
