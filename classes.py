from __future__ import annotations
from pprint import pprint
from typing import List, Optional, AnyStr
from numpy.polynomial import Polynomial


class ExpandedField:
    def __init__(self, expanding_poly: FieldPoly) -> None:
        self.expanding_poly = expanding_poly
        self.q = expanding_poly.q
        self.m = len(self.expanding_poly.coefs) - 1

        self.xi_powers = []
        self.calculate_basic_xi_powers()

    def calculate_basic_xi_powers(self):
        max_pow_decomp = -self.expanding_poly
        max_pow_decomp.coefs = max_pow_decomp.coefs[:-1]
        basic_power_count = len(max_pow_decomp.coefs)

        for i in range(basic_power_count):
            array = [0] * (i + 1)
            array[i] = 1
            self.xi_powers.append(FieldPoly(self.q, array))
        self.xi_powers.append(max_pow_decomp)

    def simplify_xi(self, poly: FieldPoly) -> FieldPoly:
        new_poly = poly.copy()
        for power, coef in reversed(list(enumerate(poly.coefs))):
            if power < self.m:
                break
            xi_pow_decomp = self.xi_powers[power].copy()
            xi_pow_decomp.coefs += [-1]
            xi_pow_decomp *= coef
            new_poly += xi_pow_decomp

        return new_poly

    def calculate_xi_powers(self):
        just_xi = FieldPoly(self.q, [0, 1])
        last_xi = self.xi_powers[-1]

        while True:
            last_xi *= just_xi
            last_xi = self.simplify_xi(last_xi)

            if last_xi.coefs == [1]:
                break

            self.xi_powers.append(last_xi)


class FieldPoly:
    def __init__(self, q: int, coefs: Optional[List] = None, letter: AnyStr = "x"):
        if coefs is None:
            self.coefs = [0]
        else:
            self.coefs = coefs

        self.q = q
        self.letter = "x"

    def __add__(self, other: FieldPoly):
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

        return FieldPoly(self.q, new_coefs)

    def __neg__(self):
        new_coefs = self.coefs.copy()
        for i, coeff in enumerate(new_coefs):
            new_coefs[i] = -coeff % self.q

        return FieldPoly(self.q, new_coefs)

    def __sub__(self, other: FieldPoly):
        return self + -other

    def __mul__(self, other: FieldPoly | int):
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
                    new_coefs[i + j] = (new_coefs[i + j] + a[i] * b[j]) % q

        self.__truncate_coefs(new_coefs)
        return FieldPoly(self.q, new_coefs)

    def __pow__(self, p: int):
        if p < 0:
            raise ValueError("power must be >= 0")

        new_poly = FieldPoly(self.q, [1])
        for i in range(p):
            new_poly *= self
        return new_poly

    def __eq__(self, other: FieldPoly):
        a = self.coefs
        b = other.coefs

        if len(a) != len(b):
            return False

        return all(a[i] == b[i] for i in range(len(a)))

    @staticmethod
    def __truncate_coefs(coefs: list[int]):
        while len(coefs) > 1 and coefs[-1] == 0:
            coefs.pop()
        return coefs

    def __repr__(self):
        terms = []
        for power, coeff in reversed(list(enumerate(self.coefs))):
            if coeff:
                coeff_pref = coeff if coeff > 1 else ""
                if power == 0:
                    terms.append(f"{coeff}")
                elif power == 1:
                    terms.append(f"{coeff_pref}{self.letter}")
                else:
                    terms.append(f"{coeff_pref}{self.letter}^{power}")
        result = " + ".join(terms) if terms else "0"

        return result

    def copy(self):
        return FieldPoly(self.q, self.coefs.copy(), self.letter)


if __name__ == "__main__":
    # # word = "12201010"
    # # t = 2
    # q = 3
    # init_poly = [2, 1, 1]

    # word = "0101110101011001110001010111110"
    # # word = "0101110101011001010001010011110"
    # t=2
    init_poly = [1, 1, 0, 1, 1, 1]
    q = 2

    expanding_poly = FieldPoly(q, init_poly)
    field = ExpandedField(expanding_poly)

    print(field.xi_powers)
    # p1 = FieldPoly(q, [0, 0, 0, 0, 0, 1])
    # print(p1)
    # print(field.simplify_xi(p1))

    field.calculate_xi_powers()
    pprint(field.xi_powers)
