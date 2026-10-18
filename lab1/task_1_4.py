import math
import numpy as np
from common import *


MAX_ROTATIONS = 10000


def check_symmetric(A):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-12:
                raise ValueError(f"матрица не симметрична: a_{i + 1}{j + 1} = {A[i][j]}, "
                                 f"a_{j + 1}{i + 1} = {A[j][i]}. Метод вращений применим только "
                                 f"к симметричным матрицам")


def transpose(M):
    n = len(M)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i][j] = M[j][i]
    return T


def multiply(X, Y):
    n = len(X)
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += X[i][k] * Y[k][j]
            Z[i][j] = s
    return Z


def max_offdiagonal(A):
    n = len(A)
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > abs(A[best_i][best_j]):
                best_i, best_j = i, j
    return best_i, best_j


def offdiagonal_measure(A):
    n = len(A)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += A[i][j] ** 2
    return math.sqrt(s)


def rotation_matrix(n, i, j, phi):
    U = np.eye(n)
    U[i][i] = math.cos(phi)
    U[j][j] = math.cos(phi)
    U[i][j] = -math.sin(phi)
    U[j][i] = math.sin(phi)
    return U


def jacobi_rotations(A, eps):
    n = len(A)
    A_k = A.astype(float).copy()
    V = np.eye(n)
    t = offdiagonal_measure(A_k)
    rotations = 0

    while t >= eps:
        if rotations >= MAX_ROTATIONS:
            raise ValueError(f"не сошлось за {MAX_ROTATIONS} вращений: t(A) = {t:.2e} не опускается "
                             f"ниже eps = {eps:g}. Скорее всего, eps меньше точности компьютера")

        i, j = max_offdiagonal(A_k)

        if A_k[i][i] == A_k[j][j]:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan(2 * A_k[i][j] / (A_k[i][i] - A_k[j][j]))

        U = rotation_matrix(n, i, j, phi)
        A_k = multiply(multiply(transpose(U), A_k), U)
        V = multiply(V, U)

        rotations += 1
        t = offdiagonal_measure(A_k)

    values = []
    for i in range(n):
        values.append(A_k[i][i])
    return values, V, rotations, t


def main():
    print(f"Данные: {INPUT_1_4}")
    rows = read_rows(INPUT_1_4)
    n = int(rows[0][0])
    if len(rows) != n + 2:
        raise ValueError("в файле должно быть: n, eps, затем n строк матрицы A")
    eps = rows[1][0]
    A = np.array(rows[2:n + 2])
    if A.shape != (n, n):
        raise ValueError("нужна квадратная матрица A")
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")
    check_symmetric(A)

    print_matrix("Матрица A (симметричная)", A)

    print("\n=== 1. Заданная точность ===")
    print(f"eps = {eps:g}")
    print("критерий остановки: t(A) = корень из суммы квадратов элементов над диагональю < eps")

    values, V, rotations, t = jacobi_rotations(A, eps)

    print(f"\nвыполнено вращений: {rotations}, итоговое t(A) = {t:.2e} < eps")

    print("\n=== 2. Собственные значения ===")
    for i in range(n):
        print(f"lambda_{i + 1} = {values[i]:.10f}")
    trace = 0.0
    total = 0.0
    for i in range(n):
        trace += A[i][i]
        total += values[i]
    print(f"проверка: сумма lambda = {total:.8f}, след A = {trace:.8f} (должны совпадать)")

    print("\n=== 3. Собственные векторы и матрица V ===")
    for i in range(n):
        column = []
        for r in range(n):
            column.append(V[r][i])
        print_vector(f"x_{i + 1}", column)
    print_matrix("V (столбцы — собственные векторы)", V)
    print(f"max|V^T*V - E| = {np.max(np.abs(V.T @ V - np.eye(n))):.2e}  (векторы ортонормированы)")

    print("\n=== 4. Проверка A*V = V*Lambda ===")
    Lambda = np.zeros((n, n))
    for i in range(n):
        Lambda[i][i] = values[i]
    print_matrix("A*V", A @ V)
    print_matrix("V*Lambda", V @ Lambda)
    print(f"max|A*V - V*Lambda| = {np.max(np.abs(A @ V - V @ Lambda)):.2e}")


if __name__ == "__main__":
    main()
