import numpy as np
from common import *


def lu_decompose(A):
    n = len(A)
    U = A.astype(float).copy()
    L = np.zeros((n, n))
    perm = list(range(n))
    swaps = 0

    for k in range(n - 1):
        pivot = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[pivot][k]):
                pivot = i

        if abs(U[pivot][k]) < 1e-12:
            raise ValueError(f"на шаге {k + 1} столбец ниже диагонали нулевой")

        if pivot != k:
            for j in range(n):
                U[k][j], U[pivot][j] = U[pivot][j], U[k][j]
            for j in range(k):
                L[k][j], L[pivot][j] = L[pivot][j], L[k][j]
            perm[k], perm[pivot] = perm[pivot], perm[k]
            swaps += 1

        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]
            U[i][k] = 0.0

    for i in range(n):
        L[i][i] = 1.0

    return L, U, perm, swaps


def permutation_matrix(perm):
    n = len(perm)
    P = np.zeros((n, n))
    for i in range(n):
        P[i][perm[i]] = 1.0
    return P


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = b[i] - s
    return y


def back_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x


def lu_solve(L, U, perm, b):
    n = len(b)
    pb = np.zeros(n)
    for i in range(n):
        pb[i] = b[perm[i]]
    y = forward_substitution(L, pb)
    return back_substitution(U, y)


def lu_determinant(U, swaps):
    det = (-1.0) ** swaps
    for i in range(len(U)):
        det *= U[i][i]
    return det


def lu_inverse(L, U, perm):
    n = len(U)
    inverse = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        x = lu_solve(L, U, perm, e)
        for r in range(n):
            inverse[r][i] = x[r]
    return inverse


def main():
    print(f"Данные: {INPUT_1_1}")
    rows = read_rows(INPUT_1_1)
    n = int(rows[0][0])
    if len(rows) != n + 2:
        raise ValueError("в файле должно быть: n, затем n строк матрицы A, затем строка b")
    A = np.array(rows[1:n + 1])
    b = rows[n + 1]
    if A.shape != (n, n) or len(b) != n:
        raise ValueError("нужна квадратная матрица A и вектор b той же размерности")

    print_matrix("Исходная матрица A", A)
    print_vector("Правая часть b", b)

    L, U, perm, swaps = lu_decompose(A)
    P = permutation_matrix(perm)

    print("\n=== 1. Матрицы L и U ===")
    print_matrix("L", L)
    print_matrix("U", U)
    print(f"perm = {perm}, перестановок: {swaps}")
    print_matrix("P", P)

    print("\n=== 2. Произведение L*U ===")
    print_matrix("L*U", L @ U)

    print("\n=== 3. Решение системы Ax = b ===")
    x = lu_solve(L, U, perm, b)
    print_vector("x", x)
    print(f"невязка max|Ax - b| = {np.max(np.abs(A @ x - b)):.2e}")

    print("\n=== 4. Обратная матрица через LU ===")
    A_inv = lu_inverse(L, U, perm)
    print_matrix("A^-1", A_inv)

    print("\n=== 5. Определитель через LU ===")
    det = lu_determinant(U, swaps)
    diagonal = " * ".join(f"{U[i][i]:.4f}" for i in range(n))
    print(f"det A = (-1)^{swaps} * {diagonal} = {det:.6f}")

    print("\n=== 6. Проверка A*A^-1 = E ===")
    print_matrix("A*A^-1", A @ A_inv)
    print(f"max|A*A^-1 - E| = {np.max(np.abs(A @ A_inv - np.eye(n))):.2e}")

    print("\n=== 7. Проверка L*U = P*A ===")
    print_matrix("P*A", P @ A)
    print(f"max|L*U - P*A| = {np.max(np.abs(L @ U - P @ A)):.2e}")

    print("\n=== Сверка с NumPy (только проверка) ===")
    print(f"det A у NumPy = {np.linalg.det(A):.6f}")
    print(f"max|x - x_NumPy| = {np.max(np.abs(x - np.linalg.solve(A, b))):.2e}")
    print(f"max|A^-1 - A^-1_NumPy| = {np.max(np.abs(A_inv - np.linalg.inv(A))):.2e}")


if __name__ == "__main__":
    main()
