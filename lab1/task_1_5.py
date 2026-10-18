import cmath
import math
import numpy as np
from common import *
from task_1_4 import multiply


MAX_ITERATIONS = 10000


def householder_matrix(v):
    n = len(v)
    vv = 0.0
    for i in range(n):
        vv += v[i] * v[i]
    H = np.eye(n)
    for i in range(n):
        for j in range(n):
            H[i][j] -= 2 * v[i] * v[j] / vv
    return H


def householder_qr(A):
    n = len(A)
    R = A.astype(float).copy()
    Q = np.eye(n)

    for k in range(n - 1):
        norm = 0.0
        for i in range(k, n):
            norm += R[i][k] ** 2
        norm = math.sqrt(norm)
        if norm < 1e-15:
            continue

        v = np.zeros(n)
        for i in range(k, n):
            v[i] = R[i][k]
        if R[k][k] >= 0:
            v[k] += norm
        else:
            v[k] -= norm

        H = householder_matrix(v)
        R = multiply(H, R)
        Q = multiply(Q, H)

    for i in range(n):
        for j in range(i):
            R[i][j] = 0.0

    return Q, R


def below(A, j, start):
    s = 0.0
    for l in range(start, len(A)):
        s += A[l][j] ** 2
    return math.sqrt(s)


def block_eigenvalues(A, j):
    p, q = A[j][j], A[j][j + 1]
    r, s = A[j + 1][j], A[j + 1][j + 1]
    trace = p + s
    det = p * s - q * r
    root = cmath.sqrt(trace * trace - 4 * det)
    return (trace + root) / 2, (trace - root) / 2


def read_eigenvalues(A, eps, previous_blocks):
    n = len(A)
    values = []
    blocks = {}
    converged = True
    j = 0
    while j < n:
        if below(A, j, j + 1) <= eps:
            values.append(A[j][j])
            j += 1
        elif j + 1 < n and below(A, j, j + 2) <= eps:
            pair = block_eigenvalues(A, j)
            blocks[j] = pair
            old = previous_blocks.get(j)
            if old is None or abs(pair[0] - old[0]) > eps or abs(pair[1] - old[1]) > eps:
                converged = False
            values.append(pair[0])
            values.append(pair[1])
            j += 2
        else:
            converged = False
            values.append(A[j][j])
            j += 1
    return values, converged, blocks


def qr_algorithm(A, eps):
    A_k = A.astype(float).copy()
    previous_blocks = {}
    for k in range(1, MAX_ITERATIONS + 1):
        Q, R = householder_qr(A_k)
        A_k = multiply(R, Q)
        values, converged, blocks = read_eigenvalues(A_k, eps, previous_blocks)
        if converged:
            return values, A_k, k
        previous_blocks = blocks
    raise ValueError(f"QR-алгоритм не сошёлся за {MAX_ITERATIONS} итераций")


def format_value(z):
    z = complex(z)
    if abs(z.imag) < 1e-12:
        return f"{z.real:.6f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.6f} {sign} {abs(z.imag):.6f}i"


def analyse(path, title):
    print(f"\n{'=' * 70}\n{title}\nДанные: {path}\n{'=' * 70}")
    rows = read_rows(path)
    n = int(rows[0][0])
    if len(rows) != n + 2:
        raise ValueError("в файле должно быть: n, eps, затем n строк матрицы A")
    eps = rows[1][0]
    A = np.array(rows[2:n + 2])
    if A.shape != (n, n):
        raise ValueError("нужна квадратная матрица A")
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")

    print("\n=== 1. Исходная матрица A ===")
    print_matrix("A", A)

    print("\n=== 2. Матрицы Q и R ===")
    Q, R = householder_qr(A)
    print_matrix("Q", Q)
    print_matrix("R", R)

    print("\n=== 3. Произведение Q*R ===")
    print_matrix("Q*R", Q @ R)

    print("\n=== 4. Собственные значения ===")
    values, _, iterations = qr_algorithm(A, eps)
    print(f"итераций: {iterations}")
    for i in range(len(values)):
        print(f"lambda_{i + 1} = {format_value(values[i])}")


def main():
    analyse(INPUT_1_5, "Задача 1.5 — QR-разложение и QR-алгоритм")
    analyse(INPUT_1_5_COMPLEX, "Свой пример: матрица с комплексно-сопряжёнными собственными значениями")


if __name__ == "__main__":
    main()
