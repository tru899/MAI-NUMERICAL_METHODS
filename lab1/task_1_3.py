import math
import numpy as np
from common import *
from task_1_1 import lu_decompose, lu_solve


MAX_ITERATIONS = 10000


def to_iteration_form(A, b):
    n = len(A)
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        if abs(A[i][i]) < 1e-12:
            raise ValueError(f"a_{i + 1}{i + 1} = 0: выразить x_{i + 1} из уравнения {i + 1} нельзя. "
                             f"Переставьте уравнения в файле так, чтобы на диагонали не было нулей ")
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if j != i:
                alpha[i][j] = -A[i][j] / A[i][i]
    return alpha, beta


def upper_part(alpha):
    n = len(alpha)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C[i][j] = alpha[i][j]
    return C


def matrix_norm(M):
    result = 0.0
    for row in M:
        s = 0.0
        for value in row:
            s += abs(value)
        result = max(result, s)
    return result


def distance(x, y):
    result = 0.0
    for i in range(len(x)):
        result = max(result, abs(x[i] - y[i]))
    return result


def all_finite(x):
    for value in x:
        if not math.isfinite(value):
            return False
    return True


def simple_iteration(alpha, beta, eps, factor):
    n = len(beta)
    x = beta.copy()
    history = []
    for k in range(1, MAX_ITERATIONS + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += alpha[i][j] * x[j]
            x_new[i] = beta[i] + s
        if not all_finite(x_new):
            raise ValueError(f"метод простых итераций расходится: на итерации {k} значения ушли в бесконечность")
        estimate = factor * distance(x_new, x)
        history.append(estimate)
        x = x_new
        if estimate <= eps:
            return x, k, history
    raise ValueError(f"метод простых итераций не сошёлся за {MAX_ITERATIONS} итераций (расходится или сходится слишком медленно)")


def seidel(alpha, beta, eps, factor):
    n = len(beta)
    x = beta.copy()
    history = []
    for k in range(1, MAX_ITERATIONS + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(i):
                s += alpha[i][j] * x_new[j]
            for j in range(i + 1, n):
                s += alpha[i][j] * x[j]
            x_new[i] = beta[i] + s
        if not all_finite(x_new):
            raise ValueError(f"метод Зейделя расходится: на итерации {k} значения ушли в бесконечность")
        estimate = factor * distance(x_new, x)
        history.append(estimate)
        x = x_new
        if estimate <= eps:
            return x, k, history
    raise ValueError(f"метод Зейделя не сошёлся за {MAX_ITERATIONS} итераций (расходится или сходится слишком медленно)")


def main():
    print(f"Данные: {INPUT_1_3}")
    rows = read_rows(INPUT_1_3)
    n = int(rows[0][0])
    if len(rows) != n + 3:
        raise ValueError("в файле должно быть: n, eps, затем n строк матрицы A, затем строка b")
    eps = rows[1][0]
    A = np.array(rows[2:n + 2])
    b = rows[n + 2]
    if A.shape != (n, n) or len(b) != n:
        raise ValueError("нужна квадратная матрица A и вектор b той же размерности")
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")

    print_matrix("Матрица A", A)
    print_vector("Правая часть b", b)

    print("\n=== 1. Заданная точность ===")
    print(f"eps = {eps:g}")

    print("\n=== Приведение к виду x = alpha*x + beta ===")
    alpha, beta = to_iteration_form(A, b)
    print_matrix("alpha", alpha)
    print_vector("beta", beta)

    norm_alpha = matrix_norm(alpha)
    norm_C = matrix_norm(upper_part(alpha))
    print(f"||alpha|| = {norm_alpha:.6f},  ||C|| = {norm_C:.6f}")

    if norm_alpha < 1:
        factor_simple = norm_alpha / (1 - norm_alpha)
        factor_seidel = norm_C / (1 - norm_alpha)
        print("||alpha|| < 1: достаточное условие сходимости выполнено, оба метода сходятся.")
        print(f"Останов: {factor_simple:.4f} * ||x(k) - x(k-1)|| <= eps  (простые итерации)")
        print(f"         {factor_seidel:.4f} * ||x(k) - x(k-1)|| <= eps  (Зейдель)")
    else:
        factor_simple = 1.0
        factor_seidel = 1.0
        print("ПРЕДУПРЕЖДЕНИЕ: ||alpha|| >= 1 — достаточное условие сходимости не выполнено.")
        print("Считаем дальше, останов по ||x(k) - x(k-1)|| <= eps; достижение точности eps не гарантируется.")

    x_simple, it_simple, hist_simple = simple_iteration(alpha, beta, eps, factor_simple)
    x_seidel, it_seidel, hist_seidel = seidel(alpha, beta, eps, factor_seidel)

    print("\n=== 2. Количество итераций ===")
    print(f"метод простых итераций: {it_simple}")
    print(f"метод Зейделя:          {it_seidel}")

    print("\n=== 3. Решения и эталон ===")
    L, U, perm, _ = lu_decompose(A)
    x_exact = lu_solve(L, U, perm, b)
    print_vector("простые итерации", x_simple)
    print_vector("Зейдель         ", x_seidel)
    print_vector("эталон (LU, 1.1)", x_exact)
    error_simple = distance(x_simple, x_exact)
    error_seidel = distance(x_seidel, x_exact)
    print(f"фактическая погрешность ||x - x_эталон||: простые итерации {error_simple:.2e}, "
          f"Зейдель {error_seidel:.2e}")
    if error_simple <= eps and error_seidel <= eps:
        print("обе погрешности не больше eps — заданная точность достигнута")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: погрешность больше eps — заданная точность не достигнута")

    print("\n=== 4. Сравнение скорости сходимости ===")
    print("оценка погрешности eps(k) по итерациям:")
    print("   k   простые итерации        Зейдель")
    rows = min(max(it_simple, it_seidel), 30)
    for k in range(rows):
        left = f"{hist_simple[k]:16.3e}" if k < it_simple else " " * 16
        right = f"{hist_seidel[k]:16.3e}" if k < it_seidel else ""
        print(f"{k + 1:4d} {left} {right}")
    if rows < max(it_simple, it_seidel):
        print("   … (показаны первые 30 итераций)")

    if it_seidel < it_simple:
        print(f"\nЗейдель сошёлся за {it_seidel} итераций против {it_simple} — "
              f"в {it_simple / it_seidel:.2f} раза быстрее.")
    elif it_seidel == it_simple:
        print("\nНа этих данных методы сошлись за одинаковое число итераций.")
    else:
        print(f"\nНа этих данных Зейдель оказался медленнее: {it_seidel} итераций против {it_simple}.")
    print("Зейдель использует уже пересчитанные компоненты, поэтому обычно сходится быстрее, "
          "но гарантии нет.")


if __name__ == "__main__":
    main()
