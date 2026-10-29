import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


MAX_ITERATIONS = 10000
GRID_POINTS = 1000
PLOT_FILE = "plot_2_1.png"


def f(x):
    return 2 ** x + x * x - 2


def df(x):
    return 2 ** x * math.log(2) + 2 * x


def d2f(x):
    return 2 ** x * math.log(2) ** 2 + 2


def grid(a, b):
    points = []
    for i in range(GRID_POINTS + 1):
        points.append(a + (b - a) * i / GRID_POINTS)
    return points


def newton(x0, eps):
    x = x0
    history = [x]
    for k in range(1, MAX_ITERATIONS + 1):
        if abs(df(x)) < 1e-12:
            raise ValueError(f"f'(x) = 0 на итерации {k}: метод Ньютона не может продолжить")
        x_new = x - f(x) / df(x)
        history.append(x_new)
        if abs(x_new - x) <= eps:
            return x_new, k, history
        x = x_new
    raise ValueError(f"метод Ньютона не сошёлся за {MAX_ITERATIONS} итераций")


def simple_iteration(phi, x0, eps, q):
    x = x0
    history = [x]
    for k in range(1, MAX_ITERATIONS + 1):
        x_new = phi(x)
        history.append(x_new)
        if q / (1 - q) * abs(x_new - x) <= eps:
            return x_new, k, history
        x = x_new
    raise ValueError(f"метод простой итерации не сошёлся за {MAX_ITERATIONS} итераций")


def print_history(history, root):
    print("   k            x(k)    |x(k) - x(k-1)|      |x(k) - x*|")
    for k in range(len(history)):
        step = f"{abs(history[k] - history[k - 1]):18.3e}" if k > 0 else " " * 18
        print(f"{k:4d} {history[k]:15.10f} {step} {abs(history[k] - root):16.3e}")


def plot(a, b):
    xs = grid(-2, 2)
    plt.plot(xs, [2 ** x for x in xs], label="y = 2^x")
    plt.plot(xs, [2 - x * x for x in xs], label="y = 2 - x^2")
    plt.axvspan(a, b, alpha=0.2, label=f"[{a:g}, {b:g}]")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.ylim(-2, 4)
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILE)


def main():
    print(f"Данные: {INPUT_2_1}")
    rows = read_rows(INPUT_2_1)
    if len(rows) != 2 or len(rows[1]) != 2:
        raise ValueError("в файле должно быть: eps, затем строка с концами отрезка a b")
    eps = rows[0][0]
    a, b = rows[1]
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")
    if a >= b:
        raise ValueError("нужно a < b")

    print("\nУравнение: 2^x + x^2 - 2 = 0")
    print(f"eps = {eps:g}")

    plot(a, b)
    print(f"график сохранён в {PLOT_FILE}: корень — пересечение y = 2^x и y = 2 - x^2")

    print(f"\nотрезок [{a:g}, {b:g}]: f(a) = {f(a):.6f}, f(b) = {f(b):.6f}")
    if f(a) * f(b) >= 0:
        raise ValueError("на концах отрезка f одного знака: корень на нём не отделён")

    root, _, _ = newton(b if f(b) * d2f(b) > 0 else a, 1e-12)

    print("\n=== Метод простой итерации ===")
    derivatives = [df(x) for x in grid(a, b)]
    m, M = min(derivatives), max(derivatives)
    if m * M <= 0:
        raise ValueError("f'(x) меняет знак на отрезке: возьмите отрезок уже")
    lam = 2 / (m + M)
    phi = lambda x: x - lam * f(x)
    print(f"x = phi(x) = x - lambda * f(x),  min f' = {m:.6f}, max f' = {M:.6f}, lambda = 2 / (min + max) = {lam:.6f}")
    q = max(abs(1 - lam * d) for d in derivatives)
    print(f"q = max |phi'(x)| = max |1 - lambda * f'(x)| = {q:.6f}")
    if q >= 1:
        raise ValueError("q >= 1: условие сходимости метода простой итерации не выполнено")
    inside = True
    for x in grid(a, b):
        if not a <= phi(x) <= b:
            inside = False
    if inside:
        print(f"q < 1 и phi(x) лежит в [{a:g}, {b:g}]: условия теоремы 2.3 выполнены, метод сходится")
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: phi(x) выходит за [{a:g}, {b:g}], условие теоремы 2.3 не выполнено")

    x0 = (a + b) / 2
    print(f"x(0) = {x0:g},  останов: q / (1 - q) * |x(k) - x(k-1)| <= eps")
    x_simple, it_simple, hist_simple = simple_iteration(phi, x0, eps, q)
    print_history(hist_simple, root)
    print(f"корень: {x_simple:.10f}, итераций: {it_simple}")

    print("\n=== Метод Ньютона ===")
    if f(b) * d2f(b) > 0:
        x0 = b
    elif f(a) * d2f(a) > 0:
        x0 = a
    else:
        raise ValueError("ни на одном конце отрезка не выполнено f(x0) * f''(x0) > 0")
    print(f"x(0) = {x0:g}: f(x0) * f''(x0) = {f(x0) * d2f(x0):.6f} > 0")
    print("останов: |x(k) - x(k-1)| <= eps")
    x_newton, it_newton, hist_newton = newton(x0, eps)
    print_history(hist_newton, root)
    print(f"корень: {x_newton:.10f}, итераций: {it_newton}")

    print("\n=== Сравнение ===")
    print(f"x* = {root:.12f} (метод Ньютона до 1e-12) — для столбца |x(k) - x*|")
    print(f"метод простой итерации: {it_simple} итераций, погрешность {abs(x_simple - root):.2e}")
    print(f"метод Ньютона:          {it_newton} итераций, погрешность {abs(x_newton - root):.2e}")


if __name__ == "__main__":
    main()
