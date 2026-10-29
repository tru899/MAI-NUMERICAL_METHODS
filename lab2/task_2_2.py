import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


MAX_ITERATIONS = 10000
GRID_POINTS = 100
PLOT_FILE = "plot_2_2.png"


def f1(x1, x2, a):
    return x1 * x1 + x2 * x2 - a * a


def f2(x1, x2, a):
    return x1 - math.exp(x2) + a


def jacobian(x1, x2):
    return [[2 * x1, 2 * x2],
            [1.0, -math.exp(x2)]]


def phi(x1, x2, a):
    if a * a - x2 * x2 <= 0:
        raise ValueError(f"phi1 не определена: a^2 - x2^2 <= 0 при x2 = {x2}")
    if x1 + a <= 0:
        raise ValueError(f"phi2 не определена: x1 + a <= 0 при x1 = {x1}")
    return math.sqrt(a * a - x2 * x2), math.log(x1 + a)


def phi_derivative(x1, x2, a):
    return [[0.0, -x2 / math.sqrt(a * a - x2 * x2)],
            [1 / (x1 + a), 0.0]]


def det2(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


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


def grid(lo, hi):
    points = []
    for i in range(GRID_POINTS + 1):
        points.append(lo + (hi - lo) * i / GRID_POINTS)
    return points


def newton(x0, a, eps):
    x1, x2 = x0
    history = [[x1, x2]]
    for k in range(1, MAX_ITERATIONS + 1):
        F = [f1(x1, x2, a), f2(x1, x2, a)]
        J = jacobian(x1, x2)
        det_J = det2(J)
        if abs(det_J) < 1e-12:
            raise ValueError(f"det J = 0 на итерации {k}: метод Ньютона не может продолжить")
        det_A1 = det2([[F[0], J[0][1]], [F[1], J[1][1]]])
        det_A2 = det2([[J[0][0], F[0]], [J[1][0], F[1]]])
        x_new = [x1 - det_A1 / det_J, x2 - det_A2 / det_J]
        history.append(x_new)
        if distance(x_new, [x1, x2]) <= eps:
            return x_new, k, history
        x1, x2 = x_new
    raise ValueError(f"метод Ньютона не сошёлся за {MAX_ITERATIONS} итераций")


def simple_iteration(x0, a, eps, q):
    x1, x2 = x0
    history = [[x1, x2]]
    for k in range(1, MAX_ITERATIONS + 1):
        x_new = list(phi(x1, x2, a))
        history.append(x_new)
        if q / (1 - q) * distance(x_new, [x1, x2]) <= eps:
            return x_new, k, history
        x1, x2 = x_new
    raise ValueError(f"метод простой итерации не сошёлся за {MAX_ITERATIONS} итераций")


def print_history(history, root):
    print("   k           x1(k)          x2(k)   ||x(k) - x(k-1)||    ||x(k) - x*||")
    for k in range(len(history)):
        step = f"{distance(history[k], history[k - 1]):20.3e}" if k > 0 else " " * 20
        print(f"{k:4d} {history[k][0]:15.10f} {history[k][1]:14.10f} {step} {distance(history[k], root):16.3e}")


def plot(a, box1, box2):
    ts = [2 * math.pi * i / 400 for i in range(401)]
    plt.plot([a * math.cos(t) for t in ts], [a * math.sin(t) for t in ts], label="x1^2 + x2^2 = a^2")
    ys = grid(-3, 3)
    plt.plot([math.exp(y) - a for y in ys], ys, label="x1 = e^x2 - a")
    plt.gca().add_patch(plt.Rectangle((box1[0], box2[0]), box1[1] - box1[0], box2[1] - box2[0],
                                      alpha=0.3, label="G"))
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlim(-a - 1, a + 1)
    plt.ylim(-a - 1, a + 1)
    plt.gca().set_aspect("equal")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(PLOT_FILE)


def main():
    print(f"Данные: {INPUT_2_2}")
    rows = read_rows(INPUT_2_2)
    if len(rows) != 4 or len(rows[2]) != 2 or len(rows[3]) != 2:
        raise ValueError("в файле должно быть: eps, a, границы G по x1, границы G по x2")
    eps = rows[0][0]
    a = rows[1][0]
    box1, box2 = rows[2], rows[3]
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")
    if box1[0] >= box1[1] or box2[0] >= box2[1]:
        raise ValueError("границы области G должны идти по возрастанию")

    print(f"\nСистема (a = {a:g}):")
    print("  x1^2 + x2^2 - a^2 = 0")
    print("  x1 - e^x2 + a = 0")
    print(f"eps = {eps:g}")

    plot(a, box1, box2)
    print(f"график сохранён в {PLOT_FILE}: решение — пересечение окружности и кривой x1 = e^x2 - a")
    print(f"область G: {box1[0]:g} <= x1 <= {box1[1]:g}, {box2[0]:g} <= x2 <= {box2[1]:g}")
    x0 = [(box1[0] + box1[1]) / 2, (box2[0] + box2[1]) / 2]
    print_vector("x(0) = центр G", x0)

    root, _, _ = newton(x0, a, 1e-12)

    print("\n=== Метод простой итерации ===")
    print("x1 = phi1(x2) = sqrt(a^2 - x2^2),  x2 = phi2(x1) = ln(x1 + a)")
    q = 0.0
    inside = True
    for x1 in grid(box1[0], box1[1]):
        for x2 in grid(box2[0], box2[1]):
            q = max(q, matrix_norm(phi_derivative(x1, x2, a)))
            y1, y2 = phi(x1, x2, a)
            if not (box1[0] <= y1 <= box1[1] and box2[0] <= y2 <= box2[1]):
                inside = False
    print(f"q = max по G ||phi'(x)|| = {q:.6f}")
    if q >= 1:
        raise ValueError("q >= 1: условие сходимости метода простой итерации не выполнено")
    if inside:
        print("q < 1 и phi(G) лежит в G: условия теоремы 2.4 выполнены, метод сходится")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: phi(x) выходит за G, условие теоремы 2.4 не выполнено")
    print("останов: q / (1 - q) * ||x(k) - x(k-1)|| <= eps")
    x_simple, it_simple, hist_simple = simple_iteration(x0, a, eps, q)
    print_history(hist_simple, root)
    print_vector("решение", x_simple)
    print(f"итераций: {it_simple}")

    print("\n=== Метод Ньютона ===")
    print("x1(k+1) = x1(k) - det A1 / det J,  x2(k+1) = x2(k) - det A2 / det J")
    print("останов: ||x(k) - x(k-1)|| <= eps")
    x_newton, it_newton, hist_newton = newton(x0, a, eps)
    print_history(hist_newton, root)
    print_vector("решение", x_newton)
    print(f"итераций: {it_newton}")

    print("\n=== Сравнение ===")
    print(f"x* = ({root[0]:.12f}, {root[1]:.12f}) (метод Ньютона до 1e-12) — для столбца ||x(k) - x*||")
    print(f"метод простой итерации: {it_simple} итераций, погрешность {distance(x_simple, root):.2e}")
    print(f"метод Ньютона:          {it_newton} итераций, погрешность {distance(x_newton, root):.2e}")


if __name__ == "__main__":
    main()
