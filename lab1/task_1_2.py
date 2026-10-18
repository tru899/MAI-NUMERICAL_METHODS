from common import *


def check_structure(a, b, c, d):
    n = len(b)
    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError(f"длины a, b, c, d должны совпадать: {len(a)}, {n}, {len(c)}, {len(d)}")
    if a[0] != 0:
        raise ValueError("a_1 должно быть 0: у первого уравнения нет соседа слева")
    if c[n - 1] != 0:
        raise ValueError("c_n должно быть 0: у последнего уравнения нет соседа справа")


def check_dominance(a, b, c):
    bad_rows = []
    strict = False
    for i in range(len(b)):
        diagonal = abs(b[i])
        neighbours = abs(a[i]) + abs(c[i])
        if diagonal < neighbours:
            bad_rows.append(i + 1)
        elif diagonal > neighbours:
            strict = True
    return bad_rows, strict


def thomas_forward(a, b, c, d):
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n

    if abs(b[0]) < 1e-12:
        raise ValueError("b_1 = 0: прогонку не начать, первый знаменатель равен нулю")
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] + a[i] * P[i - 1]
        if abs(denominator) < 1e-12:
            raise ValueError(f"на шаге {i + 1} знаменатель b + a*P равен нулю: метод не может продолжить")
        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

    P[n - 1] = 0.0

    return P, Q


def thomas_backward(P, Q):
    n = len(P)
    x = [0.0] * n
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


def main():
    print(f"Данные: {INPUT_1_2}")
    rows = read_rows(INPUT_1_2)
    if len(rows) != 5:
        raise ValueError("в файле должно быть 5 строк: n, затем a, b, c, d")
    n = int(rows[0][0])
    a, b, c, d = rows[1], rows[2], rows[3], rows[4]
    if len(b) != n:
        raise ValueError(f"n = {n}, а на главной диагонали {len(b)} чисел")
    check_structure(a, b, c, d)

    bad_rows, strict = check_dominance(a, b, c)
    if bad_rows:
        print(f"\nПРЕДУПРЕЖДЕНИЕ: нет диагонального преобладания в строках {bad_rows}.")
        print("Условие достаточное: считаем дальше, но устойчивость заранее не гарантирована.")
    elif not strict:
        print("\nПРЕДУПРЕЖДЕНИЕ: преобладание нигде не строгое, устойчивость заранее не гарантирована.")
    else:
        print("\nДиагональное преобладание выполнено: метод корректен и устойчив.")

    P, Q = thomas_forward(a, b, c, d)

    print("\n=== 1. Прогоночные коэффициенты ===")
    print("   i            P_i            Q_i")
    for i in range(n):
        print(f"{i + 1:4d} {P[i]:14.6f} {Q[i]:14.6f}")

    max_P = 0.0
    for i in range(n):
        max_P = max(max_P, abs(P[i]))
    print(f"max|P_i| = {max_P:.6f}")
    if max_P > 1:
        print("ПРЕДУПРЕЖДЕНИЕ: есть |P_i| > 1 — затухание погрешности не гарантировано, смотри невязку.")

    print("\n=== 2. Решение системы ===")
    x = thomas_backward(P, Q)
    print_vector("x", x)


if __name__ == "__main__":
    main()
