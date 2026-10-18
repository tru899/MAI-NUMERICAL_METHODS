INPUT_1_1 = "data/1_1.txt"
INPUT_1_2 = "data/1_2.txt"
INPUT_1_3 = "data/1_3.txt"
INPUT_1_4 = "data/1_4.txt"
INPUT_1_5 = "data/1_5.txt"
INPUT_1_5_COMPLEX = "data/1_5_complex.txt"


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append([float(x) for x in line.split()])
    return rows


def print_matrix(name, M):
    print(f"{name}:")
    for row in M:
        print("  " + " ".join(f"{x:10.4f}" for x in row))


def print_vector(name, v):
    print(f"{name}: " + " ".join(f"{x:10.4f}" for x in v))
