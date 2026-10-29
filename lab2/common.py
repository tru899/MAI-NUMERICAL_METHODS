INPUT_2_1 = "data/2_1.txt"
INPUT_2_2 = "data/2_2.txt"


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
