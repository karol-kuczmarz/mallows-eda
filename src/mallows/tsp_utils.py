import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def read_tsp_file(filepath):
    def get_dimension(file):
        for line in file:
            if line.startswith("DIMENSION"):
                return int(line.split()[-1])

    def get_edge_weight_type(file):
        for line in file:
            if line.startswith("EDGE_WEIGHT_TYPE"):
                return line.split()[-1]

    def get_edge_weight_format(file):
        for line in file:
            if line.startswith("EDGE_WEIGHT_FORMAT"):
                return line.split()[-1]

    def get_edge_weight_section_loc(file):
        for index, line in enumerate(file):
            if line.startswith("EDGE_WEIGHT_SECTION"):
                return index

    def get_dist_matrix_from_upper_row(lines):
        dist_matrix = np.zeros((n, n))
        for index, line in enumerate(lines):
            line = np.array(
                list(map(lambda x: int(x), filter(lambda x: x != "", line.split())))
            )
            dist_matrix[index, index + 1 :] = line
            dist_matrix[index + 1 :, index] = line
        return dist_matrix

    def get_dist_matrix_from_full_matrix(lines):
        dist_matrix = np.zeros((n, n))
        for index, line in enumerate(lines):
            line = np.array(
                list(map(lambda x: int(x), filter(lambda x: x != "", line.split())))
            )
            dist_matrix[index, :] = line
        return dist_matrix

    def get_node_coord_section_loc(file, section_name):
        for index, line in enumerate(file):
            if line.startswith(section_name):
                return index

    def get_coords(lines):
        coords = np.zeros((n, 2))
        for index, line in enumerate(lines):
            coords[index, :] = np.array(
                list(
                    map(
                        lambda x: float(x),
                        list(filter(lambda x: x != "", line.split()))[1:],
                    )
                )
            )
        return coords

    def get_dist_matrix_from_coords(coords):
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i, j] = np.sqrt(np.sum((coords[i, :] - coords[j, :]) ** 2))
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    def get_solution_loc(lines):
        for index, line in enumerate(lines):
            if line.startswith("TOUR_SECTION"):
                return index

    def get_optimal_solution(lines):
        optimal_solution = np.zeros(n, dtype=np.int64)
        for index, line in enumerate(lines):
            optimal_solution[index] = int(line.split()[0]) - 1
        return optimal_solution

    is_opt = str(filepath).endswith("tour.tsp")

    with open(filepath, "r") as file:
        lines = file.readlines()

    n = get_dimension(lines)
    edge_weight_type = get_edge_weight_type(lines)
    edge_weight_format = get_edge_weight_format(lines)

    if edge_weight_type == "EXPLICIT":
        if edge_weight_format == "UPPER_ROW":
            edge_weight_section_loc = get_edge_weight_section_loc(lines)
            dist_matrix = get_dist_matrix_from_upper_row(
                lines[edge_weight_section_loc + 1 : edge_weight_section_loc + n]
            )
        elif edge_weight_format == "FULL_MATRIX":
            edge_weight_section_loc = get_edge_weight_section_loc(lines)
            dist_matrix = get_dist_matrix_from_full_matrix(
                lines[edge_weight_section_loc + 1 : edge_weight_section_loc + n + 1]
            )
        node_coord_loc = get_node_coord_section_loc(lines, "DISPLAY_DATA_SECTION")
    elif edge_weight_type == "EUC_2D":
        node_coord_loc = get_node_coord_section_loc(lines, "NODE_COORD_SECTION")
    coords = get_coords(lines[node_coord_loc + 1 : node_coord_loc + n + 1])
    if edge_weight_type == "EUC_2D":
        dist_matrix = get_dist_matrix_from_coords(coords)

    if is_opt:
        lines = open(filepath, "r").readlines()
        tour_section_loc = get_solution_loc(lines)
        optimal_solution = get_optimal_solution(
            lines[tour_section_loc + 1 : tour_section_loc + n + 1]
        )
        return coords, dist_matrix, optimal_solution

    return coords, dist_matrix, None


def plot_solution(p, title, coords, dist_matrix):
    route = p
    n = len(route)

    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.plot(coords[:, 0], coords[:, 1], "o")

    for i in range(n):
        plt.text(
            coords[i, 0] + 8,
            coords[i, 1] + 8,
            str(i),
            fontdict={"weight": "bold", "size": 8},
        )
    for i in range(0, len(route)):
        ax.add_line(
            Line2D(
                [coords[route[i - 1], 0], coords[route[i], 0]],
                [coords[route[i - 1], 1], coords[route[i], 1]],
                linewidth=1,
                color="gray",
            )
        )
        plt.text(
            (coords[route[i - 1], 0] + coords[route[i], 0]) / 2 + 6,
            (coords[route[i - 1], 1] + coords[route[i], 1]) / 2 + 6,
            "%d" % dist_matrix[route[i - 1], route[i]],
            fontdict={"weight": "normal", "size": 7},
        )
    ax.add_line(
        Line2D(
            [coords[route[-1], 0], coords[route[0], 0]],
            [coords[route[-1], 1], coords[route[0], 1]],
            linewidth=1,
            color="gray",
        )
    )
    plt.text(
        (coords[route[-1], 0] + coords[route[0], 0]) / 2 + 6,
        (coords[route[-1], 1] + coords[route[0], 1]) / 2 + 6,
        "%d" % dist_matrix[route[-1], route[0]],
        fontdict={"weight": "normal", "size": 7},
    )

    plt.title(title)

    plt.show()
