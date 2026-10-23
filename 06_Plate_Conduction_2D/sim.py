from __future__ import annotations

import csv
import math
import sys

import numpy as np

from common import relative_l2, tdma, write_multiline_vtk, write_structured_grid_vtk


def analytical_solution(x: np.ndarray, y: np.ndarray, length: float, height: float, t_left: float) -> np.ndarray:
    denominator = math.sinh(math.pi * length / height)
    return t_left * np.sinh(math.pi * (length - x) / height) * np.sin(math.pi * y / height) / denominator


def apply_boundaries(field: np.ndarray, y: np.ndarray, height: float, t_left: float) -> None:
    field[:, 0] = t_left * np.sin(math.pi * y / height)
    field[:, -1] = 0.0
    field[0, :] = 0.0
    field[-1, :] = 0.0


def main() -> int:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 81
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 61
    length = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    height = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    t_left = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0
    max_iterations = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
    tolerance = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-8
    omega = float(sys.argv[8]) if len(sys.argv) > 8 else 1.55
    screen_every = int(sys.argv[9]) if len(sys.argv) > 9 else 1000
    if nx < 5 or ny < 5 or length <= 0.0 or height <= 0.0 or max_iterations < 1 or tolerance <= 0.0 or not 0.0 < omega < 2.0:
        raise ValueError("usage: python exp06_plate_conduction_explicit_implicit.py nx ny L H Tleft max_iterations tolerance omega screen_every")
    x1 = np.linspace(0.0, length, nx)
    y1 = np.linspace(0.0, height, ny)
    x, y = np.meshgrid(x1, y1)
    dx = x1[1] - x1[0]
    dy = y1[1] - y1[0]
    beta = (dx / dy) ** 2
    analytical = analytical_solution(x, y, length, height, t_left)
    jacobi = np.zeros((ny, nx), dtype=float)
    line_tdma = np.zeros((ny, nx), dtype=float)
    apply_boundaries(jacobi, y1, height, t_left)
    apply_boundaries(line_tdma, y1, height, t_left)
    history: list[list[object]] = []
    jacobi_iterations = max_iterations
    jacobi_residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        old = jacobi.copy()
        jacobi[1:-1, 1:-1] = (old[1:-1, 2:] + old[1:-1, :-2] + beta * (old[2:, 1:-1] + old[:-2, 1:-1])) / (2.0 * (1.0 + beta))
        apply_boundaries(jacobi, y1, height, t_left)
        jacobi_residual = float(np.max(np.abs(jacobi - old)))
        history.append(["jacobi", iteration, jacobi_residual, relative_l2(jacobi, analytical)])
        if iteration == 1 or iteration == max_iterations or iteration % max(1, screen_every) == 0:
            print(f"jacobi_iteration={iteration} residual={jacobi_residual:.6e}")
        if jacobi_residual < tolerance:
            jacobi_iterations = iteration
            break
    tdma_iterations = max_iterations
    tdma_residual = float("inf")
    n_unknown = nx - 2
    for iteration in range(1, max_iterations + 1):
        old_iteration = line_tdma.copy()
        for j in range(1, ny - 1):
            a = np.full(n_unknown, -1.0)
            b = np.full(n_unknown, 2.0 * (1.0 + beta))
            c = np.full(n_unknown, -1.0)
            rhs = beta * (line_tdma[j + 1, 1:-1] + line_tdma[j - 1, 1:-1])
            rhs[0] += line_tdma[j, 0]
            rhs[-1] += line_tdma[j, -1]
            a[0] = 0.0
            c[-1] = 0.0
            solved = tdma(a, b, c, rhs)
            line_tdma[j, 1:-1] = (1.0 - omega) * line_tdma[j, 1:-1] + omega * solved
        apply_boundaries(line_tdma, y1, height, t_left)
        tdma_residual = float(np.max(np.abs(line_tdma - old_iteration)))
        history.append(["line_tdma", iteration, tdma_residual, relative_l2(line_tdma, analytical)])
        if iteration == 1 or iteration == max_iterations or iteration % max(1, screen_every) == 0:
            print(f"line_tdma_iteration={iteration} residual={tdma_residual:.6e}")
        if tdma_residual < tolerance:
            tdma_iterations = iteration
            break
    jacobi_error = relative_l2(jacobi, analytical)
    tdma_error = relative_l2(line_tdma, analytical)
    write_structured_grid_vtk("plate_conduction_2d.vtk", x, y, {"temperature_jacobi": jacobi, "temperature_line_tdma": line_tdma, "temperature_analytical": analytical, "jacobi_absolute_error": np.abs(jacobi - analytical), "line_tdma_absolute_error": np.abs(line_tdma - analytical)}, title="2D plate heat conduction")
    write_multiline_vtk("plate_boundaries.vtk", [(np.array([0.0, length]), np.array([0.0, 0.0])), (np.array([length, length]), np.array([0.0, height])), (np.array([length, 0.0]), np.array([height, height])), (np.array([0.0, 0.0]), np.array([height, 0.0]))], "Plate boundaries")
    center_j = ny // 2
    center_i = nx // 2
    with open("plate_conduction_profiles.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["coordinate", "horizontal_jacobi", "horizontal_line_tdma", "horizontal_analytical", "vertical_jacobi", "vertical_line_tdma", "vertical_analytical"])
        nrows = max(nx, ny)
        for k in range(nrows):
            coordinate = k / max(nrows - 1, 1)
            ih = min(int(round(coordinate * (nx - 1))), nx - 1)
            jv = min(int(round(coordinate * (ny - 1))), ny - 1)
            writer.writerow([coordinate, jacobi[center_j, ih], line_tdma[center_j, ih], analytical[center_j, ih], jacobi[jv, center_i], line_tdma[jv, center_i], analytical[jv, center_i]])
    with open("plate_conduction_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "iteration", "residual_linf", "relative_l2_error"])
        writer.writerows(history)
    with open("plate_conduction_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Steady 2D Laplace heat conduction\n")
        handle.write(f"nx {nx}\nny {ny}\nL {length:.12e}\nH {height:.12e}\nTleft {t_left:.12e}\n")
        handle.write(f"jacobi_iterations {jacobi_iterations}\njacobi_final_residual {jacobi_residual:.12e}\njacobi_relative_L2_error {jacobi_error:.12e}\n")
        handle.write(f"line_tdma_iterations {tdma_iterations}\nline_tdma_final_residual {tdma_residual:.12e}\nline_tdma_relative_L2_error {tdma_error:.12e}\n")
    print(f"Jacobi iterations={jacobi_iterations} residual={jacobi_residual:.6e} L2_error={jacobi_error:.6e}")
    print(f"Line_TDMA iterations={tdma_iterations} residual={tdma_residual:.6e} L2_error={tdma_error:.6e}")
    print("Wrote plate_conduction_2d.vtk, plate_boundaries.vtk, plate_conduction_profiles.csv, plate_conduction_history.csv, plate_conduction_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
