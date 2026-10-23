from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).ravel()
    x = np.asarray(coordinates, dtype=float).ravel()
    if y.size != x.size or y.size < 2:
        raise ValueError("trapezoidal integration requires equally sized arrays with at least two points")
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def mesh_xy(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x0, x0 + lx, nx)
    y = np.linspace(y0, y0 + ly, ny)
    return np.meshgrid(x, y)


def write_structured_grid_vtk(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    scalars: Mapping[str, np.ndarray] | None = None,
    vectors: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    title: str = "CFD lab output",
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be equally shaped 2D arrays")
    ny, nx = x.shape
    scalar_map = dict(scalars or {})
    vector_map = dict(vectors or {})
    for name, values in scalar_map.items():
        if np.asarray(values).shape != x.shape:
            raise ValueError(f"scalar {name} has shape {np.asarray(values).shape}, expected {x.shape}")
    for name, values in vector_map.items():
        if len(values) != 2 or np.asarray(values[0]).shape != x.shape or np.asarray(values[1]).shape != x.shape:
            raise ValueError(f"vector {name} has incompatible shape")
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_GRID\n")
        handle.write(f"DIMENSIONS {nx} {ny} 1\n")
        handle.write(f"POINTS {nx * ny} float\n")
        for j in range(ny):
            for i in range(nx):
                handle.write(f"{x[j, i]:.12e} {y[j, i]:.12e} 0\n")
        handle.write(f"POINT_DATA {nx * ny}\n")
        for name, values in scalar_map.items():
            data = np.asarray(values, dtype=float)
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in data.ravel(order="C"):
                handle.write(f"{value:.12e}\n")
        for name, values in vector_map.items():
            vx = np.asarray(values[0], dtype=float)
            vy = np.asarray(values[1], dtype=float)
            handle.write(f"VECTORS {name} float\n")
            for ax, ay in zip(vx.ravel(order="C"), vy.ravel(order="C")):
                handle.write(f"{ax:.12e} {ay:.12e} 0\n")


def write_polyline_vtk(path: str | Path, x: np.ndarray, y: np.ndarray, title: str = "polyline", close: bool = False) -> None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("polyline requires equally sized x and y arrays")
    ids = list(range(x.size))
    if close:
        ids.append(0)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {x.size} float\n")
        for xi, yi in zip(x, y):
            handle.write(f"{xi:.12e} {yi:.12e} 0\n")
        handle.write(f"LINES 1 {len(ids) + 1}\n")
        handle.write(f"{len(ids)} {' '.join(str(i) for i in ids)}\n")


def write_multiline_vtk(path: str | Path, lines: list[tuple[np.ndarray, np.ndarray]], title: str = "lines") -> None:
    points: list[tuple[float, float]] = []
    connectivity: list[list[int]] = []
    for x, y in lines:
        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        if xa.size != ya.size or xa.size < 2:
            raise ValueError("each line needs equally sized x and y arrays")
        ids = list(range(len(points), len(points) + xa.size))
        points.extend(zip(xa.tolist(), ya.tolist()))
        connectivity.append(ids)
    size = sum(len(ids) + 1 for ids in connectivity)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n")
        for xi, yi in points:
            handle.write(f"{xi:.12e} {yi:.12e} 0\n")
        handle.write(f"LINES {len(connectivity)} {size}\n")
        for ids in connectivity:
            handle.write(f"{len(ids)} {' '.join(str(i) for i in ids)}\n")


def write_points_vtk(path: str | Path, x: np.ndarray, y: np.ndarray, scalars: Mapping[str, np.ndarray] | None = None, title: str = "points") -> None:
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        raise ValueError("point arrays must have equal size")
    scalar_map = dict(scalars or {})
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {xa.size} float\n")
        for xi, yi in zip(xa, ya):
            handle.write(f"{xi:.12e} {yi:.12e} 0\n")
        handle.write(f"VERTICES {xa.size} {2 * xa.size}\n")
        for i in range(xa.size):
            handle.write(f"1 {i}\n")
        if scalar_map:
            handle.write(f"POINT_DATA {xa.size}\n")
            for name, values in scalar_map.items():
                data = np.asarray(values, dtype=float).ravel()
                if data.size != xa.size:
                    raise ValueError(f"scalar {name} has wrong size")
                handle.write(f"SCALARS {name} float 1\n")
                handle.write("LOOKUP_TABLE default\n")
                for value in data:
                    handle.write(f"{value:.12e}\n")


def gradient(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(field, dy, dx, edge_order=2)
    return gx, gy


def laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(field)
    out[1:-1, 1:-1] = (
        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx * dx)
        + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dy * dy)
    )
    return out


def upwind_derivative_x(field: np.ndarray, velocity: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(field)
    backward = (field[1:-1, 1:-1] - field[1:-1, :-2]) / dx
    forward = (field[1:-1, 2:] - field[1:-1, 1:-1]) / dx
    out[1:-1, 1:-1] = np.where(velocity[1:-1, 1:-1] >= 0.0, backward, forward)
    return out


def upwind_derivative_y(field: np.ndarray, velocity: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(field)
    backward = (field[1:-1, 1:-1] - field[:-2, 1:-1]) / dy
    forward = (field[2:, 1:-1] - field[1:-1, 1:-1]) / dy
    out[1:-1, 1:-1] = np.where(velocity[1:-1, 1:-1] >= 0.0, backward, forward)
    return out


def poisson_jacobi(phi: np.ndarray, rhs: np.ndarray, dx: float, dy: float, iterations: int, boundary: str = "neumann", mask: np.ndarray | None = None) -> np.ndarray:
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)
    for _ in range(iterations):
        old = phi.copy()
        phi[1:-1, 1:-1] = (
            (old[1:-1, 2:] + old[1:-1, :-2]) * dy2
            + (old[2:, 1:-1] + old[:-2, 1:-1]) * dx2
            - rhs[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        if mask is not None:
            phi[mask > 0.5] = 0.0
        if boundary == "neumann":
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = 0.0
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
        elif boundary == "closed":
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi -= np.mean(phi)
        else:
            raise ValueError("unsupported pressure boundary")
    return phi


def tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()
    n = d.size
    if n == 0 or a.size != n or b.size != n or c.size != n:
        raise ValueError("TDMA arrays must have equal nonzero size")
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-30:
            raise FloatingPointError("zero TDMA pivot")
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.empty(n, dtype=float)
    if abs(b[-1]) < 1e-30:
        raise FloatingPointError("zero TDMA pivot")
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        if abs(b[i]) < 1e-30:
            raise FloatingPointError("zero TDMA pivot")
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(numerical, dtype=float)
    b = np.asarray(reference, dtype=float)
    denominator = float(np.linalg.norm(b.ravel()))
    numerator = float(np.linalg.norm((a - b).ravel()))
    return numerator / max(denominator, 1e-30)


def require_finite(*arrays: np.ndarray) -> None:
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise FloatingPointError("non-finite numerical field detected")


def positive_float(value: str, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def positive_int(value: str, name: str, minimum: int = 1) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result
