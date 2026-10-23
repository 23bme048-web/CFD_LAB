from __future__ import annotations

import csv
import math
import sys

import numpy as np

from common import gradient, laplacian, mesh_xy, poisson_jacobi, require_finite, upwind_derivative_x, upwind_derivative_y, write_polyline_vtk, write_structured_grid_vtk


def naca0012_half(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, 0.0, 1.0)
    yt = 5.0 * 0.12 * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1036 * xc**4)
    return np.where((x >= 0.0) & (x <= 1.0), yt, -1.0)


def airfoil_geometry(alpha_deg: float, chord: float = 1.0, x_le: float = 1.35, y_c: float = 0.0, n: int = 401) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linspace(0.0, math.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = naca0012_half(xc)
    xb = np.concatenate((xc[::-1], xc[1:])) * chord
    yb = np.concatenate((-yt[::-1], yt[1:])) * chord
    angle = math.radians(alpha_deg)
    xq = x_le + 0.25 * chord
    yq = y_c
    xr = xb - 0.25 * chord
    x = xq + math.cos(angle) * xr - math.sin(angle) * yb
    y = yq + math.sin(angle) * xr + math.cos(angle) * yb
    return x, y


def airfoil_mask(x: np.ndarray, y: np.ndarray, alpha_deg: float, dx: float, dy: float, chord: float = 1.0, x_le: float = 1.35, y_c: float = 0.0) -> np.ndarray:
    angle = math.radians(alpha_deg)
    xq = x_le + 0.25 * chord
    xr = math.cos(angle) * (x - xq) + math.sin(angle) * (y - y_c) + 0.25 * chord
    yr = -math.sin(angle) * (x - xq) + math.cos(angle) * (y - y_c)
    xc = xr / chord
    yt = naca0012_half(xc) * chord
    distance = yt - np.abs(yr)
    epsilon = 0.75 * max(dx, dy)
    raw = np.clip(0.5 + 0.5 * distance / epsilon, 0.0, 1.0)
    smooth = raw * raw * (3.0 - 2.0 * raw)
    return np.where(yt >= 0.0, smooth, 0.0)


def apply_bc(u: np.ndarray, v: np.ndarray, p: np.ndarray, speed: float) -> None:
    u[:, 0] = speed
    v[:, 0] = 0.0
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    p[:, -1] = 0.0
    p[:, 0] = p[:, 1]
    u[0, :] = speed
    u[-1, :] = speed
    v[0, :] = 0.0
    v[-1, :] = 0.0
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]


def main() -> int:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    reynolds = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    alpha_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    poisson_iterations = int(sys.argv[6]) if len(sys.argv) > 6 else 80
    if nx < 24 or ny < 16 or reynolds <= 0.0 or iterations < 1 or poisson_iterations < 1:
        raise ValueError("usage: python exp01_airfoil_fvm_simple.py nx ny Re alpha_deg iterations poisson_iterations")
    lx = 5.0
    ly = 2.5
    speed = 1.0
    chord = 1.0
    nu = speed * chord / reynolds
    x1 = np.linspace(0.0, lx, nx)
    y1 = np.linspace(-0.5 * ly, 0.5 * ly, ny)
    x, y = np.meshgrid(x1, y1)
    dx = x1[1] - x1[0]
    dy = y1[1] - y1[0]
    chi = airfoil_mask(x, y, alpha_deg, dx, dy)
    u = speed * (1.0 - chi)
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    dt = min(0.10 * min(dx, dy) / speed, 0.18 * min(dx * dx, dy * dy) / nu)
    eta = max(0.5 * dt, 1e-5)
    apply_bc(u, v, p, speed)
    history_rows: list[list[float]] = []
    for step in range(1, iterations + 1):
        du_dx = upwind_derivative_x(u, u, dx)
        du_dy = upwind_derivative_y(u, v, dy)
        dv_dx = upwind_derivative_x(v, u, dx)
        dv_dy = upwind_derivative_y(v, v, dy)
        us = u + dt * (-(u * du_dx + v * du_dy) + nu * laplacian(u, dx, dy) - chi * u / eta)
        vs = v + dt * (-(u * dv_dx + v * dv_dy) + nu * laplacian(v, dx, dy) - chi * v / eta)
        apply_bc(us, vs, p, speed)
        rhs = np.zeros_like(p)
        rhs[1:-1, 1:-1] = ((us[1:-1, 2:] - us[1:-1, :-2]) / (2.0 * dx) + (vs[2:, 1:-1] - vs[:-2, 1:-1]) / (2.0 * dy)) / dt
        phi = poisson_jacobi(np.zeros_like(p), rhs, dx, dy, poisson_iterations, boundary="neumann", mask=chi)
        gx, gy = gradient(phi, dx, dy)
        u_corrected = us - dt * gx
        v_corrected = vs - dt * gy
        p += 0.25 * phi
        force_x = float(np.sum(chi * u_corrected / eta) * dx * dy)
        force_y = float(np.sum(chi * v_corrected / eta) * dx * dy)
        u = u_corrected * (1.0 - chi)
        v = v_corrected * (1.0 - chi)
        apply_bc(u, v, p, speed)
        divergence = np.zeros_like(u)
        divergence[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
        fluid = chi < 0.5
        continuity_l2 = float(np.sqrt(np.mean(divergence[fluid] ** 2)))
        angle = math.radians(alpha_deg)
        drag = force_x * math.cos(angle) + force_y * math.sin(angle)
        lift = -force_x * math.sin(angle) + force_y * math.cos(angle)
        qref = 0.5 * speed * speed * chord
        cd = drag / qref
        cl = lift / qref
        umax = float(np.max(np.sqrt(u * u + v * v)))
        history_rows.append([step, step * dt, continuity_l2, umax, cd, cl])
        require_finite(u, v, p)
        if umax > 25.0:
            raise FloatingPointError(f"solution diverged at iteration {step}")
        if step == 1 or step == iterations or step % max(1, iterations // 5) == 0:
            print(f"iteration={step} continuity_l2={continuity_l2:.6e} max_speed={umax:.6e} Cd_penalty={cd:.6e} Cl_penalty={cl:.6e}")
    speed_field = np.sqrt(u * u + v * v)
    vorticity = np.zeros_like(u)
    vorticity[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)
    write_structured_grid_vtk("airfoil_final.vtk", x, y, {"pressure": p, "speed": speed_field, "vorticity": vorticity, "solid_mask": chi}, {"velocity": (u, v)}, "NACA 0012 immersed-boundary flow")
    wall_x, wall_y = airfoil_geometry(alpha_deg)
    write_polyline_vtk("airfoil_wall.vtk", wall_x, wall_y, "NACA 0012 wall", close=True)
    write_structured_grid_vtk("airfoil_mask.vtk", x, y, {"solid_mask": chi}, title="NACA 0012 mask")
    with open("history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "pseudo_time", "continuity_l2", "max_speed", "Cd_penalty_qualitative", "Cl_penalty_qualitative"])
        writer.writerows(history_rows)
    with open("airfoil_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Teaching-level immersed-boundary projection solution\n")
        handle.write(f"nx {nx}\nny {ny}\nRe {reynolds:.12g}\nalpha_deg {alpha_deg:.12g}\n")
        handle.write(f"dt {dt:.12e}\niterations {iterations}\npoisson_iterations {poisson_iterations}\n")
        handle.write(f"final_continuity_l2 {history_rows[-1][2]:.12e}\n")
        handle.write("penalty_force_coefficients_are_qualitative_only true\n")
    print("Wrote airfoil_final.vtk, airfoil_wall.vtk, airfoil_mask.vtk, history.csv, airfoil_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
