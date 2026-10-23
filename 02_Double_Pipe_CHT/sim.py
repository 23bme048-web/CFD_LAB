from __future__ import annotations

import csv
import sys

import numpy as np

from common import require_finite, trapezoidal_integral, write_structured_grid_vtk


def main() -> int:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    ny_cold = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    ny_wall = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    ny_hot = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    re_hot = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0
    re_cold = float(sys.argv[6]) if len(sys.argv) > 6 else 100.0
    ks_ratio = float(sys.argv[7]) if len(sys.argv) > 7 else 20.0
    counter_flow = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    max_iterations = int(sys.argv[9]) if len(sys.argv) > 9 else 3000
    prandtl = float(sys.argv[10]) if len(sys.argv) > 10 else 7.0
    t_hot_in = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
    t_cold_in = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0
    u_hot_mean = float(sys.argv[13]) if len(sys.argv) > 13 else 1.0
    u_cold_mean = float(sys.argv[14]) if len(sys.argv) > 14 else 1.0
    if nx < 20 or ny_cold < 4 or ny_wall < 2 or ny_hot < 4:
        raise ValueError("grid is too small")
    if re_hot <= 0.0 or re_cold <= 0.0 or ks_ratio <= 0.0 or prandtl <= 0.0 or max_iterations < 1:
        raise ValueError("physical and iteration parameters must be positive")
    counter_flow = 1 if counter_flow else 0
    ny = ny_cold + ny_wall + ny_hot
    length = 8.0
    height = 2.0
    x1 = np.linspace(0.0, length, nx)
    y1 = np.linspace(0.0, height, ny)
    x, y = np.meshgrid(x1, y1)
    dx = x1[1] - x1[0]
    dy = y1[1] - y1[0]
    zone = np.zeros((ny, nx), dtype=float)
    zone[ny_cold:ny_cold + ny_wall, :] = 1.0
    zone[ny_cold + ny_wall:, :] = 2.0
    velocity = np.zeros((ny, nx), dtype=float)
    cold_eta = np.linspace(0.0, 1.0, ny_cold)
    hot_eta = np.linspace(0.0, 1.0, ny_hot)
    cold_profile = 6.0 * cold_eta * (1.0 - cold_eta)
    hot_profile = 6.0 * hot_eta * (1.0 - hot_eta)
    cold_direction = -1.0 if counter_flow else 1.0
    velocity[:ny_cold, :] = cold_direction * u_cold_mean * cold_profile[:, None]
    velocity[ny_cold + ny_wall:, :] = u_hot_mean * hot_profile[:, None]
    gamma_hot = max(abs(u_hot_mean), 1e-8) / (re_hot * prandtl)
    gamma_cold = max(abs(u_cold_mean), 1e-8) / (re_cold * prandtl)
    gamma_wall = ks_ratio * 0.5 * (gamma_hot + gamma_cold)
    gamma = np.full((ny, nx), gamma_cold, dtype=float)
    gamma[ny_cold:ny_cold + ny_wall, :] = gamma_wall
    gamma[ny_cold + ny_wall:, :] = gamma_hot
    temperature = np.full((ny, nx), 0.5 * (t_hot_in + t_cold_in), dtype=float)
    temperature[:ny_cold, :] = t_cold_in
    temperature[ny_cold + ny_wall:, :] = t_hot_in
    temperature[ny_cold:ny_cold + ny_wall, :] = np.linspace(t_cold_in, t_hot_in, ny_wall)[:, None]
    relax = 0.82
    history: list[list[float]] = []

    def apply_boundaries(field: np.ndarray) -> None:
        field[ny_cold + ny_wall:, 0] = t_hot_in
        field[ny_cold + ny_wall:, -1] = field[ny_cold + ny_wall:, -2]
        if counter_flow:
            field[:ny_cold, -1] = t_cold_in
            field[:ny_cold, 0] = field[:ny_cold, 1]
        else:
            field[:ny_cold, 0] = t_cold_in
            field[:ny_cold, -1] = field[:ny_cold, -2]
        field[ny_cold:ny_cold + ny_wall, 0] = field[ny_cold:ny_cold + ny_wall, 1]
        field[ny_cold:ny_cold + ny_wall, -1] = field[ny_cold:ny_cold + ny_wall, -2]
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]

    apply_boundaries(temperature)
    for iteration in range(1, max_iterations + 1):
        old = temperature.copy()
        gp = gamma[1:-1, 1:-1]
        ge = 2.0 * gp * gamma[1:-1, 2:] / np.maximum(gp + gamma[1:-1, 2:], 1e-30)
        gw = 2.0 * gp * gamma[1:-1, :-2] / np.maximum(gp + gamma[1:-1, :-2], 1e-30)
        gn = 2.0 * gp * gamma[2:, 1:-1] / np.maximum(gp + gamma[2:, 1:-1], 1e-30)
        gs = 2.0 * gp * gamma[:-2, 1:-1] / np.maximum(gp + gamma[:-2, 1:-1], 1e-30)
        flow = velocity[1:-1, 1:-1] * dy
        a_e = ge * dy / dx + np.maximum(-flow, 0.0)
        a_w = gw * dy / dx + np.maximum(flow, 0.0)
        a_n = gn * dx / dy
        a_s = gs * dx / dy
        a_p = np.maximum(a_e + a_w + a_n + a_s, 1e-30)
        candidate = (
            a_e * old[1:-1, 2:]
            + a_w * old[1:-1, :-2]
            + a_n * old[2:, 1:-1]
            + a_s * old[:-2, 1:-1]
        ) / a_p
        temperature[1:-1, 1:-1] = (1.0 - relax) * old[1:-1, 1:-1] + relax * candidate
        apply_boundaries(temperature)
        residual = float(np.max(np.abs(temperature - old)))
        hot_out = float(np.average(temperature[ny_cold + ny_wall:, -1], weights=np.maximum(velocity[ny_cold + ny_wall:, -1], 0.0) + 1e-12))
        cold_column = 0 if counter_flow else -1
        cold_weights = np.abs(velocity[:ny_cold, cold_column]) + 1e-12
        cold_out = float(np.average(temperature[:ny_cold, cold_column], weights=cold_weights))
        history.append([iteration, residual, hot_out, cold_out, float(np.min(temperature)), float(np.max(temperature))])
        require_finite(temperature)
        if iteration == 1 or iteration == max_iterations or iteration % max(1, max_iterations // 10) == 0:
            print(f"iteration={iteration} residual={residual:.6e} hot_out={hot_out:.6f} cold_out={cold_out:.6f}")
        if residual < 1e-9 and iteration > 50:
            break
    hot_in_weights = np.maximum(velocity[ny_cold + ny_wall:, 0], 0.0) + 1e-12
    hot_out_weights = np.maximum(velocity[ny_cold + ny_wall:, -1], 0.0) + 1e-12
    cold_in_col = -1 if counter_flow else 0
    cold_out_col = 0 if counter_flow else -1
    cold_in_weights = np.abs(velocity[:ny_cold, cold_in_col]) + 1e-12
    cold_out_weights = np.abs(velocity[:ny_cold, cold_out_col]) + 1e-12
    hot_in = float(np.average(temperature[ny_cold + ny_wall:, 0], weights=hot_in_weights))
    hot_out = float(np.average(temperature[ny_cold + ny_wall:, -1], weights=hot_out_weights))
    cold_in = float(np.average(temperature[:ny_cold, cold_in_col], weights=cold_in_weights))
    cold_out = float(np.average(temperature[:ny_cold, cold_out_col], weights=cold_out_weights))
    mdot_hot = trapezoidal_integral(np.maximum(velocity[ny_cold + ny_wall:, 0], 0.0), y1[ny_cold + ny_wall:])
    mdot_cold = trapezoidal_integral(np.abs(velocity[:ny_cold, cold_in_col]), y1[:ny_cold])
    q_hot = mdot_hot * (hot_in - hot_out)
    q_cold = mdot_cold * (cold_out - cold_in)
    imbalance = abs(q_hot - q_cold) / max(abs(q_hot), abs(q_cold), 1e-30)
    effectiveness = q_cold / max(min(mdot_hot, mdot_cold) * (t_hot_in - t_cold_in), 1e-30)
    write_structured_grid_vtk("double_pipe_cht.vtk", x, y, {"temperature": temperature, "zone": zone, "conductivity_ratio_field": gamma / max(gamma_cold, 1e-30)}, {"velocity": (velocity, np.zeros_like(velocity))}, "Double-pipe conjugate heat transfer")
    with open("history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "residual_linf", "hot_outlet_temperature", "cold_outlet_temperature", "minimum_temperature", "maximum_temperature"])
        writer.writerows(history)
    with open("summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Teaching-level 2D conjugate heat-transfer analogue\n")
        handle.write(f"nx {nx}\nny_cold {ny_cold}\nny_wall {ny_wall}\nny_hot {ny_hot}\n")
        handle.write(f"counter_flow {counter_flow}\nRe_hot {re_hot:.12g}\nRe_cold {re_cold:.12g}\nPr {prandtl:.12g}\nks_by_kf {ks_ratio:.12g}\n")
        handle.write(f"hot_inlet_temperature {hot_in:.12e}\nhot_outlet_temperature {hot_out:.12e}\n")
        handle.write(f"cold_inlet_temperature {cold_in:.12e}\ncold_outlet_temperature {cold_out:.12e}\n")
        handle.write(f"hot_stream_heat_loss {q_hot:.12e}\ncold_stream_heat_gain {q_cold:.12e}\n")
        handle.write(f"relative_energy_imbalance {imbalance:.12e}\neffectiveness {effectiveness:.12e}\n")
        handle.write(f"iterations_completed {len(history)}\nfinal_residual_linf {history[-1][1]:.12e}\n")
    print("Wrote double_pipe_cht.vtk, history.csv, summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
