from __future__ import annotations

import csv
import math
import sys

import numpy as np

from common import require_finite, write_multiline_vtk, write_points_vtk, write_structured_grid_vtk


def main() -> int:
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    rpm = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.0
    reynolds = float(sys.argv[4]) if len(sys.argv) > 4 else 500.0
    t_in = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0
    t_wall = float(sys.argv[6]) if len(sys.argv) > 6 else 330.0
    iterations = int(sys.argv[7]) if len(sys.argv) > 7 else 1200
    swirl = float(sys.argv[8]) if len(sys.argv) > 8 else 0.85
    wall_htc = float(sys.argv[9]) if len(sys.argv) > 9 else 120.0
    loss_fraction = float(sys.argv[10]) if len(sys.argv) > 10 else 0.04
    if nr < 12 or nt < 24 or rpm <= 0.0 or reynolds <= 0.0 or iterations < 1 or wall_htc < 0.0 or loss_fraction < 0.0:
        raise ValueError("usage: python exp03_centrifugal_pump_srf.py nr nt rpm Re Tin Twall iterations swirl wallHTC lossFraction")
    r1 = 0.030
    r2 = 0.120
    depth = 0.010
    rho = 997.0
    mu = 0.0010
    cp = 4180.0
    conductivity = 0.60
    gravity = 9.81
    omega = rpm * 2.0 * math.pi / 60.0
    hydraulic_diameter = 2.0 * depth
    inlet_velocity = reynolds * mu / (rho * hydraulic_diameter)
    r = np.linspace(r1, r2, nr)
    theta = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    u_r = inlet_velocity * r1 / rr
    u_theta_abs = swirl * omega * rr
    u_theta_rel = (swirl - 1.0) * omega * rr
    u_abs_x = u_r * np.cos(tt) - u_theta_abs * np.sin(tt)
    u_abs_y = u_r * np.sin(tt) + u_theta_abs * np.cos(tt)
    u_rel_x = u_r * np.cos(tt) - u_theta_rel * np.sin(tt)
    u_rel_y = u_r * np.sin(tt) + u_theta_rel * np.cos(tt)
    pressure = 0.5 * rho * omega * omega * (rr * rr - r1 * r1) - 12.0 * mu * inlet_velocity * r1 * np.log(rr / r1) / (depth * depth)
    alpha = conductivity / (rho * cp)
    wall_lambda = wall_htc * 2.0 / (rho * cp * depth)
    loss_heat = loss_fraction * omega * omega * rr * np.maximum(u_r, 0.0) / cp
    temperature = np.full((nr, nt), t_in, dtype=float)
    history: list[list[float]] = []
    relax = 0.85
    for iteration in range(1, iterations + 1):
        old = temperature.copy()
        west = np.vstack((np.full((1, nt), t_in), old[:-1, :]))
        east = np.vstack((old[1:, :], old[-1:, :]))
        south = np.roll(old, 1, axis=1)
        north = np.roll(old, -1, axis=1)
        ar = np.abs(u_r) / dr
        at = np.abs(u_theta_rel) / np.maximum(rr * dtheta, 1e-30)
        d_r = alpha / (dr * dr)
        d_t = alpha / np.maximum(rr * rr * dtheta * dtheta, 1e-30)
        angular_upwind = np.where(u_theta_rel >= 0.0, south, north)
        candidate = (
            ar * west
            + at * angular_upwind
            + d_r * (east + west)
            + d_t * (north + south)
            + wall_lambda * t_wall
            + loss_heat
        ) / np.maximum(ar + at + 2.0 * d_r + 2.0 * d_t + wall_lambda, 1e-30)
        temperature = (1.0 - relax) * old + relax * candidate
        temperature[0, :] = t_in
        temperature[-1, :] = temperature[-2, :]
        lower = min(t_in, t_wall) - 100.0
        upper = max(t_in, t_wall) + 100.0
        temperature = np.clip(temperature, lower, upper)
        residual = float(np.max(np.abs(temperature - old)))
        t_out = float(np.mean(temperature[-1, :]))
        history.append([iteration, residual, float(np.min(temperature)), float(np.max(temperature)), t_out])
        require_finite(temperature)
        if iteration == 1 or iteration == iterations or iteration % max(1, iterations // 8) == 0:
            print(f"iteration={iteration} residual={residual:.6e} outlet_temperature={t_out:.6f}")
        if residual < 1e-10 and iteration > 50:
            break
    pressure_rise = float(np.mean(pressure[-1, :]) - np.mean(pressure[0, :]))
    pump_head = pressure_rise / (rho * gravity)
    outlet_temperature = float(np.mean(temperature[-1, :]))
    inlet_area = 2.0 * math.pi * r1 * depth
    mass_flow = rho * inlet_velocity * inlet_area
    heat_gain = mass_flow * cp * (outlet_temperature - t_in)
    speed_abs = np.sqrt(u_abs_x * u_abs_x + u_abs_y * u_abs_y)
    speed_rel = np.sqrt(u_rel_x * u_rel_x + u_rel_y * u_rel_y)
    write_structured_grid_vtk("pump_annulus_srf.vtk", x, y, {"temperature": temperature, "pressure": pressure, "absolute_speed": speed_abs, "relative_speed": speed_rel, "radius": rr}, {"absolute_velocity": (u_abs_x, u_abs_y), "relative_velocity": (u_rel_x, u_rel_y)}, "Centrifugal pump annular SRF model")
    circle_theta = np.linspace(0.0, 2.0 * math.pi, 361)
    inner_x = r1 * np.cos(circle_theta)
    inner_y = r1 * np.sin(circle_theta)
    outer_x = r2 * np.cos(circle_theta)
    outer_y = r2 * np.sin(circle_theta)
    write_multiline_vtk("pump_boundaries.vtk", [(inner_x, inner_y), (outer_x, outer_y)], "Pump annular boundaries")
    guides: list[tuple[np.ndarray, np.ndarray]] = []
    radial = np.linspace(r1, r2, 60)
    for angle in np.linspace(0.0, 2.0 * math.pi, 13)[:-1]:
        guides.append((radial * np.cos(angle), radial * np.sin(angle)))
    write_multiline_vtk("pump_flow_path_guides.vtk", guides, "Pump radial flow guides")
    blade_guides: list[tuple[np.ndarray, np.ndarray]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 9)[:-1]:
        blade_angle = angle + 0.75 * (radial - r1) / (r2 - r1)
        blade_guides.append((radial * np.cos(blade_angle), radial * np.sin(blade_angle)))
    write_multiline_vtk("pump_blade_guides.vtk", blade_guides, "Pump blade guides")
    seed_angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    write_points_vtk("pump_stream_seeds.vtk", 1.05 * r1 * np.cos(seed_angles), 1.05 * r1 * np.sin(seed_angles), title="Pump stream seeds")
    with open("pump_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "residual_linf", "minimum_temperature", "maximum_temperature", "outlet_temperature"])
        writer.writerows(history)
    with open("pump_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Simplified 2D annular single-rotating-frame centrifugal pump heat-transfer model\n")
        handle.write(f"nr {nr}\nnt {nt}\nrpm {rpm:.12g}\nomega_rad_per_s {omega:.12e}\nRe {reynolds:.12g}\n")
        handle.write(f"r1_m {r1:.12e}\nr2_m {r2:.12e}\nchannel_depth_m {depth:.12e}\n")
        handle.write(f"Tin_K {t_in:.12e}\nTwall_K {t_wall:.12e}\nTout_K {outlet_temperature:.12e}\n")
        handle.write(f"pressure_rise_Pa {pressure_rise:.12e}\npump_head_m {pump_head:.12e}\n")
        handle.write(f"inlet_velocity_m_per_s {inlet_velocity:.12e}\nmass_flow_rate_kg_per_s {mass_flow:.12e}\nfluid_heat_gain_W {heat_gain:.12e}\n")
        handle.write(f"final_residual_linf {history[-1][1]:.12e}\n")
        handle.write("model_scope teaching_level_reduced_order_srf\n")
    print("Wrote pump_annulus_srf.vtk, pump_boundaries.vtk, pump_flow_path_guides.vtk, pump_blade_guides.vtk, pump_stream_seeds.vtk, pump_history.csv, pump_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
