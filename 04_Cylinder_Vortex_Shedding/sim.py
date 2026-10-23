from __future__ import annotations

import csv
import math
import sys

import numpy as np

from common import require_finite, write_points_vtk, write_polyline_vtk, write_structured_grid_vtk


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, velocities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.empty((9, *rho.shape), dtype=float)
    speed_squared = ux * ux + uy * uy
    for q, (cx, cy) in enumerate(velocities):
        cu = cx * ux + cy * uy
        result[q] = weights[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared)
    return result


def estimate_strouhal(lift: np.ndarray, sample_spacing: float, diameter_lattice: float, inlet_speed_lattice: float) -> float:
    if lift.size < 64:
        return 0.0
    start = lift.size // 2
    signal = lift[start:] - np.mean(lift[start:])
    if float(np.std(signal)) < 1e-5:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
    frequency = np.fft.rfftfreq(signal.size, d=sample_spacing)
    strouhal = frequency * diameter_lattice / inlet_speed_lattice
    admissible = (strouhal >= 0.05) & (strouhal <= 0.35)
    if not np.any(admissible):
        return 0.0
    return float(strouhal[admissible][np.argmax(spectrum[admissible])])


def main() -> int:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 181
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 81
    reynolds = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    legacy_iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 45
    output_every = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    if nx < 31 or ny < 17 or reynolds < 20.0 or steps < 1 or legacy_iterations < 1 or output_every < 1:
        raise ValueError("usage: python exp04_cylinder_karman_vortex.py nx ny Re steps numerical_iterations output_every")
    substeps = max(1, min(4, int(round(legacy_iterations / 10.0))))
    inlet_speed = 0.10
    diameter = max(4, int(round(ny / 8.0)))
    radius = 0.5 * diameter
    center_x = int(round(0.25 * nx))
    center_y = 0.5 * (ny - 1)
    nu = inlet_speed * diameter / reynolds
    tau = 0.5 + 3.0 * nu
    if tau <= 0.505:
        raise ValueError("LBM relaxation time is too close to 0.5; increase ny or reduce Reynolds number")
    relaxation = 1.0 / tau
    velocities = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
    weights = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)
    j_grid, i_grid = np.mgrid[0:ny, 0:nx]
    solid = (i_grid - center_x) ** 2 + (j_grid - center_y) ** 2 <= radius * radius
    fluid = ~solid
    rho = np.ones((ny, nx), dtype=float)
    ux = np.full((ny, nx), inlet_speed, dtype=float)
    uy = 0.05 * inlet_speed * np.sin(2.0 * math.pi * j_grid / ny) * np.exp(-((i_grid - center_x) / (2.0 * diameter)) ** 2)
    ux[solid] = 0.0
    uy[solid] = 0.0
    populations = equilibrium(rho, ux, uy, velocities, weights)
    populations[:, solid] = 0.0
    inlet_rho = np.ones((ny, 1), dtype=float)
    inlet_ux = np.full((ny, 1), inlet_speed, dtype=float)
    inlet_uy = np.zeros((ny, 1), dtype=float)
    inlet_equilibrium = equilibrium(inlet_rho, inlet_ux, inlet_uy, velocities, weights)[:, :, 0]
    history: list[list[float]] = []
    lattice_time = 0
    for outer_step in range(1, steps + 1):
        force_x_accumulated = 0.0
        force_y_accumulated = 0.0
        for _ in range(substeps):
            rho = np.sum(populations, axis=0)
            safe_rho = np.where(rho > 1e-30, rho, 1.0)
            ux = np.tensordot(velocities[:, 0], populations, axes=(0, 0)) / safe_rho
            uy = np.tensordot(velocities[:, 1], populations, axes=(0, 0)) / safe_rho
            ux[solid] = 0.0
            uy[solid] = 0.0
            local_equilibrium = equilibrium(safe_rho, ux, uy, velocities, weights)
            post_collision = populations - relaxation * (populations - local_equilibrium)
            post_collision[:, solid] = 0.0
            streamed = np.zeros_like(populations)
            for q, (cx, cy) in enumerate(velocities):
                destination_is_solid = fluid & np.roll(solid, shift=(-cy, -cx), axis=(0, 1))
                source = post_collision[q]
                force_x_accumulated += 2.0 * float(np.sum(source[destination_is_solid])) * cx
                force_y_accumulated += 2.0 * float(np.sum(source[destination_is_solid])) * cy
                streamed[q] += np.roll(source * (~destination_is_solid), shift=(cy, cx), axis=(0, 1))
                streamed[opposite[q]][destination_is_solid] += source[destination_is_solid]
            populations = streamed
            populations[:, :, 0] = inlet_equilibrium
            populations[:, :, -1] = populations[:, :, -2]
            populations[:, solid] = 0.0
            lattice_time += 1
        rho = np.sum(populations, axis=0)
        safe_rho = np.where(rho > 1e-30, rho, 1.0)
        ux = np.tensordot(velocities[:, 0], populations, axes=(0, 0)) / safe_rho
        uy = np.tensordot(velocities[:, 1], populations, axes=(0, 0)) / safe_rho
        ux[solid] = 0.0
        uy[solid] = 0.0
        denominator = 0.5 * inlet_speed * inlet_speed * diameter * substeps
        drag = force_x_accumulated / denominator
        lift = force_y_accumulated / denominator
        fluid_density = rho[fluid]
        maximum_speed = float(np.max(np.sqrt(ux * ux + uy * uy)))
        density_deviation = float(np.max(np.abs(fluid_density - 1.0)))
        history.append([outer_step, lattice_time, lattice_time * inlet_speed / diameter, drag, lift, maximum_speed, float(np.min(fluid_density)), float(np.max(fluid_density)), density_deviation])
        require_finite(rho[fluid], ux[fluid], uy[fluid])
        if maximum_speed > 0.45 or density_deviation > 0.25:
            raise FloatingPointError(f"LBM solution became nonphysical at outer step {outer_step}")
        if outer_step % output_every == 0 or outer_step == steps:
            physical_x = i_grid / diameter
            physical_y = (j_grid - center_y) / diameter
            pressure = np.where(fluid, (rho - 1.0) / 3.0, 0.0)
            vorticity = np.zeros_like(ux)
            vorticity[1:-1, 1:-1] = (uy[1:-1, 2:] - uy[1:-1, :-2]) / 2.0 - (ux[2:, 1:-1] - ux[:-2, 1:-1]) / 2.0
            vorticity *= diameter / inlet_speed
            write_structured_grid_vtk(f"cylinder_vortex_{outer_step:06d}.vtk", physical_x, physical_y, {"pressure_lbm": pressure, "density": np.where(fluid, rho, 1.0), "vorticity_nondimensional": vorticity, "solid_mask": solid.astype(float), "speed_lattice": np.sqrt(ux * ux + uy * uy)}, {"velocity_lattice": (ux, uy)}, "D2Q9 LBM cylinder vortex shedding")
        if outer_step == 1 or outer_step == steps or outer_step % max(1, steps // 8) == 0:
            print(f"step={outer_step} lattice_time={lattice_time} Cd={drag:.6e} Cl={lift:.6e} max_speed={maximum_speed:.6e} density_deviation={density_deviation:.6e}")
    physical_x = i_grid / diameter
    physical_y = (j_grid - center_y) / diameter
    pressure = np.where(fluid, (rho - 1.0) / 3.0, 0.0)
    vorticity = np.zeros_like(ux)
    vorticity[1:-1, 1:-1] = (uy[1:-1, 2:] - uy[1:-1, :-2]) / 2.0 - (ux[2:, 1:-1] - ux[:-2, 1:-1]) / 2.0
    vorticity *= diameter / inlet_speed
    write_structured_grid_vtk("cylinder_vortex_final.vtk", physical_x, physical_y, {"pressure_lbm": pressure, "density": np.where(fluid, rho, 1.0), "vorticity_nondimensional": vorticity, "solid_mask": solid.astype(float), "speed_lattice": np.sqrt(ux * ux + uy * uy)}, {"velocity_lattice": (ux, uy)}, "D2Q9 LBM cylinder vortex shedding final")
    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    wall_x = center_x / diameter + 0.5 * np.cos(theta)
    wall_y = 0.5 * np.sin(theta)
    write_polyline_vtk("cylinder_wall.vtk", wall_x, wall_y, "Cylinder wall", close=True)
    seed_y = np.linspace(float(np.min(physical_y)) + 0.2, float(np.max(physical_y)) - 0.2, 30)
    write_points_vtk("cylinder_stream_seeds.vtk", np.full_like(seed_y, 0.5), seed_y, title="Cylinder stream seeds")
    with open("cylinder_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["outer_step", "lattice_time", "dimensionless_time_Ut_over_D", "Cd_momentum_exchange", "Cl_momentum_exchange", "maximum_lattice_speed", "minimum_density", "maximum_density", "maximum_density_deviation"])
        writer.writerows(history)
    lift_history = np.asarray(history)[:, 4]
    strouhal = estimate_strouhal(lift_history, float(substeps), float(diameter), inlet_speed)
    with open("cylinder_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("D2Q9 BGK lattice-Boltzmann cylinder vortex-shedding solver\n")
        handle.write(f"nx {nx}\nny {ny}\nRe {reynolds:.12g}\nouter_steps {steps}\nsubsteps_per_outer_step {substeps}\nlattice_time {lattice_time}\n")
        handle.write(f"cylinder_diameter_lattice {diameter}\ninlet_speed_lattice {inlet_speed:.12e}\nkinematic_viscosity_lattice {nu:.12e}\nrelaxation_time_tau {tau:.12e}\n")
        handle.write(f"final_dimensionless_time_Ut_over_D {lattice_time * inlet_speed / diameter:.12e}\nestimated_Strouhal {strouhal:.12e}\n")
        handle.write("expected_Re100_reference_Strouhal 0.16_to_0.17\n")
        handle.write("force_method momentum_exchange\n")
        handle.write("top_and_bottom_boundary periodic_far_field_approximation\n")
    print(f"Estimated Strouhal={strouhal:.6f}")
    print("Wrote cylinder_vortex_final.vtk, cylinder_wall.vtk, cylinder_stream_seeds.vtk, cylinder_history.csv, cylinder_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
