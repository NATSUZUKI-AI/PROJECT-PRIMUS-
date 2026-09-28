"""
evolve.py — Project Primus: True Vector Sensing, Curriculum, & Metabolism.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import pickle
import zlib
import mujoco
import numpy as np

from brain import FastNeuropil, Morphology
from world import (
    ISLANDS,
    DynamicAgentLayout,
    WorldConfig,
    build_dynamic_swarm_model,
    scattered_agent_spawns,
    scattered_food_positions,
    predator_spawn_positions,
    quat_to_yaw,
    steer_towards,
)
from gemini_director import consult_director, DirectorAdjustments

GENERATIONS = 200
SIM_STEPS = 1000
N_NEURONS = 100_000
FAN_IN = 25
DECIMATION = 8
MASTER_SEED = 12345

# --- Lethal Metabolism (Extended Window) ---
ENERGY_MAX = 100.0
ENERGY_START = 60.0          # Gives them ~500-600 ticks of runway
ENERGY_BASE_DECAY = 0.05     # Idle death now happens around tick 600
ENERGY_MOVE_COST = 0.4       # Less punishing movement penalty
FOOD_ENERGY_GAIN = 45.0      # Successful forage gives more sustainable life
FOOD_QUEUE_LEN = 200

# --- Gemini director ---
DIRECTOR_INTERVAL = 5
CHECKPOINT_PATH = "primus_checkpoint.pkl"
CHECKPOINT_INTERVAL = 10

CFG = WorldConfig(num_agents=12)
VALIDATION_SEEDS = [1001, 2002, 3003]

@dataclass
class AgentMetrics:
    score: float = 0.0
    foods_eaten: float = 0.0
    danger_ticks: float = 0.0
    survival_ticks: float = 0.0
    survived_full_trial: float = 0.0
    water_death: int = 0
    pred_death: int = 0
    starve_death: int = 0
    phenotype_drift: float = 0.0
    distance_traveled: float = 0.0
    unique_cells_visited: int = 0

def is_point_on_any_island(x: float, y: float, margin: float = 0.0) -> bool:
    for isl in ISLANDS:
        cx, cy = isl["center"]
        hx, hy = isl["half_extent"]
        if abs(x - cx) <= (hx - margin) and abs(y - cy) <= (hy - margin):
            return True
    return False

def precompute_ids(model: mujoco.MjModel, num_foods: int, num_predators: int) -> tuple[np.ndarray, list[int]]:
    food_mocap_ids = np.array([model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"food_body_{f}")] for f in range(num_foods)])
    pred_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"predator_{p}") for p in range(num_predators)]
    return food_mocap_ids, pred_body_ids

def run_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: DynamicAgentLayout,
    population: list[FastNeuropil],
    trial_seed: int,
    food_mocap_ids: np.ndarray,
    pred_body_ids: list[int],
    predator_force_mult: float = 1.0,
    energy_decay_mult: float = 1.0,
    food_energy_mult: float = 1.0,
) -> tuple[list[AgentMetrics], int]:
    
    mujoco.mj_resetData(model, data)
    data.qvel.fill(0.0)
    data.qacc.fill(0.0)
    data.ctrl.fill(0.0)
    data.xfrc_applied.fill(0.0)

    trial_rng = np.random.default_rng(trial_seed)
    num_agents = len(population)
    num_predators = layout.num_predators
    alive_status = np.ones(num_agents, dtype=bool)

    # Isolated sub-RNGs prevent early deaths from shifting noise across peers
    agent_rngs = [np.random.default_rng(int(trial_rng.integers(0, 2**31))) for _ in range(num_agents)]

    foods_eaten = np.zeros(num_agents)
    danger_ticks = np.zeros(num_agents)
    survival_ticks = np.zeros(num_agents)
    water_death = np.zeros(num_agents, dtype=int)
    pred_death = np.zeros(num_agents, dtype=int)
    starve_death = np.zeros(num_agents, dtype=int)
    distance_traveled = np.zeros(num_agents)
    visited_cells = [set() for _ in range(num_agents)]
    energy = np.full(num_agents, ENERGY_START, dtype=np.float32)

    food_positions = scattered_food_positions(CFG.num_foods, seed=int(trial_rng.integers(0, 2**31)))
    data.mocap_pos[food_mocap_ids] = food_positions

    food_queues = [scattered_food_positions(FOOD_QUEUE_LEN, seed=int(trial_rng.integers(0, 2**31))) for _ in range(CFG.num_foods)]
    food_queue_idx = np.zeros(CFG.num_foods, dtype=int)

    pred_spawns = predator_spawn_positions(CFG.predators)
    for p in range(num_predators):
        qb = layout.pred_qpos_base(p)
        data.qpos[qb : qb + 3] = pred_spawns[p]
        data.qpos[qb + 3 : qb + 7] = [1.0, 0.0, 0.0, 0.0]

    spawns = scattered_agent_spawns(num_agents, seed=int(trial_rng.integers(0, 2**31)))
    slot_for_index = trial_rng.permutation(num_agents)
    yaws = trial_rng.uniform(0.0, 2.0 * np.pi, num_agents)

    for i in range(num_agents):
        population[i].reset_state()
        q_idx = layout.qpos_indices[i]
        data.qpos[q_idx[0:3]] = spawns[slot_for_index[i]]
        half_yaw = yaws[i] / 2.0
        data.qpos[q_idx[3:7]] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        if len(q_idx) > 7:
            data.qpos[q_idx[7:]] = 0.0

    mujoco.mj_forward(model, data)

    pred_headings = [np.array([1.0, 0.0], dtype=np.float32) for _ in range(num_predators)]
    patrol_angles = np.array([i * (2.0 * np.pi / max(num_predators, 1)) for i in range(num_predators)], dtype=np.float32)
    home_positions = np.array([
        ISLANDS[pt.home_island]["center"] if pt.home_island is not None else (0.0, 0.0)
        for pt in CFG.predators
    ], dtype=np.float32)

    prev_pos = np.array([data.qpos[layout.qpos_indices[i][0:2]] for i in range(num_agents)])

    for step in range(SIM_STEPS):
        data.xfrc_applied.fill(0.0)
        pred_positions = np.array([data.qpos[layout.pred_qpos_base(p) : layout.pred_qpos_base(p) + 2] for p in range(num_predators)])
        pred_zs = np.array([data.qpos[layout.pred_qpos_base(p) + 2] for p in range(num_predators)])

        closest_agent_pos: list[np.ndarray | None] = [None] * num_predators
        min_dist_per_pred = np.full(num_predators, 999.0)

        should_update_brain = (step % DECIMATION == 0)

        for i in range(num_agents):
            if not alive_status[i]:
                continue

            q_idx = layout.qpos_indices[i]
            pos = data.qpos[q_idx[0:3]]
            ctrl_idx = layout.ctrl_indices[i]

            step_dist = float(np.hypot(pos[0] - prev_pos[i, 0], pos[1] - prev_pos[i, 1]))
            distance_traveled[i] += step_dist
            prev_pos[i] = pos[:2]
            visited_cells[i].add((int(np.floor(pos[0] / 0.5)), int(np.floor(pos[1] / 0.5))))

            # Biological metabolic scaling + 0.02 flat buffer
            bw, bl, bh = population[i].morphology.body_scale
            mass_factor = bw * bl * bh
            motor_penalty = 0.1 * population[i].morphology.n_motors
            
            step_decay = energy_decay_mult * (
                0.02 + 
                (ENERGY_BASE_DECAY * mass_factor * (1.0 + motor_penalty)) + 
                (ENERGY_MOVE_COST * step_dist * mass_factor)
            )
            energy[i] -= step_decay

            dists_to_preds = np.linalg.norm(pred_positions - pos[:2], axis=1) if num_predators else np.array([])
            for p in range(num_predators):
                if dists_to_preds[p] < min_dist_per_pred[p]:
                    min_dist_per_pred[p] = dists_to_preds[p]
                    closest_agent_pos[p] = pos[:2].copy()
            dist_to_pred = float(dists_to_preds.min()) if dists_to_preds.size else 999.0

            def _kill_agent(death_array):
                alive_status[i] = False
                death_array[i] = 1
                for c in ctrl_idx:
                    data.ctrl[c] = 0.0

            if pos[2] < 0.08:
                _kill_agent(water_death)
                continue

            if dist_to_pred < 0.16:
                _kill_agent(pred_death)
                continue

            dists_center = np.linalg.norm(food_positions[:, :2] - pos[:2], axis=1)
            nearest_idx = np.argmin(dists_center)
            nearest_dist = dists_center[nearest_idx]

            # Eat food before starvation test
            if nearest_dist < 0.18:
                foods_eaten[i] += 1.0
                energy[i] = min(ENERGY_MAX, energy[i] + FOOD_ENERGY_GAIN * food_energy_mult)
                slot = int(nearest_idx)
                queue = food_queues[slot]
                new_food_pos = queue[food_queue_idx[slot] % len(queue)]
                food_queue_idx[slot] += 1
                food_positions[slot] = new_food_pos
                data.mocap_pos[food_mocap_ids[slot]] = new_food_pos

            if energy[i] <= 0.0:
                _kill_agent(starve_death)
                continue

            survival_ticks[i] += 1.0
            if dist_to_pred < 1.5:
                danger_ticks[i] += 1.0

            if should_update_brain:
                qw, qx, qy, qz = data.qpos[q_idx[3:7]]
                yaw = quat_to_yaw(qw, qx, qy, qz)

                # --- TRUE VECTOR SENSING ---
                # 1. Food Vector
                dx_f = food_positions[nearest_idx, 0] - pos[0]
                dy_f = food_positions[nearest_idx, 1] - pos[1]
                
                # Rotate into agent's local frame (X is forward, Y is left)
                local_food_x = dx_f * np.cos(-yaw) - dy_f * np.sin(-yaw)
                local_food_y = dx_f * np.sin(-yaw) + dy_f * np.cos(-yaw)
                
                food_intensity = float(np.exp(-nearest_dist * 0.8))
                food_dir_x = local_food_x / (nearest_dist + 1e-4) # -1.0 to 1.0 (Forward/Back)
                food_dir_y = local_food_y / (nearest_dist + 1e-4) # -1.0 to 1.0 (Left/Right)

                # 2. Predator Vector
                if num_predators > 0:
                    nearest_pred_idx = np.argmin(dists_to_preds)
                    dx_p = pred_positions[nearest_pred_idx, 0] - pos[0]
                    dy_p = pred_positions[nearest_pred_idx, 1] - pos[1]
                    local_pred_x = dx_p * np.cos(-yaw) - dy_p * np.sin(-yaw)
                    local_pred_y = dx_p * np.sin(-yaw) + dy_p * np.cos(-yaw)
                    
                    pred_intensity = float(np.exp(-dists_to_preds[nearest_pred_idx] * 0.8))
                    pred_dir_x = local_pred_x / (dists_to_preds[nearest_pred_idx] + 1e-4)
                    pred_dir_y = local_pred_y / (dists_to_preds[nearest_pred_idx] + 1e-4)
                else:
                    pred_intensity, pred_dir_x, pred_dir_y = 0.0, 0.0, 0.0

                probe_x = pos[0] + 0.10 * np.cos(yaw)
                probe_y = pos[1] + 0.10 * np.sin(yaw)
                cliff_alarm = 0.0 if is_point_on_any_island(probe_x, probe_y, margin=0.05) else 1.0

                hunger = float(np.clip(1.0 - energy[i] / ENERGY_MAX, 0.0, 1.0))
                dopamine = (8.0 if nearest_dist < 0.18 else (0.2 if nearest_dist < 0.6 else 0.0)) * (0.5 + hunger)
                octopamine = 1.0 if cliff_alarm > 0.5 else (0.5 if dist_to_pred < 0.8 else 0.0)

                # Map directly into the 9 existing channels to preserve genome compatibility
                sensors = np.array([
                    food_intensity, food_dir_x, food_dir_y,  
                    pred_intensity, pred_dir_x, pred_dir_y,  
                    pos[2], cliff_alarm, hunger,
                ], dtype=np.float32)

                motors = population[i].forward(sensors, dopamine, octopamine, rng=agent_rngs[i])
                for m_idx, c_id in enumerate(ctrl_idx):
                    if m_idx < len(motors):
                        data.ctrl[c_id] = motors[m_idx]

        # Predator AI
        for p, pt in enumerate(CFG.predators):
            body_id = pred_body_ids[p]
            pred_pos = pred_positions[p]
            pred_z = pred_zs[p]
            qvb = layout.pred_qvel_base(p)

            if pred_z < pt.hover_z + 0.15:
                z_err = pt.hover_z - pred_z
                vz = data.qvel[qvb + 2]
                if pred_z < CFG.water_surface_z:
                    data.qvel[qvb : qvb + 3] *= 0.90
                data.xfrc_applied[body_id][2] = 9.81 * pt.mass + pt.hover_kp * z_err - pt.hover_kd * vz
            else:
                data.xfrc_applied[body_id][2] = 0.0

            if step < pt.grace_steps:
                continue

            engaged = closest_agent_pos[p] is not None and min_dist_per_pred[p] < pt.sight_range
            force = pt.force * predator_force_mult

            if pt.behavior == "hunter":
                if engaged:
                    pred_dir = closest_agent_pos[p] - pred_positions[p]
                    dist = np.linalg.norm(pred_dir)
                    if dist > 0.05:
                        target_dir = pred_dir / dist
                        pred_headings[p] = steer_towards(pred_headings[p], target_dir, pt.turn_rate * CFG.timestep)
                        thrust = force * (1.8 if pred_z < 0.10 else 1.0)
                        data.xfrc_applied[body_id][0] = pred_headings[p][0] * thrust
                        data.xfrc_applied[body_id][1] = pred_headings[p][1] * thrust
                else:
                    data.qvel[qvb : qvb + 2] *= 0.85

            elif pt.behavior == "skimmer":
                if engaged:
                    pred_dir = closest_agent_pos[p] - pred_positions[p]
                    dist = np.linalg.norm(pred_dir)
                    if dist > 0.05:
                        target_dir = pred_dir / dist
                        pred_headings[p] = steer_towards(pred_headings[p], target_dir, pt.turn_rate * CFG.timestep)
                        thrust = force * pt.dash_multiplier
                        data.xfrc_applied[body_id][0] = pred_headings[p][0] * thrust
                        data.xfrc_applied[body_id][1] = pred_headings[p][1] * thrust
                else:
                    patrol_angles[p] += 0.01
                    target_dir = np.array([np.cos(patrol_angles[p]), np.sin(patrol_angles[p])], dtype=np.float32)
                    pred_headings[p] = steer_towards(pred_headings[p], target_dir, pt.turn_rate * CFG.timestep)
                    thrust = force * 0.45
                    data.xfrc_applied[body_id][0] = pred_headings[p][0] * thrust
                    data.xfrc_applied[body_id][1] = pred_headings[p][1] * thrust

            elif pt.behavior == "ambusher":
                if engaged:
                    pred_dir = closest_agent_pos[p] - pred_positions[p]
                    dist = np.linalg.norm(pred_dir)
                    if dist > 0.05:
                        target_dir = pred_dir / dist
                        pred_headings[p] = steer_towards(pred_headings[p], target_dir, pt.turn_rate * CFG.timestep)
                        thrust = force * pt.dash_multiplier
                        data.xfrc_applied[body_id][0] = pred_headings[p][0] * thrust
                        data.xfrc_applied[body_id][1] = pred_headings[p][1] * thrust
                else:
                    home_dir = home_positions[p] - pred_positions[p]
                    dist = np.linalg.norm(home_dir)
                    if dist > 0.15:
                        target_dir = home_dir / dist
                        pred_headings[p] = steer_towards(pred_headings[p], target_dir, pt.turn_rate * CFG.timestep)
                        thrust = force * 0.3
                        data.xfrc_applied[body_id][0] = pred_headings[p][0] * thrust
                        data.xfrc_applied[body_id][1] = pred_headings[p][1] * thrust
                    else:
                        data.qvel[qvb : qvb + 2] *= 0.8

        mujoco.mj_step(model, data)
        for p, pt in enumerate(CFG.predators):
            qvb = layout.pred_qvel_base(p)
            speed = np.linalg.norm(data.qvel[qvb : qvb + 2])
            if speed > pt.max_speed:
                data.qvel[qvb : qvb + 2] *= pt.max_speed / speed

        if np.sum(alive_status) == 0:
            break

    # Evaluate survival cleanly post-simulation
    survived_full_trial = alive_status.astype(float)
    metrics_list = []

    for i in range(num_agents):
        exploration_score = float(len(visited_cells[i]))
        
        # Progressive Multiplier: Exploration is heavily rewarded ONLY if successful foraging occurs
        if foods_eaten[i] <= 0:
            explore_mult = 0.5
        else:
            explore_mult = min(5.0, 1.0 + foods_eaten[i])

        score = (
            (foods_eaten[i] * 400.0)
            + (exploration_score * explore_mult)
            + (survival_ticks[i] * 0.05)
            + (survived_full_trial[i] * 100.0)
            - (danger_ticks[i] * 0.5)
        )
        drift = float(np.linalg.norm(population[i].W_out - population[i].W_out_base))

        metrics_list.append(AgentMetrics(
            score=score,
            foods_eaten=foods_eaten[i],
            danger_ticks=danger_ticks[i],
            survival_ticks=survival_ticks[i],
            survived_full_trial=survived_full_trial[i],
            water_death=water_death[i],
            pred_death=pred_death[i],
            starve_death=starve_death[i],
            phenotype_drift=drift,
            distance_traveled=distance_traveled[i],
            unique_cells_visited=int(exploration_score),
        ))

    return metrics_list, int(np.sum(alive_status))

def _fresh_state(master_rng: np.random.Generator) -> dict:
    population = [
        FastNeuropil(
            n_neurons=N_NEURONS, 
            fan_in=FAN_IN, 
            rng=np.random.default_rng(int(master_rng.integers(0, 2**31)))
        ) 
        for _ in range(CFG.num_agents)
    ]
    return dict(
        generation=0,
        population=population,
        hall_of_fame=None,
        hall_of_fame_score=-9999.0,
        archive={},
        gen_history=[],
        rng_state=master_rng.bit_generator.state,
        director_params=dict(mutation_scale=1.0, predator_aggression=1.0, food_energy=1.0, energy_pressure=1.0),
    )

def _load_state(master_rng: np.random.Generator) -> dict:
    if not os.path.exists(CHECKPOINT_PATH):
        return _fresh_state(master_rng)
    try:
        with open(CHECKPOINT_PATH, "rb") as f:
            state = pickle.load(f)
        if "rng_state" in state:
            master_rng.bit_generator.state = state["rng_state"]
        print(f"Resuming from {CHECKPOINT_PATH} at generation {state['generation'] + 1}/{GENERATIONS}.")
        return state
    except Exception as exc:  # noqa: BLE001
        print(f"Checkpoint load failed ({exc!r}) -- starting a fresh run instead.")
        return _fresh_state(master_rng)

def _save_state(state: dict, master_rng: np.random.Generator) -> None:
    try:
        state["rng_state"] = master_rng.bit_generator.state
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(state, f)
    except Exception as exc:  # noqa: BLE001
        print(f"  [>] Checkpoint save failed: {exc!r}")

def main() -> None:
    master_rng = np.random.default_rng(MASTER_SEED)
    state = _load_state(master_rng)

    population: list[FastNeuropil] = state["population"]
    hall_of_fame: FastNeuropil | None = state["hall_of_fame"]
    hall_of_fame_score: float = state["hall_of_fame_score"]
    archive: dict[str, tuple[float, FastNeuropil]] = state["archive"]
    gen_history: list[dict] = state["gen_history"]
    director_adj = DirectorAdjustments(**state["director_params"])
    start_gen = state["generation"]

    if start_gen == 0:
        print("Starting Project Primus (Dynamic Morphology | Co-Evolution)...")
    print(f"Predators: {', '.join(pt.name for pt in CFG.predators)}")
    print("-" * 85)

    for gen in range(start_gen, GENERATIONS):
        # 1. Compile Physics Model Dynamically
        morphologies = [p.morphology for p in population]
        dummy_foods = scattered_food_positions(CFG.num_foods)
        model = build_dynamic_swarm_model(CFG, dummy_foods, morphologies)
        data = mujoco.MjData(model)
        
        # 2. Extract Exact Joint Layouts from compiled Model
        layout = DynamicAgentLayout.from_model(model, CFG.num_agents, CFG.num_predators)
        food_mocap_ids, pred_body_ids = precompute_ids(model, CFG.num_foods, CFG.num_predators)

        # 3. Brain <-> Physics Invariant Assertions
        for i, agent in enumerate(population):
            expected = agent.morphology.n_motors
            actual = len(layout.ctrl_indices[i])
            assert actual == expected, f"Agent {i}: morphology says {expected} motors, MuJoCo compiled {actual}"
            assert len(layout.qpos_indices[i]) >= 7, f"Agent {i} qpos missing freejoint"
            assert len(layout.qvel_indices[i]) >= 6, f"Agent {i} qvel missing freejoint"

        # Deterministic cross-platform seed derivation
        seed_A = zlib.crc32(f"{MASTER_SEED}_{gen}_0".encode())
        seed_B = zlib.crc32(f"{MASTER_SEED}_{gen}_1".encode())

        # Curriculum injected into TRAINING loops via Director scaling
        metrics_A, surv_A = run_trial(
            model, data, layout, population, trial_seed=seed_A,
            food_mocap_ids=food_mocap_ids, pred_body_ids=pred_body_ids,
            predator_force_mult=director_adj.predator_aggression,
            energy_decay_mult=director_adj.energy_pressure,
            food_energy_mult=director_adj.food_energy,
        )
        metrics_B, surv_B = run_trial(
            model, data, layout, population, trial_seed=seed_B,
            food_mocap_ids=food_mocap_ids, pred_body_ids=pred_body_ids,
            predator_force_mult=director_adj.predator_aggression,
            energy_decay_mult=director_adj.energy_pressure,
            food_energy_mult=director_adj.food_energy,
        )

        avg_scores = np.array([(a.score + b.score) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        avg_foods = np.array([(a.foods_eaten + b.foods_eaten) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        avg_survival = np.array([(a.survival_ticks + b.survival_ticks) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        avg_danger = np.array([(a.danger_ticks + b.danger_ticks) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        avg_distance = np.array([(a.distance_traveled + b.distance_traveled) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        avg_drift = np.array([(a.phenotype_drift + b.phenotype_drift) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        avg_cells = np.array([(a.unique_cells_visited + b.unique_cells_visited) / 2.0 for a, b in zip(metrics_A, metrics_B)])
        evader_scores = avg_survival - avg_danger * 2.0

        total_survivors = surv_A + surv_B
        total_water_deaths = sum((a.water_death + b.water_death) for a, b in zip(metrics_A, metrics_B))
        total_pred_deaths = sum((a.pred_death + b.pred_death) for a, b in zip(metrics_A, metrics_B))
        total_starve_deaths = sum((a.starve_death + b.starve_death) for a, b in zip(metrics_A, metrics_B))

        sorted_indices = np.argsort(avg_scores)[::-1]
        best_idx, second_idx, third_idx = sorted_indices[0], sorted_indices[1], sorted_indices[2]

        gen_best_score = avg_scores[best_idx]
        elites = [
            population[best_idx].clone(rng=master_rng),
            population[second_idx].clone(rng=master_rng),
            population[third_idx].clone(rng=master_rng),
        ]

        if gen_best_score > hall_of_fame_score:
            hall_of_fame_score = gen_best_score
            hall_of_fame = elites[0].clone(rng=master_rng)

        def _maybe_archive(trait: str, values: np.ndarray) -> None:
            idx = int(np.argmax(values))
            value = float(values[idx])
            if trait not in archive or value > archive[trait][0]:
                archive[trait] = (value, population[idx].clone(rng=master_rng))

        _maybe_archive("overall", avg_scores)
        _maybe_archive("forager", avg_foods)
        _maybe_archive("survivor", avg_survival)
        _maybe_archive("evader", evader_scores)
        _maybe_archive("explorer", avg_cells)

        best_m = elites[0].morphology
        best_agent_metrics = AgentMetrics(
            foods_eaten=(metrics_A[best_idx].foods_eaten + metrics_B[best_idx].foods_eaten) / 2.0,
            danger_ticks=(metrics_A[best_idx].danger_ticks + metrics_B[best_idx].danger_ticks) / 2.0,
            phenotype_drift=(metrics_A[best_idx].phenotype_drift + metrics_B[best_idx].phenotype_drift) / 2.0,
        )

        print(f"Gen {gen + 1:02d}/{GENERATIONS} | Gen Best: {gen_best_score:7.1f} | HoF: {hall_of_fame_score:7.1f} | Surv: {total_survivors}/{CFG.num_agents * 2}")
        print(f"  -> Champ Morphology: {best_m.wheel_count}-wheel, {len(best_m.limbs)}-limbs | Motors: {best_m.n_motors} | Scale: {best_m.body_scale[0]:.2f}x{best_m.body_scale[1]:.2f}")
        print(f"  -> Deaths: Water {total_water_deaths} | Predator {total_pred_deaths} | Starvation {total_starve_deaths}")
        print(f"  -> Gen Champ: Foods {best_agent_metrics.foods_eaten:.1f} | Danger Ticks {best_agent_metrics.danger_ticks:.0f} | Drift {best_agent_metrics.phenotype_drift:.4f}")
        print(f"  -> Archive: " + " | ".join(f"{t}={v:.1f}" for t, (v, _) in archive.items()))

        # Challenge trial: Baseline 1.0 Benchmark Check (Ignores Director tweaks to stay comparable)
        challenge_score = None
        if (gen + 1) % DIRECTOR_INTERVAL == 0 and hall_of_fame is not None:
            chal_cfg = WorldConfig(num_agents=1, num_foods=CFG.num_foods, predators=CFG.predators)
            chal_model = build_dynamic_swarm_model(chal_cfg, dummy_foods, [hall_of_fame.morphology])
            chal_data = mujoco.MjData(chal_model)
            chal_layout = DynamicAgentLayout.from_model(chal_model, 1, chal_cfg.num_predators)
            chal_food_mocap_ids, chal_pred_body_ids = precompute_ids(chal_model, CFG.num_foods, CFG.num_predators)
            
            challenge_metrics, _ = run_trial(
                chal_model, chal_data, chal_layout, [hall_of_fame.clone(rng=master_rng)], trial_seed=42,
                food_mocap_ids=chal_food_mocap_ids, pred_body_ids=chal_pred_body_ids,
                predator_force_mult=1.0,
                energy_decay_mult=1.0,
                food_energy_mult=1.0,
            )
            challenge_score = float(challenge_metrics[0].score)
            print(f"  [>] Challenge trial (baseline env) HoF score: {challenge_score:.1f}")

        gen_history.append(dict(
            gen=gen + 1, gen_best_score=float(gen_best_score), hof_score=float(hall_of_fame_score),
            survivors=int(total_survivors), water_deaths=int(total_water_deaths),
            pred_deaths=int(total_pred_deaths), starve_deaths=int(total_starve_deaths),
            foods_mean=float(np.mean(avg_foods)), danger_mean=float(np.mean(avg_danger)),
            distance_mean=float(np.mean(avg_distance)), drift_mean=float(np.mean(avg_drift)),
            challenge_score=challenge_score,
        ))

        # Generalization validation on unseen seeds
        if (gen + 1) % 10 == 0 and hall_of_fame is not None:
            print("  [>] Running Generalization Validation on HoF (Unseen Seeds)...")
            val_scores = []
            val_cfg = WorldConfig(num_agents=1, num_foods=CFG.num_foods, predators=CFG.predators)
            val_model = build_dynamic_swarm_model(val_cfg, dummy_foods, [hall_of_fame.morphology])
            val_data = mujoco.MjData(val_model)
            val_layout = DynamicAgentLayout.from_model(val_model, 1, val_cfg.num_predators)
            val_food_mocap_ids, val_pred_body_ids = precompute_ids(val_model, CFG.num_foods, CFG.num_predators)

            for v_seed in VALIDATION_SEEDS:
                metrics_V, _ = run_trial(
                    val_model, val_data, val_layout, [hall_of_fame.clone(rng=master_rng)], trial_seed=v_seed,
                    food_mocap_ids=val_food_mocap_ids, pred_body_ids=val_pred_body_ids,
                )
                val_scores.append(metrics_V[0].score)
            print(f"  [>] Validation Scores: {[round(s, 1) for s in val_scores]} | Avg: {np.mean(val_scores):.1f}")

        if (gen + 1) % DIRECTOR_INTERVAL == 0:
            current_params = dict(
                mutation_scale=director_adj.mutation_scale,
                predator_aggression=director_adj.predator_aggression,
                food_energy=director_adj.food_energy,
                energy_pressure=director_adj.energy_pressure,
            )
            director_adj = consult_director(gen_history, current_params)

        print("-" * 85)

        # 4. Top-3 Selection with Modulo Parent Distribution & Separated Scale/Rate
        new_pop = [e.clone(rng=master_rng) for e in elites]
        if "forager" in archive and "evader" in archive:
            # Scale applies magnitude, rate applies probability
            new_pop.append(archive["forager"][1].mutate(rate=0.04, scale=director_adj.mutation_scale, rng=master_rng))
            new_pop.append(archive["evader"][1].mutate(rate=0.04, scale=director_adj.mutation_scale, rng=master_rng))

        for rate in [0.05, 0.10]:
            for e in elites:
                if len(new_pop) < CFG.num_agents:
                    new_pop.append(e.mutate(rate=rate, scale=director_adj.mutation_scale, rng=master_rng))

        idx = 0
        while len(new_pop) < CFG.num_agents:
            parent = elites[idx % 3]
            new_pop.append(parent.mutate(rate=0.20, scale=director_adj.mutation_scale, rng=master_rng))
            idx += 1

        population = new_pop

        if (gen + 1) % CHECKPOINT_INTERVAL == 0 or (gen + 1) == GENERATIONS:
            _save_state(dict(
                generation=gen + 1,
                population=population,
                hall_of_fame=hall_of_fame,
                hall_of_fame_score=hall_of_fame_score,
                archive=archive,
                gen_history=gen_history,
                director_params=dict(
                    mutation_scale=director_adj.mutation_scale,
                    predator_aggression=director_adj.predator_aggression,
                    food_energy=director_adj.food_energy,
                    energy_pressure=director_adj.energy_pressure,
                ),
            ), master_rng)
            print(f"  [>] Checkpoint saved -> {CHECKPOINT_PATH} (generation {gen + 1})")
            print("-" * 85)

    print(f"\nEvolution complete. Checkpoint saved to {CHECKPOINT_PATH}.")

if __name__ == "__main__":
    main()