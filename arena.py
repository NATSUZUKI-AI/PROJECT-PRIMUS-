"""
arena.py — Project Primus: legacy single-agent + predator test rig.

Note: this uses a different body plan (a 2-legged hinge creature) than
the wheeled swarm in world.py/evolve.py/watch.py, and nothing in this
project currently imports PrimusArena. Keeping it in case it's still
useful for quick single-agent smoke tests, but it is not part of the
swarm training pipeline.

Bug fixed: the agent body has a freejoint (7 qpos / 6 qvel) *plus* two
hinge joints (joint_l, joint_r), so the agent occupies qpos[0:9] /
qvel[0:8] — not qpos[0:7] / qvel[0:6] as the original comments assumed.
The predator's freejoint therefore starts at qpos[9] / qvel[8], not
qpos[7] / qvel[6]. The original offsets were writing into the agent's
own leg-joint qpos/qvel and only partially into the predator's freejoint,
which produced a degenerate (all-zero) orientation quaternion for the
predator and never actually moved it — mj_step would either raise on
the invalid quaternion or the predator would simply sit still.
"""

import mujoco
import numpy as np

MJCF_WORLD = """
<mujoco model="primus_archipelago">
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <worldbody>
    <geom name="water" type="plane" size="12 12 0.01" pos="0 0 0" rgba="0.1 0.3 0.6 0.8" contype="0" conaffinity="0"/>

    <geom name="island_1" type="box" size="1.8 1.8 0.1" pos="0 0 0.05" rgba="0.65 0.6 0.5 1" friction="0.8 0.1 0.1"/>
    <geom name="island_2" type="box" size="1.4 1.4 0.2" pos="3.0 2.0 0.1" rgba="0.45 0.45 0.45 1" friction="0.8 0.1 0.1"/>
    <geom name="island_3" type="box" size="1.2 1.2 0.15" pos="-2.5 -2.0 0.05" rgba="0.25 0.25 0.25 1" friction="0.8 0.1 0.1"/>

    <geom name="food_geom" type="sphere" size="0.08" pos="1.5 1.5 0.15" rgba="1.0 0.8 0.1 1" contype="0" conaffinity="0"/>

    <!-- Agent: freejoint (qpos 0:7 / qvel 0:6) + 2 hinges (qpos 7:9 / qvel 6:8) -->
    <body name="agent" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="torso" type="box" size="0.08 0.05 0.025" mass="0.25" rgba="0.15 0.15 0.15 1"/>

      <body name="left_limb" pos="-0.09 0 0">
        <joint name="joint_l" type="hinge" axis="0 1 0" range="-60 60" damping="0.05"/>
        <geom type="capsule" size="0.015 0.05" mass="0.03" rgba="0.1 0.8 0.2 1"/>
      </body>

      <body name="right_limb" pos="0.09 0 0">
        <joint name="joint_r" type="hinge" axis="0 1 0" range="-60 60" damping="0.05"/>
        <geom type="capsule" size="0.015 0.05" mass="0.03" rgba="0.1 0.8 0.2 1"/>
      </body>
    </body>

    <!-- Predator: freejoint starts at qpos 9 / qvel 8 (after the agent's 9 qpos / 8 qvel) -->
    <body name="predator" pos="-1.5 -1.5 0.3">
      <freejoint name="predator_root"/>
      <geom name="predator_geom" type="box" size="0.1 0.1 0.1" mass="1.0" rgba="0.9 0.1 0.1 1"/>
    </body>

  </worldbody>

  <actuator>
    <motor name="motor_l" joint="joint_l" ctrlrange="-1.8 1.8"/>
    <motor name="motor_r" joint="joint_r" ctrlrange="-1.8 1.8"/>
  </actuator>
</mujoco>
"""

# Derived once, in one place, instead of re-typed at every slice site.
AGENT_QPOS = slice(0, 9)   # freejoint (7) + joint_l + joint_r
AGENT_QVEL = slice(0, 8)   # freejoint (6) + joint_l + joint_r
PRED_QPOS = slice(9, 16)   # freejoint: x y z qw qx qy qz
PRED_QVEL = slice(8, 14)   # freejoint: vx vy vz wx wy wz


class PrimusArena:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MJCF_WORLD)
        self.data = mujoco.MjData(self.model)
        self.boundary_limit = 5.0
        self.food_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "food_geom")

    def reset(self, start_pos=(0.0, 0.0, 0.25)):
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = start_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:9] = [0.0, 0.0]  # leg joints start neutral

        self.data.qpos[9:12] = [-1.5, -1.5, 0.3]
        self.data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

    def step(self, motor_commands):
        self.data.ctrl[0] = motor_commands[0]
        self.data.ctrl[1] = motor_commands[1]

        agent_pos = self.data.qpos[0:3]
        pred_pos = self.data.qpos[9:12]

        direction = agent_pos - pred_pos
        direction[2] = 0
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        # Predator's linear velocity is qvel[8:10] (vx, vy) — see PRED_QVEL above.
        self.data.qvel[8:10] = direction[:2] * 0.5

        mujoco.mj_step(self.model, self.data)

        pos = self.data.qpos[0:3].copy()
        alive = self.check_alive(pos)
        return alive, pos

    def check_alive(self, pos):
        if abs(pos[0]) > self.boundary_limit or abs(pos[1]) > self.boundary_limit or pos[2] < 0.05:
            return False
        return True

    def get_sensors(self, pos, food_pos):
        self.model.geom_pos[self.food_geom_id][0] = food_pos[0]
        self.model.geom_pos[self.food_geom_id][1] = food_pos[1]

        dx = food_pos[0] - pos[0]
        dy = food_pos[1] - pos[1]
        angle_to_food = np.arctan2(dy, dx)
        tilt = self.data.qpos[4]

        dist_x_edge = self.boundary_limit - abs(pos[0])
        dist_y_edge = self.boundary_limit - abs(pos[1])

        return np.array([dist_x_edge, dist_y_edge, pos[2], angle_to_food, tilt], dtype=np.float32)