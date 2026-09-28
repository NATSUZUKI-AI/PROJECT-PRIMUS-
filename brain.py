"""
brain.py — Project Primus: True Liquid State Machine & Decoupled Mutation.
"""

from __future__ import annotations
import copy
import numpy as np
from scipy import sparse
from dataclasses import dataclass, field

N_SENSORS = 9
MAX_MOTORS = 12

@dataclass
class LimbGene:
    segments: int = 1  # Hard invariant for V1
    segment_length: float = 0.05
    joint_type: str = "hinge"  # Restricted to hinge for V1
    attachment: tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class Morphology:
    body_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    wheel_count: int = 2
    limbs: list[LimbGene] = field(default_factory=list)

    @property
    def n_motors(self) -> int:
        return self.wheel_count + sum(limb.segments for limb in self.limbs)

    def is_valid(self) -> bool:
        if self.wheel_count not in (0, 2, 3, 4): return False
        if any(d < 0.2 or d > 3.0 for d in self.body_scale): return False
        if self.n_motors > MAX_MOTORS: return False
        if self.n_motors == 0: return False  # Cannot be inert
        for limb in self.limbs:
            if limb.segments != 1: return False
            if limb.joint_type != "hinge": return False
            if limb.segment_length < 0.01 or limb.segment_length > 0.3: return False
        return True

    def mutate(self, rate: float, scale: float, rng: np.random.Generator) -> "Morphology":
        child = copy.deepcopy(self)
        
        # 1. Scale Mutation (Rate dictates IF it happens, Scale dictates HOW MUCH)
        if rng.random() < rate:
            idx = rng.integers(0, 3)
            new_scale = list(child.body_scale)
            new_scale[idx] = float(np.clip(new_scale[idx] + rng.normal(0, 0.1 * scale), 0.5, 2.0))
            child.body_scale = tuple(new_scale)
            
        # 2. Wheel Mutation (Structural)
        if rng.random() < (rate * 0.5):
            child.wheel_count = int(rng.choice([0, 2, 3, 4]))
            
        # 3. Limb Mutation (Structural vs Parametric)
        if rng.random() < (rate * 0.2):
            if child.limbs and rng.random() < 0.5:
                # Parametric mutation of an existing limb (uses scale)
                l_idx = rng.integers(0, len(child.limbs))
                limb = child.limbs[l_idx]
                if rng.random() < 0.5:
                    limb.segment_length = float(np.clip(limb.segment_length + rng.normal(0, 0.02 * scale), 0.02, 0.15))
                else:
                    ax, ay, az = limb.attachment
                    limb.attachment = (
                        float(np.clip(ax + rng.normal(0, 0.01 * scale), -0.05, 0.05)),
                        float(np.clip(ay + rng.normal(0, 0.01 * scale), -0.05, 0.05)),
                        az
                    )
            else:
                # Structural mutation (add/remove)
                if child.limbs and rng.random() < 0.3:
                    child.limbs.pop(rng.integers(0, len(child.limbs)))
                elif len(child.limbs) < 4:
                    child.limbs.append(LimbGene(
                        segments=1,
                        segment_length=float(np.clip(rng.normal(0.06, 0.02 * scale), 0.02, 0.15)),
                        attachment=(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0)
                    ))

        # Revert safely if invalid physics logic triggered
        return child if child.is_valid() else copy.deepcopy(self)

class FastNeuropil:
    def __init__(
        self,
        n_neurons: int = 100_000,  # Dropped from 100k to a tractable dimension
        leak_rate: float = 0.2,
        fan_in: int = 25,
        excitatory_frac: float = 0.8,
        morphology: Morphology | None = None,
        rng: np.random.Generator | None = None
    ) -> None:
        self.n_neurons = n_neurons
        self.leak_rate = leak_rate
        self.fan_in = fan_in
        self.morphology = morphology or Morphology()
        
        # Deep RNG Propagation
        gen = rng or np.random.default_rng()

        # Build initial chaotic reservoir (This will now be frozen across generations)
        self.W_rec = self._build_sparse_reservoir(n_neurons, fan_in, excitatory_frac, gen)
        self.W_rec_T = self.W_rec.T.tocsr()

        dense_in = gen.uniform(-2.0, 2.0, size=(N_SENSORS, n_neurons))
        mask = gen.random((N_SENSORS, n_neurons)) < 0.1
        self.W_in = sparse.csr_matrix(dense_in * mask, dtype=np.float32)

        scale = 1.0 / np.sqrt(n_neurons)
        self.W_out_base = gen.normal(0, scale, (n_neurons, MAX_MOTORS)).astype(np.float32)
        self.W_out = self.W_out_base.copy()
        self._w_out_clip = 20.0 * scale

        self.voltage = np.zeros(n_neurons, dtype=np.float32)
        self.firing_rates = np.zeros(n_neurons, dtype=np.float32)
        self.plasticity_rate = 0.02 / np.sqrt(n_neurons)

    @staticmethod
    def _build_sparse_reservoir(n_neurons: int, fan_in: int, excitatory_frac: float, rng: np.random.Generator) -> sparse.csr_matrix:
        is_excitatory = rng.random(n_neurons) < excitatory_frac
        rows = np.repeat(np.arange(n_neurons, dtype=np.int64), fan_in)
        cols = rng.integers(0, n_neurons, size=n_neurons * fan_in)
        magnitude = np.abs(rng.normal(0, 0.04, size=n_neurons * fan_in)).astype(np.float32)
        sign = np.where(is_excitatory[rows], 1.0, -2.0).astype(np.float32)
        return sparse.csr_matrix((magnitude * sign, (rows, cols)), shape=(n_neurons, n_neurons))

    def forward(self, sensors: np.ndarray, dopamine: float = 0.0, octopamine: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        gen = rng or np.random.default_rng()
        
        input_current = self.W_in.T.dot(sensors).astype(np.float32)
        if octopamine > 0:
            input_current += gen.normal(0, octopamine * 0.2, self.n_neurons).astype(np.float32)

        recurrent_current = self.W_rec_T.dot(self.firing_rates).astype(np.float32)
        self.voltage = (1.0 - self.leak_rate) * self.voltage + self.leak_rate * (recurrent_current + input_current)
        self.firing_rates = np.tanh(self.voltage)

        # Action Noise Injection: 10% random twitching gives Hebbian learning behaviors to actually discover
        action_noise = gen.normal(0, 0.1, MAX_MOTORS).astype(np.float32)
        motors = np.tanh(np.dot(self.firing_rates, self.W_out) + action_noise)

        net_reward = dopamine - octopamine
        if net_reward != 0.0:
            self.W_out += (self.plasticity_rate * net_reward) * (self.firing_rates[:, None] * motors)
            np.clip(self.W_out, -self._w_out_clip, self._w_out_clip, out=self.W_out)

        # Slice down to actual active joints
        return (motors * 8.0)[:self.morphology.n_motors]

    def reset_state(self) -> None:
        self.voltage.fill(0.0)
        self.firing_rates.fill(0.0)
        self.W_out = self.W_out_base.copy()

    def mutate(self, rate: float, scale: float, rng: np.random.Generator) -> "FastNeuropil":
        child = FastNeuropil.__new__(FastNeuropil)
        child.n_neurons = self.n_neurons
        child.leak_rate = self.leak_rate
        child.fan_in = self.fan_in
        
        child.morphology = self.morphology.mutate(rate, scale, rng)

        # True Liquid State Machine: W_rec and W_in are NEVER mutated. 
        # The chaotic pool remains identical; evolution only searches the Readout mapping.
        child.W_rec = self.W_rec
        child.W_rec_T = self.W_rec_T
        child.W_in = self.W_in
        
        child.W_out_base = self.W_out_base.copy()
        
        if rate > 0.0:
            # Probability ('rate') controls HOW MANY connections mutate
            out_mask = rng.random(child.W_out_base.shape) < rate
            n_masked = int(out_mask.sum())
            if n_masked:
                # Magnitude ('scale') controls HOW FAR the weights drift
                child.W_out_base[out_mask] += rng.normal(0, (0.05 * scale) / np.sqrt(self.n_neurons), n_masked).astype(np.float32)

        child.W_out = child.W_out_base.copy()  # Wiping lifetime plasticity enforces Baldwinian learning
        child.voltage = np.zeros(self.n_neurons, dtype=np.float32)
        child.firing_rates = np.zeros(self.n_neurons, dtype=np.float32)
        child.plasticity_rate = self.plasticity_rate
        child._w_out_clip = self._w_out_clip
        return child

    def clone(self, rng: np.random.Generator) -> "FastNeuropil":
        return self.mutate(rate=0.0, scale=1.0, rng=rng)