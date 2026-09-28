"""
gemini_director.py — Project Primus: optional Gemini-governed "director".

This looks at a summary of recent generations and proposes bounded
adjustments to a small set of tunable knobs. evolve.py applies whatever
comes back, clamped to safe ranges, and keeps training even if this module
is unavailable or errors out.

Setup:
    pip install requests
    export GEMINI_API_KEY="..."          # from https://aistudio.google.com/apikey
    # PowerShell: $env:GEMINI_API_KEY="..."

CHANGES vs. previous version (per review feedback):
  * Auth: current Gemini REST docs (ai.google.dev/api) specify the API key
    goes in the `x-goog-api-key` header, not the `?key=` query string.
    Fixed here -- the query-string form isn't guaranteed going forward.
  * `food_abundance` renamed to `food_energy` everywhere, and the prompt/
    docstring now honestly describe what it does: it scales how much
    energy each food item restores, NOT how many food items exist (food
    count is baked into the MuJoCo model at build time and isn't something
    this director can touch without a model rebuild).
  * Suggestions are now also clamped to a max +/-15% relative change from
    the current value per consultation, on top of the absolute BOUNDS --
    the prompt asked for "small nudges" but nothing enforced that before.
  * The prompt now includes population-average foods/danger/distance/drift
    and (when available) the fixed-environment "challenge trial" score, not
    just win/loss-style death counts, so suggestions are less of a guess.

Model name: defaults to "gemini-2.5-flash". Gemini's model lineup moves
fast -- if this 404s, check https://ai.google.dev/gemini-api/docs/models
for the current flash-tier model name and update GEMINI_MODEL below.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Absolute safe bounds -- a suggestion is clamped into these no matter what.
BOUNDS = {
    "mutation_scale": (0.5, 2.0),        # multiplier applied to the base mutation_rates list
    "predator_aggression": (0.6, 1.8),   # multiplier applied to every predator's `force`
    "food_energy": (0.5, 2.0),           # multiplier applied to FOOD_ENERGY_GAIN (not food count)
    "energy_pressure": (0.5, 2.0),       # multiplier applied to metabolic decay/move-cost
}

# On top of the absolute bounds: a single consultation can move any knob by
# at most this fraction of its current value, so "small nudges" is actually
# enforced rather than just claimed in the prompt.
MAX_RELATIVE_STEP = 0.15


@dataclass
class DirectorAdjustments:
    mutation_scale: float = 1.0
    predator_aggression: float = 1.0
    food_energy: float = 1.0
    energy_pressure: float = 1.0
    rationale: str = ""

    def clamped(self, current: "DirectorAdjustments | None" = None) -> "DirectorAdjustments":
        d = asdict(self)
        cur = asdict(current) if current is not None else None
        for key, (lo, hi) in BOUNDS.items():
            value = d[key]
            if cur is not None:
                step = abs(cur[key]) * MAX_RELATIVE_STEP
                value = min(cur[key] + step, max(cur[key] - step, value))
            d[key] = float(min(hi, max(lo, value)))
        d["rationale"] = str(d.get("rationale", ""))[:400]
        return DirectorAdjustments(**d)


_PROMPT_TMPL = """You are directing an evolutionary-robotics simulation called Project Primus.
Wheeled agents forage for food on islands while evading predators. Each generation,
the top scorers are cloned and mutated to make the next population. Selection itself
always runs in a FIXED baseline environment (your knobs never touch the trials used
to actually rank/select agents) -- what you tune only affects (a) how aggressively the
next generation is mutated, and (b) a separate "challenge" environment used purely to
stress-test the current champion, whose score is reported back to you for context.

Recent generation history (most recent last), one line per generation:
{history_lines}

Current tunable multipliers (all relative to their defaults of 1.0):
{current_params}

Field meanings:
  mutation_scale       - scales how much genetic variation is injected per generation
  predator_aggression  - scales every predator's thrust force
  food_energy          - scales how much energy eating one food item restores (NOT food count)
  energy_pressure      - scales metabolic upkeep + movement energy cost

Your job: propose SMALL nudges (each will additionally be capped at +/-15% of its
current value regardless of what you say) to help training escape stagnation or
overcorrect for imbalance. E.g.: agents never dying and score plateaued = raise
predator_aggression or energy_pressure a little, or raise mutation_scale if diversity
looks collapsed; agents dying too fast to learn anything = lower predator_aggression/
energy_pressure a little; population converged and steadily improving = leave things
near 1.0 or lower mutation_scale slightly to let it exploit.

Respond with ONLY a JSON object, no markdown fences, no commentary outside the JSON:
{{
  "mutation_scale": <float 0.5-2.0>,
  "predator_aggression": <float 0.6-1.8>,
  "food_energy": <float 0.5-2.0>,
  "energy_pressure": <float 0.5-2.0>,
  "rationale": "<one short sentence>"
}}
"""


def _build_prompt(history: list[dict], current_params: dict) -> str:
    lines = []
    for h in history[-12:]:
        challenge = h.get("challenge_score")
        challenge_str = f" challenge={challenge:.1f}" if challenge is not None else ""
        lines.append(
            f"gen={h.get('gen')} best={h.get('gen_best_score'):.1f} hof={h.get('hof_score'):.1f}{challenge_str} "
            f"survivors={h.get('survivors')} deaths(water={h.get('water_deaths')},"
            f"pred={h.get('pred_deaths')},starve={h.get('starve_deaths')}) "
            f"avg_foods={h.get('foods_mean', 0):.1f} avg_danger_ticks={h.get('danger_mean', 0):.0f} "
            f"avg_distance={h.get('distance_mean', 0):.1f} avg_drift={h.get('drift_mean', 0):.3f}"
        )
    return _PROMPT_TMPL.format(
        history_lines="\n".join(lines) or "(no history yet)",
        current_params=json.dumps(current_params),
    )


def consult_director(
    history: list[dict],
    current_params: dict,
    api_key: str | None = None,
    model: str = GEMINI_MODEL,
    timeout: float = 20.0,
) -> DirectorAdjustments:
    """Ask Gemini for adjustments. Returns the CURRENT settings unchanged
    (a no-op) on any missing key, missing `requests`, network error, or bad
    response -- this is designed to never crash or stall a training run."""
    current = DirectorAdjustments(**{**DirectorAdjustments().__dict__, **current_params})

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  [director] GEMINI_API_KEY not set -- skipping, training continues unmanaged.")
        return current
    if requests is None:
        print("  [director] `requests` not installed (pip install requests) -- skipping.")
        return current

    prompt = _build_prompt(history, current_params)
    url = GEMINI_URL_TMPL.format(model=model)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        proposed = DirectorAdjustments(
            mutation_scale=float(parsed.get("mutation_scale", current.mutation_scale)),
            predator_aggression=float(parsed.get("predator_aggression", current.predator_aggression)),
            food_energy=float(parsed.get("food_energy", current.food_energy)),
            energy_pressure=float(parsed.get("energy_pressure", current.energy_pressure)),
            rationale=str(parsed.get("rationale", "")),
        )
        adj = proposed.clamped(current=current)
        print(f"  [director] {model} -> {asdict(adj)}")
        return adj
    except Exception as exc:  # noqa: BLE001 - deliberately broad, never crash training
        print(f"  [director] call failed ({exc!r}) -- keeping current settings.")
        return current