from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
gate = pd.read_csv(ROOT / "results" / "gate_results.csv")
policy = pd.read_csv(ROOT / "results" / "policy_summary.csv").iloc[0]

gate = gate[gate["held_out_decision"] != "uncertain"].copy()

fig, ax = plt.subplots(figsize=(10, 6))

series = [
    ("PBM-D", "point", "PBM-D · point"),
    ("PBM-D", "lower_bound", "PBM-D · lower bound"),
    ("INTERPOL-3", "point", "INTERPOL-3 · point"),
    ("INTERPOL-3", "lower_bound", "INTERPOL-3 · lower bound"),
]

for estimator, rule, label in series:
    g = gate[
        (gate["estimator"] == estimator)
        & (gate["gate_rule"] == rule)
    ].sort_values("required_uplift_pct")

    ax.plot(
        g["required_uplift_pct"],
        g["wrong_decision_rate_pct"],
        marker="o",
        linewidth=2,
        label=label,
    )

ax.axvspan(
    policy["held_out_uplift_envelope_lower"] * 100,
    policy["held_out_uplift_envelope_upper"] * 100,
    alpha=0.12,
    label="Held-out decision uncertain",
)
ax.axvline(
    policy["held_out_target_uplift"] * 100,
    linestyle=":",
    linewidth=1.5,
    label="Held-out uplift point estimate",
)

ax.set_title("Gate error rate by minimum uplift")
ax.set_xlabel("Minimum uplift before A/B, %")
ax.set_ylabel("Wrong decision across bootstraps, %")
ax.set_ylim(-3, 103)
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
out = ROOT / "figures" / "gate_decision_errors.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
