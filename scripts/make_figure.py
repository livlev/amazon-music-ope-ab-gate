"""
Rebuilds the gate error-rate figure from results/uncertainty_aware_uplift_gate.csv.

This does not recompute anything — it plots exactly what analysis.ipynb
already produced and saved. Run analysis.ipynb first if the CSV is missing
or you want to regenerate it against a fresh sample.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

ENVELOPE_LOWER_PCT = 0.112037
ENVELOPE_UPPER_PCT = 1.353628
POINT_UPLIFT_PCT = 0.740041


def main() -> None:
    gate = pd.read_csv(RESULTS_DIR / "uncertainty_aware_uplift_gate.csv")

    fig, ax = plt.subplots(figsize=(9.5, 5.3))

    line_specs = [
        ("PBM-D", "point", "PBM-D · point"),
        ("PBM-D", "lower_bound", "PBM-D · lower bound"),
        ("INTERPOL-3", "point", "INTERPOL-3 · point"),
        ("INTERPOL-3", "lower_bound", "INTERPOL-3 · lower bound"),
    ]

    for estimator, gate_rule, label in line_specs:
        g = gate[
            (gate["estimator"] == estimator) & (gate["gate_rule"] == gate_rule)
        ].sort_values("required_uplift_pct")

        ax.plot(
            g["required_uplift_pct"],
            g["wrong_decision_rate_pct"],
            marker="o",
            label=label,
        )

    ax.axvspan(
        ENVELOPE_LOWER_PCT,
        ENVELOPE_UPPER_PCT,
        alpha=0.12,
        label="Held-out decision uncertain",
    )
    ax.axvline(POINT_UPLIFT_PCT, linestyle=":", label="Held-out uplift point estimate")

    ax.set_xlabel("Minimum uplift required before A/B, %")
    ax.set_ylabel("Wrong gate decision across bootstraps, %")
    ax.set_title("OPE gate errors, scored only where the held-out decision is definite")
    ax.set_ylim(-2, 102)
    ax.legend()
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "gate_wrong_decision_rate.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
