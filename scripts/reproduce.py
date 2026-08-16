"""Reproduce the Amazon Music OPE gate experiment."""
from pathlib import Path
import os
import sys
import time
import re

import numpy as np
import pandas as pd
import requests
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
AMAZON_REPO = ROOT / "music-off-policy-evaluation-benchmark"
CACHE_DIR = ROOT / ".cache" / "amazon_music_ope"
RESULTS_DIR = ROOT / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "amazon/music-off-policy-evaluation-benchmark"
SEED = 42
N_D1 = 250_000
N_D2 = 200_000
N_BOOTSTRAPS = 30
BOOTSTRAP_N = 100_000
OMEGA = 1.018306600302338

KEEP_COLUMNS = [
    "rewards",
    "logging_selected_actions",
    "target_selected_actions",
    "propensities",
]

PB = np.array([
    1.0,
    0.76674848,
    0.33742719,
    0.26136553,
    0.20732642,
    0.16384161,
    0.13263467,
    0.11458964,
    0.11255035,
    0.09426026,
    0.07283572,
    0.07004059,
    0.06315475,
    0.05890552,
    0.05496263,
])

UPLIFT_BARS = np.array([
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
    0.060,
    0.080,
])


def hf_parquet_files(dataset_id, split):
    response = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset_id},
        timeout=60,
    )
    response.raise_for_status()

    files = [
        x
        for x in response.json()["parquet_files"]
        if x["split"] == split
    ]
    if not files:
        raise RuntimeError(f"No parquet files found for {split=}.")
    return files


def parquet_row_group_catalog(files):
    rows = []

    for file_idx, item in enumerate(files):
        with fsspec.open(
            item["url"],
            "rb",
            block_size=8 * 1024 * 1024,
            cache_type="readahead",
        ) as f:
            pf = pq.ParquetFile(f)

            for row_group in range(pf.num_row_groups):
                meta = pf.metadata.row_group(row_group)
                rows.append({
                    "file_idx": file_idx,
                    "row_group": row_group,
                    "n_rows": meta.num_rows,
                })

    catalog = pd.DataFrame(rows)
    catalog["start"] = catalog["n_rows"].cumsum().shift(fill_value=0)
    catalog["stop"] = catalog["start"] + catalog["n_rows"]
    return catalog


def random_rows_from_remote_split(dataset_id, split, n_rows, seed, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        print("Using cached sample:", out_path)
        return out_path

    files = hf_parquet_files(dataset_id, split)
    catalog = parquet_row_group_catalog(files)
    total_rows = int(catalog["n_rows"].sum())

    rng = np.random.default_rng(seed)
    global_indices = np.sort(
        rng.choice(total_rows, size=n_rows, replace=False)
    )

    row_group_ids = np.searchsorted(
        catalog["stop"].to_numpy(),
        global_indices,
        side="right",
    )

    selections = {}
    for global_idx, row_group_id in zip(global_indices, row_group_ids):
        row = catalog.iloc[int(row_group_id)]
        key = (int(row["file_idx"]), int(row["row_group"]))
        local_idx = int(global_idx - int(row["start"]))
        selections.setdefault(key, []).append(local_idx)

    writer = None
    written = 0
    t0 = time.time()

    try:
        for file_idx, item in enumerate(files):
            wanted = {
                rg: idxs
                for (fidx, rg), idxs in selections.items()
                if fidx == file_idx
            }
            if not wanted:
                continue

            with fsspec.open(
                item["url"],
                "rb",
                block_size=8 * 1024 * 1024,
                cache_type="readahead",
            ) as f:
                pf = pq.ParquetFile(f)

                for row_group, local_indices in sorted(wanted.items()):
                    table = pf.read_row_group(
                        row_group,
                        columns=KEEP_COLUMNS,
                    )
                    selected = table.take(
                        pa.array(local_indices, type=pa.int64())
                    )

                    if writer is None:
                        writer = pq.ParquetWriter(
                            out_path,
                            selected.schema,
                            compression="zstd",
                        )
                    writer.write_table(selected)
                    written += selected.num_rows
                    print(f"{split}: {written:,}/{n_rows:,}", end="\r")
    finally:
        if writer is not None:
            writer.close()

    if written != n_rows:
        raise RuntimeError(
            f"Expected {n_rows:,} rows, wrote {written:,}."
        )

    print(
        f"\nSaved random {split} sample from {total_rows:,} rows "
        f"in {time.time() - t0:.1f}s"
    )
    return out_path


if not AMAZON_REPO.exists():
    raise FileNotFoundError(
        "Clone Amazon's benchmark repository into:\n"
        f"  {AMAZON_REPO}\n\n"
        "git clone "
        "https://github.com/amazon-science/"
        "music-off-policy-evaluation-benchmark.git "
        "music-off-policy-evaluation-benchmark"
    )

sys.path.insert(0, str(AMAZON_REPO))

from data.batch import DataBatch
from estimators.ground_truth import GroundTruth
from estimators.pbm import PBM, PBMType
from estimators.interpol import INTERPOL


# Reduced-schema path: fail if these estimators start using action features.
files_to_audit = [
    AMAZON_REPO / "estimators" / "ground_truth.py",
    AMAZON_REPO / "estimators" / "pbm.py",
    AMAZON_REPO / "estimators" / "interpol.py",
]

for path in files_to_audit:
    text = path.read_text(encoding="utf-8")
    if ".actions" in text:
        raise RuntimeError(
            f"{path.name} references DataBatch.actions; "
            "the reduced-schema path is no longer safe."
        )


def from_record_low_disk(batch):
    data = batch.to_pydict()
    placeholder_actions = []

    for logging_ids, target_ids in zip(
        data["logging_selected_actions"],
        data["target_selected_actions"],
    ):
        ids = list(logging_ids) + list(target_ids)
        min_len = (max(ids) + 1) if ids else 0
        placeholder_actions.append(range(min_len))

    return DataBatch(
        actions=placeholder_actions,
        rewards=data["rewards"],
        logging_actions=data["logging_selected_actions"],
        target_actions=data["target_selected_actions"],
        propensities=data["propensities"],
        n_rows=batch.num_rows,
    )


DataBatch.from_record = staticmethod(from_record_low_disk)


def make_estimators():
    return {
        "PBM-D": PBM(
            position_bias=PB,
            type=PBMType.DETERMINISTIC,
        ),
        "INTERPOL-3": INTERPOL(
            window_size=3,
            position_bias=PB,
        ),
    }


def ground_truth(path, seed):
    np.random.seed(seed)
    return GroundTruth(omega=OMEGA).evaluate(str(path))


def evaluate_estimator(name, estimator, d2_path, gt_obj, seed):
    np.random.seed(seed)
    est_name, estimate, error = estimator.benchmark(
        name,
        str(d2_path),
        gt_obj,
        None,
    )

    return {
        "estimator": est_name,
        "ground_truth": float(gt_obj.metric),
        "estimate": float(estimate.metric),
        "reported_lower": float(estimate.ci.lower_bound),
        "reported_upper": float(estimate.ci.upper_bound),
        "reported_interval_contains_gt_point": (
            float(estimate.ci.lower_bound)
            <= float(gt_obj.metric)
            <= float(estimate.ci.upper_bound)
        ),
        "bias2": float(error.bias2),
        "variance": float(error.var),
        "mse": float(error.mse),
    }


D1_LOCAL = random_rows_from_remote_split(
    DATASET_ID,
    "D1",
    N_D1,
    SEED + 1,
    CACHE_DIR / f"D1_random_{N_D1}_seed{SEED+1}.parquet",
)

D2_LOCAL = random_rows_from_remote_split(
    DATASET_ID,
    "D2",
    N_D2,
    SEED + 2,
    CACHE_DIR / f"D2_random_{N_D2}_seed{SEED+2}.parquet",
)

target_gt = ground_truth(D1_LOCAL, SEED + 10_000)
logging_gt = ground_truth(D2_LOCAL, SEED + 20_000)

target_value = float(target_gt.metric)
target_lower = float(target_gt.ci.lower_bound)
target_upper = float(target_gt.ci.upper_bound)

logging_value = float(logging_gt.metric)
logging_lower = float(logging_gt.ci.lower_bound)
logging_upper = float(logging_gt.ci.upper_bound)

held_out_uplift = target_value / logging_value - 1.0

# Conservative envelope from the reported marginal intervals.
held_out_uplift_lower = target_lower / logging_upper - 1.0
held_out_uplift_upper = target_upper / logging_lower - 1.0

policy_summary = pd.DataFrame([{
    "logging_policy_value_D2": logging_value,
    "logging_reported_lower": logging_lower,
    "logging_reported_upper": logging_upper,
    "target_policy_reference_D1": target_value,
    "target_reported_lower": target_lower,
    "target_reported_upper": target_upper,
    "held_out_target_uplift": held_out_uplift,
    "held_out_uplift_envelope_lower": held_out_uplift_lower,
    "held_out_uplift_envelope_upper": held_out_uplift_upper,
}])
policy_summary.to_csv(RESULTS_DIR / "policy_summary.csv", index=False)

d2_table = pq.read_table(D2_LOCAL, columns=KEEP_COLUMNS)
bootstrap_dir = CACHE_DIR / "bootstrap_tmp"
bootstrap_dir.mkdir(exist_ok=True)


def run_bootstrap(replicate):
    rng = np.random.default_rng(SEED + 100_000 + replicate)
    indices = rng.integers(
        low=0,
        high=d2_table.num_rows,
        size=BOOTSTRAP_N,
        dtype=np.int64,
    )

    sample = d2_table.take(pa.array(indices))
    path = bootstrap_dir / f"boot_{replicate:03d}.parquet"
    pq.write_table(sample, path, compression="zstd")

    try:
        sample_logging_gt = ground_truth(
            path,
            SEED + 200_000 + replicate,
        )
        sample_logging_value = float(sample_logging_gt.metric)

        rows = []
        estimators = make_estimators()

        for estimator_index, (name, estimator) in enumerate(
            estimators.items()
        ):
            row = evaluate_estimator(
                name,
                estimator,
                path,
                target_gt,
                SEED + 300_000 + 100 * replicate + estimator_index,
            )
            row["replicate"] = replicate
            row["logging_policy_value"] = sample_logging_value
            row["estimated_uplift"] = (
                row["estimate"] / sample_logging_value - 1.0
            )
            row["reported_lower_uplift"] = (
                row["reported_lower"] / sample_logging_value - 1.0
            )
            row["reported_upper_uplift"] = (
                row["reported_upper"] / sample_logging_value - 1.0
            )
            rows.append(row)

        return rows
    finally:
        path.unlink(missing_ok=True)


bootstrap_rows = []
for replicate in range(N_BOOTSTRAPS):
    bootstrap_rows.extend(run_bootstrap(replicate))
    print(f"bootstrap {replicate + 1}/{N_BOOTSTRAPS}", end="\r")
print()

bootstrap = pd.DataFrame(bootstrap_rows)

bootstrap_summary = (
    bootstrap
    .groupby("estimator", as_index=False)
    .agg(
        mean_estimate=("estimate", "mean"),
        bootstrap_sd=("estimate", "std"),
        median_estimated_uplift=("estimated_uplift", "median"),
        p10_estimated_uplift=(
            "estimated_uplift",
            lambda x: np.quantile(x, 0.10),
        ),
        p90_estimated_uplift=(
            "estimated_uplift",
            lambda x: np.quantile(x, 0.90),
        ),
        reported_interval_contains_gt_rate=(
            "reported_interval_contains_gt_point",
            "mean",
        ),
    )
)
bootstrap_summary.to_csv(
    RESULTS_DIR / "bootstrap_summary.csv",
    index=False,
)


def held_out_decision(bar):
    if bar < held_out_uplift_lower:
        return "definite_pass"
    if bar > held_out_uplift_upper:
        return "definite_fail"
    return "uncertain"


gate_rows = []

for estimator, group in bootstrap.groupby("estimator"):
    for bar in UPLIFT_BARS:
        truth = held_out_decision(bar)

        rules = {
            "point": group["estimated_uplift"] > bar,
            "lower_bound": group["reported_lower_uplift"] > bar,
        }

        for rule, pass_series in rules.items():
            predicted_pass = pass_series.to_numpy(dtype=bool)

            if truth == "definite_pass":
                false_positive_rate = 0.0
                false_negative_rate = float((~predicted_pass).mean())
                wrong_rate = false_negative_rate
            elif truth == "definite_fail":
                false_positive_rate = float(predicted_pass.mean())
                false_negative_rate = 0.0
                wrong_rate = false_positive_rate
            else:
                false_positive_rate = np.nan
                false_negative_rate = np.nan
                wrong_rate = np.nan

            gate_rows.append({
                "estimator": estimator,
                "gate_rule": rule,
                "required_uplift_pct": bar * 100,
                "held_out_decision": truth,
                "pass_rate_pct": float(predicted_pass.mean() * 100),
                "false_positive_rate_pct": (
                    false_positive_rate * 100
                    if not np.isnan(false_positive_rate)
                    else np.nan
                ),
                "false_negative_rate_pct": (
                    false_negative_rate * 100
                    if not np.isnan(false_negative_rate)
                    else np.nan
                ),
                "wrong_decision_rate_pct": (
                    wrong_rate * 100
                    if not np.isnan(wrong_rate)
                    else np.nan
                ),
            })

gate = pd.DataFrame(gate_rows)
gate.to_csv(RESULTS_DIR / "gate_results.csv", index=False)

print("\nPolicy summary")
print(policy_summary.to_string(index=False))
print("\nBootstrap summary")
print(bootstrap_summary.to_string(index=False))
print("\nGate results")
print(gate.to_string(index=False))
