"""Audit the supplied D data without changing the official files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_surrogates import DATA, METAL_BOUND


def audit():
    manifest = json.loads((DATA / "sha256_public.json").read_text(encoding="utf-8"))
    hashes = {}
    for name, expected in manifest.items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        hashes[name] = actual == expected
        if actual != expected:
            raise ValueError(f"Source checksum mismatch: {name}")

    frames = {prefix: pd.read_csv(next(DATA.glob(prefix + "_*.csv")))
              for prefix in ["02", "03", "04", "06", "07", "08", "11", "12"]}
    expected_rows = {"02": 11256, "03": 3969, "04": 3703, "06": 15225,
                     "07": 2704, "08": 8112, "11": 529, "12": 1587}
    for prefix, frame in frames.items():
        assert len(frame) == expected_rows[prefix]
        assert frame.id.is_unique and not frame.duplicated(["A", "B", "X"]).any()

    groups = {key: set(zip(frames[key].A, frames[key].B)) for key in ["02", "03", "04"]}
    assert all(not groups[a] & groups[b] for a, b in [("02", "03"), ("02", "04"), ("03", "04")])
    assert len(set.union(*groups.values())) == 52 * 52
    assert set(frames["06"].id) == set(frames["02"].id) | set(frames["03"].id)
    assert set(frames["11"].id) == set(frames["07"].id) & set(frames["04"].id)
    assert set(frames["12"].id) == set(frames["08"].id) & set(frames["04"].id)

    report = {"checksums": hashes, "rows": expected_rows,
              "pair_counts": {key: len(value) for key, value in groups.items()},
              "zero_gap_penalty_lower_bound": METAL_BOUND, "labeled": {}}
    for key in ["02", "03"]:
        frame = frames[key]
        deviations = np.column_stack([frame.e_form - 0.21, 1.4 - frame.gap,
                                      frame.gap - 3.1, 5.73 - frame.vbm, frame.cbm - 4.50])
        recalculated = np.maximum(0, deviations / 0.5) ** 2
        error = float(np.max(np.abs(recalculated.sum(axis=1) - frame.V)))
        assert error < 2e-5
        zero_gap = frame.gap == 0
        assert (frame.loc[zero_gap, "vbm"] == frame.loc[zero_gap, "cbm"]).all()
        assert frame.loc[zero_gap, "V"].min() >= METAL_BOUND - 2e-5
        report["labeled"][key] = {
            "zero_V": int((frame.V == 0).sum()), "zero_gap": int(zero_gap.sum()),
            "nonzero_gap": int((~zero_gap).sum()), "max_V_recalculation_error": error,
            "minimum_metal_V": float(frame.loc[zero_gap, "V"].min()),
            "count_V_lt_1": int((frame.V < 1).sum()),
            "count_V_lt_025": int((frame.V < 0.25).sum())}
    output = DATA.parent / "results" / "data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    audit()
