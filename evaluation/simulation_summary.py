"""Extract the reported cohort-level operating characteristics from the synthetic study outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.runtime import RUN_ROOT

AGG = RUN_ROOT / "results"
OUT = AGG / "R15G_simulation_extract.json"
S1F, S2F = "R15C0_S1.json", "R15C0_S2.json"

DESIGN = "dreyer_within_day_40"
DESIGNS_ALL = ["dreyer_within_day_40", "yang_within_day_1day_40", "yang_within_day_3days_120", "yang_cross_day_200_sw025", "yang_cross_day_200_sw040"]
GENERIC = {"dreyer_within_day_40": "one run, 40 + 40 trials",                       # publication labels (no cohort names)
           "yang_within_day_1day_40": "one run, 40 + 40 trials (second generator draw)",
           "yang_within_day_3days_120": "three same-day replicates pooled, 120 + 120 trials",
           "yang_cross_day_200_sw025": "selection and evaluation on different days, 200 + 200 trials, session SD 0.25",
           "yang_cross_day_200_sw040": "selection and evaluation on different days, 200 + 200 trials, session SD 0.40"}
DELTAS = [0, 1, 2, 3, 4, 6]
FLIPS = [-2.0, -0.5, 0.5, 2.0, 5.0]


def sha(f):
    return hashlib.sha256((AGG / f).read_bytes()).hexdigest()


def load(f):
    return json.loads((AGG / f).read_text())


def find(cells, **kw):
    hits = [(i, c) for i, c in enumerate(cells) if all(c.get(k) == v for k, v in kw.items())]
    assert len(hits) == 1, (kw, len(hits))
    return hits[0]


def fkey(F):
    return ("p" if F > 0 else "m") + f"{abs(F):g}".replace(".", "_")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    s1, s2 = load(S1F), load(S2F)
    out = {"schema": "R15G_simulation_extract_v1", "synthetic_only_for": ["S1", "S2"],
           "sources": {f: {"sha256": sha(f)} for f in (S1F, S2F)},
           "note": "values copied unchanged from the sources; S1 summaries are min / max over the listed cells", "S1": {}, "S2": {}}

    # ---------------------------------------------------------------- S1: tie-break decisiveness, min / max over the 12 cells of each (r, dose)
    cells = s1["cells_by_group"]["all"]
    for r in (0.95, 0.99, 0.999, 1.0):
        rk = f"r{r:g}".replace(".", "_")
        out["S1"][rk] = {}
        for d in ("d0.25", "d0.5", "d0.75", "d1.0"):
            idx = [i for i, c in enumerate(cells) if c["r_int"] == r]
            assert len(idx) == 12
            vals = [cells[i][d]["p_tiebreak_decisive"] for i in idx]
            out["S1"][rk][d.replace(".", "_")] = {"min": min(vals), "max": max(vals), "n_cells": len(idx),
                                                   "provenance": {"file": S1F, "group": "all", "indices": idx, "field": [d, "p_tiebreak_decisive"]}}
    out["S1"]["n_rep"] = s1["n_rep"]
    out["S1"]["fixed"] = s1["fixed"]

    # ---------------------------------------------------------------- S2: operating characteristics of the rule as used (roster tie-break)
    S2 = out["S2"]
    S2["n_rep"], S2["n_boot"], S2["r_leader"] = s2["n_rep"], s2["n_boot"], s2["r_leader"]
    S2["designs"] = {k: dict(v, label=GENERIC[k]) for k, v in s2["designs"].items()}
    g = s2["cells_by_group"]
    grid = {}
    for n in (83, 51):
        for F in FLIPS:
            for D in DELTAS:
                grp = f"main|{DESIGN}"
                i, c = find(g[grp], n=n, Delta=D, F=F)
                grid.setdefault(f"n{n}", {}).setdefault(fkey(F), {})[f"D{D}"] = {
                    "power_d05_roster": c["d0.5_roster"]["A"], "p_flip_d05": c["d0.5_roster"]["flip"], "theta_d05": c["d0.5_roster"]["theta"],
                    "provenance": {"file": S2F, "group": grp, "index": i, "fields": ["d0.5_roster|A", "d0.5_roster|flip", "d0.5_roster|theta"]}}
    S2["grid_dreyer_like"] = grid
    # the Delta = 0 cells of the grid are null cells too (no true regret): their false-regret rates, dose 0.5, roster rule
    d0 = [(n, F, grid[n][F]["D0"]["power_d05_roster"]) for n in grid for F in grid[n]]
    S2["grid_delta0_false_regret_d05_max"] = {"max": max(v for _, _, v in d0), "cells": [{"n": n, "F": F, "rate": v} for n, F, v in d0],
                                              "definition": "max over the 10 Delta = 0 cells of the cohort-like grid (both n, all F)"}
    allD0 = {}
    for dz in DESIGNS_ALL:
        for key in ("d0.5_roster", "d0.5_tieaware"):
            allD0.setdefault(dz, {})[key.replace(".", "_")] = max(c[key]["A"] for c in g[f"main|{dz}"] if c["Delta"] == 0)
    S2["all_designs_delta0_false_regret_max"] = allD0
    # dose 1.0 where the dose-0.5 contaminated winner is unchanged (F = +2 and +5), and the population scores at both doses
    d1 = {}
    for n in (83, 51):
        for F in (2.0, 5.0):
            for D in (2, 3):
                grp = f"main|{DESIGN}"
                i, c = find(g[grp], n=n, Delta=D, F=F)
                d1.setdefault(f"n{n}", {}).setdefault(fkey(F), {})[f"D{D}"] = {
                    "power_d10_roster": c["d1.0_roster"]["A"], "pop_contam_d05": c["pop_contam_B_d05"], "pop_contam_d10": c["pop_contam_B_d10"],
                    "provenance": {"file": S2F, "group": grp, "index": i, "fields": ["d1.0_roster|A", "pop_contam_B_d05", "pop_contam_B_d10"]}}
    S2["dose1_same_winner"] = d1
    # why the other dose-1.0 cells are not reported as conditional power: the third decoder overtakes at dose 1.0
    over = []
    for i, c in enumerate(g[f"main|{DESIGN}"]):
        p05, p10 = c["pop_contam_B_d05"], c["pop_contam_B_d10"]
        w05, w10 = max(range(3), key=lambda k: p05[k]), max(range(3), key=lambda k: p10[k])
        if w05 != w10:
            over.append({"index": i, "n": c["n"], "Delta": c["Delta"], "F": c["F"], "winner_d05": s2["roster"][w05], "winner_d10": s2["roster"][w10]})
    S2["dose1_winner_changes"] = {"n_cells": len(over), "of": len(g[f"main|{DESIGN}"]), "flip_margins_affected": sorted({o["F"] for o in over}),
                                  "cells": over}
    # null regimes of the cohort-like design (both doses, roster rule)
    nulls = {}
    for fam in ("null_no_effect", "noise_flip_no_effect_gap2", "null_same_best_on_C", "null_heterogeneous_gap"):
        grp = f"{fam}|{DESIGN}"
        for n in (83, 51):
            i, c = find(g[grp], n=n)
            nulls.setdefault(fam, {})[f"n{n}"] = {
                "false_regret_d05": c["d0.5_roster"]["A"], "false_regret_d10": c["d1.0_roster"]["A"], "reverse_d05": c["d0.5_roster"]["rev"],
                "reverse_d10": c["d1.0_roster"]["rev"], "p_flip_d05": c["d0.5_roster"]["flip"], "theta_d05": c["d0.5_roster"]["theta"],
                "theta_d10": c["d1.0_roster"]["theta"],
                "provenance": {"file": S2F, "group": grp, "index": i}}
    S2["nulls_dreyer_like"] = nulls
    S2["nulls_max"] = {
        "false_regret_d05": max(v[k]["false_regret_d05"] for f, v in nulls.items() if f != "null_same_best_on_C" for k in v),
        "false_regret_d10": max(v[k]["false_regret_d10"] for f, v in nulls.items() if f != "null_same_best_on_C" for k in v),
        "regret_in_beneficial_d05": max(nulls["null_same_best_on_C"][k]["false_regret_d05"] for k in ("n83", "n51")),
        "regret_in_beneficial_d10": max(nulls["null_same_best_on_C"][k]["false_regret_d10"] for k in ("n83", "n51")),
        "definition": "max over the no-effect, noise-flip and heterogeneous-gap regimes of P(regret declared); the beneficial regime "
                      "(same best model on the evaluation run) is reported separately"}
    # sensitivity: between-decoder correlation and heterogeneity (cohort-like design), and other trial-count / session designs
    sens = {}
    for fam in ("sens_sv", "sens_het"):
        grp = f"{fam}|{DESIGN}"
        for i, c in enumerate(g[grp]):
            sens.setdefault(fam, {}).setdefault(f"n{c['n']}", {})[f"sv{c['sv']:g}_D{c['Delta']}".replace(".", "_")] = {
                "power_d05_roster": c["d0.5_roster"]["A"], "sv": c["sv"], "sr": c["sr"], "Delta": c["Delta"],
                "provenance": {"file": S2F, "group": grp, "index": i}}
    for n in (83, 51):
        for D in (2, 3):
            i, c = find(g[f"main|{DESIGN}"], n=n, Delta=D, F=2.0)
            sens.setdefault("central", {}).setdefault(f"n{n}", {})[f"D{D}"] = {"power_d05_roster": c["d0.5_roster"]["A"],
                                                                                 "provenance": {"file": S2F, "group": f"main|{DESIGN}", "index": i}}
    S2["sensitivity_dreyer_like"] = sens
    des = {}
    for dz in DESIGNS_ALL:
        for n in (83, 51):
            for D in (2, 3):
                i, c = find(g[f"main|{dz}"], n=n, Delta=D, F=2.0)
                des.setdefault(dz, {}).setdefault(f"n{n}", {})[f"D{D}"] = {"power_d05_roster": c["d0.5_roster"]["A"],
                                                                             "provenance": {"file": S2F, "group": f"main|{dz}", "index": i}}
    S2["designs_power"] = des

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.name}; dose-1.0 winner changes in {len(over)} of {len(g[f'main|{DESIGN}'])} cohort-like cells "
          f"(flip margins {sorted({o['F'] for o in over})})")


if __name__ == "__main__":
    main()
