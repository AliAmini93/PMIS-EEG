"""R6-C3 adapter: Lee2019/OpenBMI MI metadata -> PMIS tasks in the G3 task format (injection design).

Lee2019 has no natural duplicates (verified by the build audit), so exact copies are *injected*: arm A = C + a'
with a' a bit-identical copy of the held-out trial a. Within each participant, folds are runs (leave one run
out: 4 folds of 100 trials; base C = the other 300 trials, one row each). In the held-out run, within each class
(50 trials) an outcome-blind random 40 are matched into 20 pairs (a, b) and the other 10 stay untouched
bystanders (off-target control). Every pair gets a placebo (C2 analogue): two distinct base trials c != e of the
same session (the other run of that session) and the same class, and a random sign; arms C + c' / C + e' keep
support |C|. There is no different-run neighbour arm in this split (every other run is in the base).
All draws use numpy default_rng([stream, participant, run, class]) on metadata only.
"""

from __future__ import annotations

import numpy as np

PAIR_STREAM, PLACEBO_STREAM = 7401, 7402
PAIRS_PER_CLASS, LABELS = 20, (1, 2)
DATASET_SEED = 2019  # stands in for the G3 fold seed in task["seed"] (keeps C2-style streams distinct)


def build_tasks(meta: dict) -> list[dict]:
    subj, sess, run, pos, lab = (np.asarray(meta[k], dtype=np.int64) for k in ("subject", "session", "run", "pos", "label"))
    tasks, k = [], 0
    for s in np.unique(subj):
        for r in range(4):
            T = np.flatnonzero((subj == s) & (run == r))
            base = np.flatnonzero((subj == s) & (run != r))
            assert T.size == 100 and base.size == 300
            T = T[np.argsort(pos[T], kind="stable")]
            pairs, paired = [], set()
            for c in LABELS:
                cand = T[lab[T] == c]
                rng = np.random.default_rng([PAIR_STREAM, int(s), r, c])
                pick = rng.permutation(cand)[:2 * PAIRS_PER_CLASS]
                sess_r = int(sess[T[0]])
                pool = base[(sess[base] == sess_r) & (lab[base] == c)]
                prng = np.random.default_rng([PLACEBO_STREAM, int(s), r, c])
                for j in range(PAIRS_PER_CLASS):
                    a, b = int(pick[2 * j]), int(pick[2 * j + 1])
                    ce = prng.choice(pool, 2, replace=False)
                    sign = int(prng.choice([-1, 1]))
                    pairs.append({"k": k, "a": a, "b": b, "label": c, "session": sess_r,
                                  "pos_dist": int(abs(pos[a] - pos[b])), "g": -1, "c": int(ce[0]), "e": int(ce[1]),
                                  "sign": sign})
                    paired.update((a, b))
                    k += 1
            tasks.append({"seed": DATASET_SEED, "subject": int(s), "fold": r, "T": T.astype(np.int64),
                          "T_session": sess[T], "T_label": lab[T],
                          "unexposed": np.array([int(t) not in paired for t in T], dtype=bool),
                          "base_canon": np.sort(base).astype(np.int64),
                          "base_multi_cnt": np.ones(base.size, dtype=np.int64), "controls": True, "pairs": pairs})
    return tasks
