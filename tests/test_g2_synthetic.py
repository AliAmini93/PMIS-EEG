"""Synthetic data only. Run with the fixed owner's HCI Python interpreter."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

from experiments.pmis_g2.contracts import ContractError, TrialRow, validate_pair
from evaluation.g2_estimands import PairedScore, participant_scores, exact_two_sided_sign_flip


def row(row_id, trial_id, *, label=1, origin=1, file_run=None,
        partition="train", canonical=True, participant="p0", session="s0"):
    if file_run is None:
        file_run = origin
    return TrialRow(row_id=row_id, trial_id=trial_id,
                    waveform_sha256="sha-" + trial_id,
                    participant=participant, session=session,
                    origin_run=origin, file_run=file_run,
                    label=label, partition=partition, canonical=canonical)


def fixture():
    base = [row("c1", "c1"), row("c2", "c2", label=2)]
    a = row("a-test", "a", partition="test")
    b = row("b-test", "b", partition="test")
    aa = row("a-alias", "a", file_run=2, canonical=False)
    bb = row("b-alias", "b", file_run=2, canonical=False)
    return dict(base=base, test=[a, b], a=a, b=b, alias_a=aa, alias_b=bb)


class Contracts(unittest.TestCase):
    def test_valid(self):
        out = validate_pair(**fixture())
        self.assertEqual(out.distinct_support, 3)
        self.assertEqual(out.base_rows, 2)
        self.assertEqual(len(out.test), 2)

    def test_multiset_variant(self):
        f = fixture()
        f["base"].append(row("c1-dup", "c1", file_run=2, canonical=False))
        self.assertEqual(validate_pair(**f, canonical_base=False).base_rows, 3)

    def test_reject_class_mismatch(self):
        f = fixture()
        f["b"] = replace(f["b"], label=2)
        f["test"][1] = f["b"]
        f["alias_b"] = replace(f["alias_b"], label=2)
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_origin_run_mismatch(self):
        f = fixture()
        f["alias_b"] = replace(f["alias_b"], origin_run=3)
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_training_file_run_mismatch(self):
        f = fixture()
        f["alias_b"] = replace(f["alias_b"], file_run=3)
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_test_identity_in_base(self):
        f = fixture()
        f["base"].append(row("hidden-a", "a"))
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_nonidentical_alias(self):
        f = fixture()
        f["alias_a"] = replace(f["alias_a"], waveform_sha256="WRONG")
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_test_side_alias(self):
        f = fixture()
        f["alias_a"] = replace(f["alias_a"], partition="test")
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_reusing_candidate(self):
        f = fixture()
        f["b"] = f["a"]
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_canonical_duplicate(self):
        f = fixture()
        f["base"].append(row("extra-copy", "c1", file_run=2, canonical=False))
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_noncanonical_test(self):
        f = fixture()
        f["test"][0] = replace(f["test"][0], canonical=False)
        with self.assertRaises(ContractError): validate_pair(**f)

    def test_reject_test_duplicate_waveforms(self):
        f = fixture()
        f["test"][1] = replace(f["test"][1], waveform_sha256=f["test"][0].waveform_sha256)
        with self.assertRaises(ContractError): validate_pair(**f)


class Estimands(unittest.TestCase):
    def test_pair_score_symmetric(self):
        a = PairedScore("p1", 1, .9, .6, .6, .9)
        self.assertAlmostEqual(a.identity_swap_gain, .3)
        b = PairedScore("p1", 1, .6, .9, .9, .6)
        self.assertAlmostEqual(b.identity_swap_gain, -.3)

    def test_pair_reject_nonprobability(self):
        with self.assertRaises(ValueError):
            _ = PairedScore("p1", 1, 1.2, 0, 0, 1).identity_swap_gain

    def test_participant_balances_classes(self):
        rs = [PairedScore("p1", 1, 1, 0, 0, 1),
              PairedScore("p1", 1, 1, 0, 0, 1),
              PairedScore("p1", 2, 0, 0, 0, 0),
              PairedScore("p2", 1, 0, 0, 0, 0),
              PairedScore("p2", 2, 0, 0, 0, 0)]
        d = participant_scores(rs)
        self.assertAlmostEqual(d["p1"], .5)
        self.assertAlmostEqual(d["p2"], 0.)

    def test_one_participant_rejected(self):
        with self.assertRaises(ValueError):
            participant_scores([PairedScore("p1", 1, .5, .5, .5, .5)])

    def test_exact_sign_flip(self):
        self.assertAlmostEqual(exact_two_sided_sign_flip([1, 1, 1, 1]), .125)
        self.assertEqual(exact_two_sided_sign_flip([0, 0, 0, 0]), 1.)


if __name__ == "__main__":
    unittest.main()


# ============================================================================ G2 adversarial extension
# Fold-aware synthetic session with cumulative run saving (file run-k holds runs 1..k, as HS-Stroke v1).
# Files run-1 and run-2 are in the TEST fold 0; run-3 is fold 1; run-4 is fold 2.
from experiments.pmis_g2.contracts import check_base, check_pair  # noqa: E402
from evaluation.g2_estimands import PairedCorrect, correct_class_probability, holm, minimum_detectable_dz, participant_class_balanced, position_strata, sign_flip_floor, summarize_balanced_accuracy_contrast, summarize_participants, t_interval

FOLD_OF_FILE = {1: 0, 2: 0, 3: 1, 4: 2}
TEST_FOLD = 0


def frow(trial_id, file_run, *, label=1, origin=1, participant="p0", session="s0", fold=None, row_id=None):
    canonical = file_run == origin
    fold = FOLD_OF_FILE[file_run] if fold is None else fold
    return TrialRow(row_id=row_id or f"{trial_id}@{file_run}", trial_id=trial_id,
                    waveform_sha256="sha-" + trial_id, participant=participant, session=session,
                    origin_run=origin, file_run=file_run, label=label,
                    partition="test" if fold == TEST_FOLD else "train", canonical=canonical, fold=fold)


TRIALS = {  # trial -> (origin run, label)
    "a": (1, 1), "b": (1, 1), "t": (1, 2), "d": (2, 1), "e": (2, 2),
    "c3": (3, 1), "c4": (3, 2), "c5": (4, 1), "c6": (4, 2),
}


def copy_of(trial, file_run, **kw):
    o, lab = TRIALS[trial]
    return frow(trial, file_run, label=lab, origin=o, **kw)


def fold_fixture(multiset=False):
    test = [copy_of(x, TRIALS[x][0]) for x in ("a", "b", "t", "d", "e")]
    base = [copy_of(x, TRIALS[x][0]) for x in ("c3", "c4", "c5", "c6")]
    if multiset:
        base += [copy_of("c3", 4), copy_of("c4", 4)]
    by = {r.trial_id: r for r in test}
    return dict(base=base, test=test, a=by["a"], b=by["b"], alias_a=copy_of("a", 3), alias_b=copy_of("b", 3),
                canonical_base=not multiset, test_fold=TEST_FOLD)


def code_of(f):
    try:
        validate_pair(**f)
    except ContractError as e:
        return e.code
    return None


class FoldAwareContracts(unittest.TestCase):
    def test_valid_canonical_and_multiset(self):
        out = validate_pair(**fold_fixture())
        self.assertEqual((out.distinct_support, out.base_rows, len(out.arm_a)), (5, 4, 5))
        out = validate_pair(**fold_fixture(multiset=True))
        self.assertEqual((out.distinct_support, out.base_rows, len(out.arm_a)), (5, 6, 7))

    def test_check_pair_reuses_checked_base(self):
        f = fold_fixture()
        cb = check_base(base=f["base"], test=f["test"], canonical_base=True, test_fold=TEST_FOLD)
        one = check_pair(cb, a=f["a"], b=f["b"], alias_a=f["alias_a"], alias_b=f["alias_b"])
        two = validate_pair(**f)
        self.assertEqual(one, two)
        lite = check_pair(cb, a=f["a"], b=f["b"], alias_a=f["alias_a"], alias_b=f["alias_b"], build_arms=False)
        self.assertEqual((lite.distinct_support, lite.base_rows), (two.distinct_support, two.base_rows))

    # ---- same-label origin-run mismatches
    def test_reject_same_label_different_origin_run_same_alias_file(self):
        f = fold_fixture()
        f["b"] = next(r for r in f["test"] if r.trial_id == "d")      # label 1, origin run 2
        f["alias_b"] = copy_of("d", 3)                                  # genuine copy in the same file
        self.assertEqual(code_of(f), "source_provenance")

    def test_reject_alias_with_forged_origin_run(self):
        f = fold_fixture()
        f["alias_b"] = replace(f["alias_b"], origin_run=2)              # same label, wrong origin metadata
        self.assertEqual(code_of(f), "alias_provenance")

    # ---- shared later alias file
    def test_reject_aliases_in_different_later_files(self):
        f = fold_fixture()
        f["alias_b"] = copy_of("b", 4)
        self.assertEqual(code_of(f), "alias_file")

    def test_reject_alias_from_another_session_same_run_number(self):
        f = fold_fixture()
        f["alias_b"] = replace(f["alias_b"], session="s1")
        self.assertIn(code_of(f), ("alias_provenance", "alias_file"))

    def test_reject_shared_alias_file_in_test_fold(self):
        f = fold_fixture()
        f["alias_a"] = replace(copy_of("a", 2), partition="train")     # test-side copy relabelled as train
        f["alias_b"] = replace(copy_of("b", 2), partition="train")
        self.assertEqual(code_of(f), "alias_in_test_fold")

    def test_reject_alias_file_fold_disagreeing_with_base_rows(self):
        f = fold_fixture()
        f["alias_a"] = replace(f["alias_a"], fold=2)
        f["alias_b"] = replace(f["alias_b"], fold=2)
        self.assertEqual(code_of(f), "split_not_whole_file")

    def test_reject_canonical_row_as_alias(self):
        f = fold_fixture()
        f["alias_a"] = replace(f["a"], partition="train", fold=1)
        self.assertIn(code_of(f), ("alias_canonical", "alias_file", "alias_not_later"))

    def test_reject_alias_row_id_equal_to_test_row(self):
        f = fold_fixture()
        f["alias_a"] = replace(f["alias_a"], row_id=f["a"].row_id)
        self.assertEqual(code_of(f), "alias_row_is_test")

    # ---- missing canonical train source
    def test_reject_multiset_copy_without_canonical_source(self):
        f = fold_fixture(multiset=True)
        f["base"] = [r for r in f["base"] if r.row_id != "c3@3"]       # c3 copy kept, canonical removed
        self.assertEqual(code_of(f), "canonical_source")

    def test_reject_two_canonical_rows_of_one_identity(self):
        f = fold_fixture(multiset=True)
        f["base"].append(replace(copy_of("c3", 3), row_id="c3@3-dup"))
        self.assertEqual(code_of(f), "canonical_source")

    def test_reject_canonical_source_inside_test_fold(self):
        f = fold_fixture()
        f["base"][0] = replace(f["base"][0], fold=TEST_FOLD)
        self.assertEqual(code_of(f), "base_in_test_fold")

    # ---- test hash leak through alternate IDs
    def test_reject_test_waveform_under_alternate_id_in_base(self):
        f = fold_fixture()
        f["base"].append(replace(copy_of("c5", 4), row_id="ghost", trial_id="a-alias-id",
                                 waveform_sha256="sha-a", label=1))
        self.assertIn(code_of(f), ("hash_alternate_id", "test_waveform_in_base"))

    def test_reject_alternate_id_alias(self):
        f = fold_fixture()
        f["alias_a"] = replace(f["alias_a"], trial_id="a-alias-id")
        self.assertEqual(code_of(f), "alias_not_identical")

    def test_reject_nontest_waveform_under_two_ids(self):
        f = fold_fixture()
        f["base"].append(replace(copy_of("c5", 4), row_id="c5x", trial_id="c5x"))
        self.assertEqual(code_of(f), "hash_alternate_id")

    def test_reject_test_ids_with_shared_waveform(self):
        f = fold_fixture()
        f["test"][2] = replace(f["test"][2], waveform_sha256="sha-a")
        self.assertEqual(code_of(f), "test_waveform_repeat")

    def test_reject_row_id_shared_between_base_and_test(self):
        f = fold_fixture()
        f["base"][0] = replace(f["base"][0], row_id=f["test"][4].row_id)
        self.assertEqual(code_of(f), "row_id_shared")

    def test_reject_alias_hash_disagreeing_with_scored_trial(self):
        f = fold_fixture()
        f["alias_b"] = replace(f["alias_b"], waveform_sha256="sha-t")
        self.assertEqual(code_of(f), "alias_not_identical")

    # ---- fold / run assignment and row sanity
    def test_reject_split_file_across_folds(self):
        f = fold_fixture(multiset=True)
        f["base"][-1] = replace(f["base"][-1], fold=1)                 # c4@4 claims fold 1, file 4 is fold 2
        self.assertEqual(code_of(f), "split_not_whole_file")

    def test_reject_scored_row_outside_test_fold(self):
        f = fold_fixture()
        f["test"][4] = replace(f["test"][4], fold=1)
        self.assertEqual(code_of(f), "test_fold")

    def test_reject_fold_metadata_without_test_fold(self):
        f = fold_fixture()
        f["test_fold"] = None
        self.assertEqual(code_of(f), "fold_unchecked")

    def test_reject_missing_fold_metadata(self):
        f = fold_fixture()
        f["base"][1] = replace(f["base"][1], fold=None)
        self.assertEqual(code_of(f), "fold_missing")

    def test_reject_other_participant_in_base(self):
        f = fold_fixture()
        f["base"].append(replace(copy_of("c5", 4), participant="p1", row_id="p1-c5", trial_id="p1-c5",
                                 waveform_sha256="sha-p1-c5"))
        self.assertEqual(code_of(f), "participant_scope")

    def test_reject_canonical_flag_outside_origin_file(self):
        f = fold_fixture()
        f["base"][0] = replace(f["base"][0], file_run=4, fold=2)
        self.assertEqual(code_of(f), "canonical_not_origin_file")

    def test_reject_copy_in_earlier_file(self):
        f = fold_fixture(multiset=True)
        f["base"][-1] = replace(f["base"][-1], file_run=2, fold=TEST_FOLD, partition="train")
        self.assertEqual(code_of(f), "copy_not_later")

    def test_every_rejection_fails_closed_before_arms_exist(self):
        # mutation sweep: every single-field corruption of the alias rows is rejected
        f0 = fold_fixture()
        for field, val in (("trial_id", "zz"), ("waveform_sha256", "sha-zz"), ("participant", "p9"),
                           ("session", "s9"), ("origin_run", 3), ("file_run", 1), ("label", 2),
                           ("partition", "test"), ("canonical", True), ("fold", 0), ("row_id", "c3@3")):
            for which in ("alias_a", "alias_b"):
                f = dict(f0)
                f[which] = replace(f0[which], **{field: val})
                with self.assertRaises(ContractError, msg=f"{which}.{field}"):
                    validate_pair(**f)


class ParticipantAggregation(unittest.TestCase):
    def test_class_balance_ignores_unequal_class_counts(self):
        rs = [PairedScore("p1", 1, 1, 0, 0, 1)] * 9 + [PairedScore("p1", 2, .5, .5, .5, .5)] + \
             [PairedScore("p2", 1, .5, .5, .5, .5), PairedScore("p2", 2, .5, .5, .5, .5)]
        d = participant_scores(rs)
        self.assertAlmostEqual(d["p1"], .5)        # (1 + 0) / 2, not 9/10
        self.assertAlmostEqual(d["p2"], 0.)

    def test_participants_weighted_equally_regardless_of_pair_count(self):
        rs = [PairedScore("p1", k, .9, .1, .1, .9) for k in (1, 2) for _ in range(200)]
        rs += [PairedScore("p2", k, .5, .5, .5, .5) for k in (1, 2)]
        s = summarize_participants(rs)
        self.assertEqual(s["n_independent_participants"], 2)
        self.assertAlmostEqual(s["mean_probability_points"], 40.0)   # (80 + 0) / 2

    def test_order_invariance(self):
        import random
        rs = [PairedScore(f"p{i % 14}", 1 + ((i // 14) % 2), *(random.Random(i).random() for _ in range(4)))
              for i in range(140)]
        a = participant_scores(rs)
        random.Random(7).shuffle(rs)
        b = participant_scores(rs)
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=12)

    def test_extra_or_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            participant_scores([PairedScore("p1", 1, .5, .5, .5, .5), PairedScore("p1", 2, .5, .5, .5, .5),
                                PairedScore("p1", 3, .5, .5, .5, .5), PairedScore("p2", 1, .5, .5, .5, .5),
                                PairedScore("p2", 2, .5, .5, .5, .5)])
        with self.assertRaises(ValueError):
            participant_scores([PairedScore("p1", 1, .5, .5, .5, .5), PairedScore("p2", 1, .5, .5, .5, .5),
                                PairedScore("p2", 2, .5, .5, .5, .5)])

    def test_fourteen_participants_summary_shape(self):
        rs = [PairedScore(f"p{i:02d}", k, .6 + .01 * i, .5, .5, .6 + .01 * i) for i in range(14) for k in (1, 2)]
        s = summarize_participants(rs)
        self.assertEqual(s["n_independent_participants"], 14)
        self.assertEqual(s["n_positive"], 14)
        self.assertEqual(s["exact_sign_flip_p"], 2 / 2 ** 14)
        self.assertLess(s["ci95_lower_probability_points"], s["mean_probability_points"])

    def test_balanced_accuracy_contrast_equals_ba_difference(self):
        import random
        rng = random.Random(3)
        rs = [PairedCorrect(f"p{i % 3}", 1 + (i % 2), *(rng.random() < .6 for _ in range(4))) for i in range(60)]
        direct = {}
        for p in {r.participant for r in rs}:
            ba_own, ba_partner = [], []
            for k in (1, 2):
                sub = [r for r in rs if r.participant == p and r.label == k]
                own = [r.c_a_given_a for r in sub] + [r.c_b_given_b for r in sub]
                par = [r.c_a_given_b for r in sub] + [r.c_b_given_a for r in sub]
                ba_own.append(sum(own) / len(own))
                ba_partner.append(sum(par) / len(par))
            direct[p] = sum(ba_own) / 2 - sum(ba_partner) / 2
        agg = participant_class_balanced(rs)
        for p in direct:
            self.assertAlmostEqual(agg[p], direct[p], places=12)
        self.assertEqual(summarize_balanced_accuracy_contrast(rs)["n_independent_participants"], 3)

    def test_correct_class_probability_picks_true_class(self):
        proba = [[.8, .2], [.3, .7]]
        self.assertEqual(list(correct_class_probability(proba, [1, 2], [1, 1])), [.8, .3])
        with self.assertRaises(ValueError):
            correct_class_probability(proba, [1, 2], [1, 3])
        with self.assertRaises(ValueError):
            correct_class_probability([[.8, .3], [.3, .7]], [1, 2], [1, 2])

    def test_position_strata_drop_and_count(self):
        rs = [PairedScore(f"p{i}", k, .6, .5, .5, .6, position_distance=d)
              for i in range(4) for k in (1, 2) for d in (2, 9)]
        rs = [r for r in rs if not (r.participant == "p3" and r.label == 2 and r.position_distance == 9)]
        out = position_strata(rs, cut=6)
        self.assertEqual(out["near"]["n_participants_dropped"], 0)
        self.assertEqual(out["far"]["n_participants_dropped"], 1)
        self.assertEqual(out["far"]["n_independent_participants"], 3)


class ExactInference(unittest.TestCase):
    def test_sign_flip_floor_n14(self):
        d = list(range(1, 15))
        self.assertEqual(exact_two_sided_sign_flip(d), 2 / 2 ** 14)
        self.assertEqual(sign_flip_floor(14), 0.0001220703125)
        self.assertEqual(exact_two_sided_sign_flip([-x for x in d]), 2 / 2 ** 14)
        d[0] = -1                        # smallest value negative: subsets {} and {1} reach |sum| >= 103
        self.assertEqual(exact_two_sided_sign_flip(d), 4 / 2 ** 14)

    def test_sign_flip_matches_frozen_r4_implementation(self):
        import numpy as np
        from models.vendor_r4 import r4_stats
        rng = np.random.default_rng(0)
        for n in range(2, 15):
            for _ in range(3):
                v = rng.normal(0.2, 1.0, n)
                self.assertAlmostEqual(exact_two_sided_sign_flip(v), r4_stats.sign_flip_p(v), places=12)
        ties = np.array([1.0, -1.0, 2.0, -2.0])
        self.assertEqual(exact_two_sided_sign_flip(ties), r4_stats.sign_flip_p(ties))

    def test_sign_flip_limits(self):
        self.assertAlmostEqual(exact_two_sided_sign_flip([1.0] * 20), 2 / 2 ** 20)
        with self.assertRaises(ValueError):
            exact_two_sided_sign_flip([1.0] * 21)
        with self.assertRaises(ValueError):
            exact_two_sided_sign_flip([1.0, float("nan")])

    def test_t_interval_and_holm_match_r4(self):
        import numpy as np
        from models.vendor_r4 import r4_stats
        v = np.linspace(-1, 3, 14)
        self.assertEqual(tuple(round(x, 12) for x in t_interval(v, .95)),
                         tuple(round(x, 12) for x in r4_stats.t_interval(v, .95)))
        p = {"a": .01, "b": .04, "c": .03}
        self.assertEqual(holm(p), r4_stats.holm(p))

    def test_minimum_detectable_effect_n14(self):
        dz = minimum_detectable_dz(14)
        self.assertTrue(0.75 < dz < 0.90, dz)


class ControlArms(unittest.TestCase):
    def setUp(self):
        f = fold_fixture()
        self.f = f
        self.cb = check_base(base=f["base"], test=f["test"], canonical_base=True, test_fold=TEST_FOLD)
        self.d = next(r for r in f["test"] if r.trial_id == "d")      # same session/class, origin run 2

    def test_valid_neighbour_arm_is_support_matched(self):
        from experiments.pmis_g2.contracts import check_neighbour_arm
        n = check_neighbour_arm(self.cb, a=self.f["a"], alias_a=self.f["alias_a"], g=self.d, alias_g=copy_of("d", 3))
        self.assertEqual(n, validate_pair(**self.f).distinct_support)

    def test_neighbour_rejections(self):
        from experiments.pmis_g2.contracts import check_neighbour_arm
        e = next(r for r in self.f["test"] if r.trial_id == "e")      # other class
        cases = {
            "neighbour_same_run": dict(g=self.f["b"], alias_g=self.f["alias_b"]),
            "neighbour_scope": dict(g=e, alias_g=copy_of("e", 3)),
            "alias_file": dict(g=self.d, alias_g=copy_of("d", 4)),
            "alias_in_test_fold": dict(g=self.d, alias_g=replace(copy_of("d", 3), fold=0)),
        }
        for code, kw in cases.items():
            with self.assertRaises(ContractError) as ctx:
                check_neighbour_arm(self.cb, a=self.f["a"], alias_a=self.f["alias_a"], **kw)
            self.assertIn(ctx.exception.code, (code, "split_not_whole_file"), code)

    def test_placebo_arms_canonical_only_and_scoped(self):
        from experiments.pmis_g2.contracts import check_placebo_arms
        base = self.f["base"] + [frow("c7", 3, label=1, origin=3)]
        TRIALS["c7"] = (3, 1)
        try:
            cb = check_base(base=base, test=self.f["test"], canonical_base=True, test_fold=TEST_FOLD)
            fk = ("p0", "s0", 4)
            self.assertEqual(check_placebo_arms(cb, c_copy=copy_of("c3", 4), e_copy=copy_of("c7", 4), alias_file=fk),
                             len(cb.base_trial_ids))
            bad = {"placebo_scope": dict(c_copy=copy_of("c3", 4), e_copy=copy_of("c4", 4)),
                   "same_trial": dict(c_copy=copy_of("c3", 4), e_copy=replace(copy_of("c3", 4), row_id="x")),
                   "alias_file": dict(c_copy=copy_of("c3", 4), e_copy=replace(copy_of("c7", 4), file_run=5, fold=2)),
                   "placebo_not_base_identity": dict(c_copy=copy_of("a", 4), e_copy=copy_of("c7", 4))}
            for code, kw in bad.items():
                with self.assertRaises(ContractError, msg=code) as ctx:
                    check_placebo_arms(cb, alias_file=fk, **kw)
                self.assertEqual(ctx.exception.code, code)
            mcb = check_base(base=base + [copy_of("c3", 4)], test=self.f["test"], canonical_base=False,
                             test_fold=TEST_FOLD)
            with self.assertRaises(ContractError) as ctx:
                check_placebo_arms(mcb, c_copy=copy_of("c3", 4), e_copy=copy_of("c7", 4), alias_file=fk)
            self.assertEqual(ctx.exception.code, "placebo_requires_canonical_base")
        finally:
            TRIALS.pop("c7")
