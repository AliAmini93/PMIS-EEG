"""Synthetic-safe PMIS identity-swap metadata invariants.

Does not read EEG, models, trial features, or private R4 manifests. The real-metadata adapter
(`pmis_g2.adapter`) builds TrialRow objects from R4 metadata in memory and calls these checks; any
derived per-trial output stays in git-ignored data/.

G2 review changes (see reports/G2_CODE_REVIEW.md): the validator is split into a fold-level
`check_base` (run once per participant x fold) and a per-pair `check_pair` (O(1) per pair), and it now
also enforces: row-level canonical/file-run sanity, single-participant scope, global row-id
uniqueness, a trial-id <-> waveform-hash bijection (alternate-ID leaks), exactly one canonical row per
base identity ("missing canonical train source"), S2 whole-file fold assignment, and same physical
alias FILE (participant, session, file-run). `validate_pair` keeps its original signature.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TrialRow:
    row_id: str
    trial_id: str
    waveform_sha256: str
    participant: str
    session: str
    origin_run: int
    file_run: int
    label: int
    partition: Literal["train", "test"]
    canonical: bool
    # S2 fold of this row's run FILE. None disables fold checks (pure-structure synthetic tests only);
    # the real-metadata adapter always sets it.
    fold: Optional[int] = None


@dataclass(frozen=True)
class CheckedBase:
    base: tuple[TrialRow, ...]
    test: tuple[TrialRow, ...]
    canonical_base: bool
    test_fold: Optional[int]
    participant: str
    test_rows: frozenset
    base_trial_ids: frozenset
    base_row_ids: frozenset
    test_trial_ids: frozenset
    test_hashes: frozenset
    test_row_ids: frozenset
    id_to_hash: Mapping[str, str]
    hash_to_id: Mapping[str, str]
    file_fold: Mapping[tuple, int]


@dataclass(frozen=True)
class CheckedPair:
    # Pure metadata output. Never write real identifiers to Git.
    arm_a: tuple[TrialRow, ...]
    arm_b: tuple[TrialRow, ...]
    test: tuple[TrialRow, ...]
    distinct_support: int
    base_rows: int


class ContractError(ValueError):
    """Fails closed when a scientific invariant is violated. `code` names the invariant."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

    def __reduce__(self):  # survives pickling from pool workers
        return (ContractError, (self.code, self.message))


def _require(ok: bool, code: str, message: str) -> None:
    if not ok:
        raise ContractError(code, message)


def _row_identical(x: TrialRow, y: TrialRow) -> bool:
    return x.trial_id == y.trial_id and x.waveform_sha256 == y.waveform_sha256


def _provenance(r: TrialRow) -> tuple:
    return (r.participant, r.session, r.origin_run, r.label)


def _file_key(r: TrialRow) -> tuple:
    return (r.participant, r.session, r.file_run)


def _row_sanity(r: TrialRow) -> None:
    _require(r.partition in ("train", "test"), "partition_value", "partition must be train or test")
    _require(isinstance(r.label, int) and not isinstance(r.label, bool), "label_type", "label must be int")
    if r.canonical:
        _require(r.file_run == r.origin_run, "canonical_not_origin_file",
                 "a canonical row must sit in its acquisition (origin) run file")
    else:
        _require(r.file_run > r.origin_run, "copy_not_later",
                 "a non-canonical copy must sit in a LATER cumulative run file")


def check_base(*, base: Sequence[TrialRow], test: Sequence[TrialRow], canonical_base: bool = True,
               test_fold: Optional[int] = None) -> CheckedBase:
    """Fold-level invariants of the common base C and the scored test set T (once per fold)."""
    C, T = tuple(base), tuple(test)
    _require(len(C) > 0 and len(T) >= 2, "size", "base/test empty or too small")
    _require(all(r.partition == "train" for r in C), "base_partition", "base includes nontraining row")
    _require(all(r.partition == "test" and r.canonical for r in T), "test_canonical", "test must be canonical")
    for r in (*C, *T):
        _row_sanity(r)
    participants = {r.participant for r in (*C, *T)}
    _require(len(participants) == 1, "participant_scope",
             "base and test must come from ONE participant (S2 per-participant folds)")

    _require(len({r.row_id for r in T}) == len(T), "test_row_repeat", "test repeats instance row")
    _require(len({r.trial_id for r in T}) == len(T), "test_identity_repeat", "test repeats physical identity")
    _require(len({r.waveform_sha256 for r in T}) == len(T), "test_waveform_repeat", "test repeats physical waveform")
    base_row_ids = [r.row_id for r in C]
    _require(len(set(base_row_ids)) == len(base_row_ids), "base_row_repeat", "duplicate training instance row")
    _require(not (set(base_row_ids) & {r.row_id for r in T}), "row_id_shared",
             "a row id occurs in both base and test")

    id_to_hash: dict[str, str] = {}
    hash_to_id: dict[str, str] = {}
    for r in (*C, *T):
        old_h = id_to_hash.setdefault(r.trial_id, r.waveform_sha256)
        _require(old_h == r.waveform_sha256, "id_hash_conflict", "same trial id has different waveforms")
        old_i = hash_to_id.setdefault(r.waveform_sha256, r.trial_id)
        _require(old_i == r.trial_id, "hash_alternate_id", "same waveform appears under two trial ids")

    test_ids = {r.trial_id for r in T}
    test_hashes = {r.waveform_sha256 for r in T}
    base_ids = {r.trial_id for r in C}
    _require(not (test_ids & base_ids), "test_identity_in_base", "test identity leaked into base")
    _require(not (test_hashes & {r.waveform_sha256 for r in C}), "test_waveform_in_base",
             "test waveform leaked into base")

    canon_count = Counter(r.trial_id for r in C if r.canonical)
    _require(all(canon_count.get(i, 0) == 1 for i in base_ids), "canonical_source",
             "every base identity needs exactly one canonical training-side row (missing canonical train source)")
    if canonical_base:
        _require(all(r.canonical for r in C), "base_noncanonical", "base includes noncanonical rows")
        _require(len(base_ids) == len(C), "canonical_base_duplicate", "canonical base has duplicates")

    file_fold: dict[tuple, int] = {}
    if test_fold is None:
        _require(all(r.fold is None for r in (*C, *T)), "fold_unchecked",
                 "rows carry fold metadata but no test_fold was given")
    else:
        _require(all(r.fold is not None for r in (*C, *T)), "fold_missing", "fold metadata missing")
        _require(all(r.fold == test_fold for r in T), "test_fold", "scored row outside the test fold")
        _require(all(r.fold != test_fold for r in C), "base_in_test_fold", "base row inside the test fold")
        for r in (*C, *T):
            old = file_fold.setdefault(_file_key(r), r.fold)
            _require(old == r.fold, "split_not_whole_file", "one run file is split across folds")

    return CheckedBase(
        base=C, test=T, canonical_base=canonical_base, test_fold=test_fold,
        participant=next(iter(participants)), test_rows=frozenset(T),
        base_trial_ids=frozenset(base_ids), base_row_ids=frozenset(base_row_ids),
        test_trial_ids=frozenset(test_ids), test_hashes=frozenset(test_hashes),
        test_row_ids=frozenset(r.row_id for r in T),
        id_to_hash=MappingProxyType(id_to_hash), hash_to_id=MappingProxyType(hash_to_id),
        file_fold=MappingProxyType(file_fold))


def check_pair(cb: CheckedBase, *, a: TrialRow, b: TrialRow, alias_a: TrialRow, alias_b: TrialRow,
               build_arms: bool = True) -> CheckedPair:
    """Per-pair invariants on top of a checked base. O(1) per pair except optional arm materialisation.

    a/b must both be scored test trials of the same participant, session, acquisition run and class,
    physically distinct. Both training aliases must be bit-identical non-canonical copies residing in
    the SAME later physical run FILE (participant, session, file-run) on the training side. The two arms
    C+[a'] and C+[b'] then have equal rows, equal distinct support |C|+1, equal class histograms, and
    expose exactly one scored identity each.
    """
    _require(a in cb.test_rows and b in cb.test_rows, "not_scored", "a or b absent from scored test")
    _require(a.trial_id != b.trial_id and a.waveform_sha256 != b.waveform_sha256, "same_trial",
             "a and b must be distinct physical trials")
    _require(_provenance(a) == _provenance(b), "source_provenance",
             "source participant/session/run/class mismatch")
    _require(a.file_run == a.origin_run and b.file_run == b.origin_run, "test_not_origin",
             "canonical test source run must be origin run")
    for al in (alias_a, alias_b):
        _row_sanity(al)
    _require(alias_a.partition == alias_b.partition == "train", "alias_partition", "alias not on training side")
    _require(not alias_a.canonical and not alias_b.canonical, "alias_canonical", "alias must be noncanonical")
    _require(_row_identical(a, alias_a) and _row_identical(b, alias_b), "alias_not_identical",
             "training aliases must be bit-identical to their scored trials")
    _require(_provenance(a) == _provenance(alias_a) == _provenance(alias_b), "alias_provenance",
             "alias provenance mismatch")
    _require(_file_key(alias_a) == _file_key(alias_b), "alias_file",
             "aliases must reside in the SAME physical later run file")
    _require(alias_a.file_run > a.origin_run and alias_b.file_run > b.origin_run, "alias_not_later",
             "alias must be in a later cumulative run file")
    _require(alias_a.row_id != alias_b.row_id, "alias_row_reuse", "same training row used twice")
    for al in (alias_a, alias_b):
        _require(al.row_id not in cb.base_row_ids, "alias_in_base", "candidate alias already belongs to base")
        _require(al.row_id not in cb.test_row_ids, "alias_row_is_test", "alias row id is a scored test row")
        _require(cb.id_to_hash.get(al.trial_id) == al.waveform_sha256, "id_hash_conflict",
                 "alias trial id / waveform inconsistent with scored set")
        _require(cb.hash_to_id.get(al.waveform_sha256) == al.trial_id, "hash_alternate_id",
                 "alias waveform registered under another trial id")
    if cb.test_fold is None:
        _require(alias_a.fold is None and alias_b.fold is None, "fold_unchecked",
                 "alias carries fold metadata but no test_fold was given")
    else:
        for al in (alias_a, alias_b):
            _require(al.fold is not None and al.fold != cb.test_fold, "alias_in_test_fold",
                     "alias file belongs to the test fold")
            known = cb.file_fold.get(_file_key(al))
            _require(known is None or known == al.fold, "split_not_whole_file",
                     "alias file fold disagrees with another row of the same file")
        _require(alias_a.fold == alias_b.fold, "split_not_whole_file", "one alias file in two folds")

    # Arms. a', b' are scored identities, hence disjoint from the base by check_base.
    n_support = len(cb.base_trial_ids) + 1
    _require(alias_a.label == alias_b.label, "class_histogram", "training class histogram mismatch")
    exposed_a = {alias_a.trial_id} & cb.test_trial_ids
    exposed_b = {alias_b.trial_id} & cb.test_trial_ids
    _require(exposed_a == {a.trial_id}, "arm_a_exposure", "arm A must expose exactly a")
    _require(exposed_b == {b.trial_id}, "arm_b_exposure", "arm B must expose exactly b")
    _require({alias_a.waveform_sha256} & cb.test_hashes == {a.waveform_sha256}, "arm_a_exposure",
             "arm A has extra test waveform")
    _require({alias_b.waveform_sha256} & cb.test_hashes == {b.waveform_sha256}, "arm_b_exposure",
             "arm B has extra test waveform")
    if build_arms:
        A = cb.base + (alias_a,)
        B = cb.base + (alias_b,)
        _require(len(A) == len(B), "rows", "training row count mismatch")
    else:
        A = B = ()
    return CheckedPair(A, B, cb.test, n_support, len(cb.base))


def validate_pair(
    *,
    base: Sequence[TrialRow],
    test: Sequence[TrialRow],
    a: TrialRow,
    b: TrialRow,
    alias_a: TrialRow,
    alias_b: TrialRow,
    canonical_base: bool = True,
    test_fold: Optional[int] = None,
) -> CheckedPair:
    """Validate two arms before any features, predictions or fitting are read (fold + pair checks)."""
    cb = check_base(base=base, test=test, canonical_base=canonical_base, test_fold=test_fold)
    out = check_pair(cb, a=a, b=b, alias_a=alias_a, alias_b=alias_b)
    # Explicit recount on the materialised arms (independent of the O(1) bookkeeping above).
    A, B = out.arm_a, out.arm_b
    test_ids = cb.test_trial_ids
    distinct_A = {r.trial_id for r in A}
    distinct_B = {r.trial_id for r in B}
    _require(len(distinct_A) == len(distinct_B) == out.distinct_support, "distinct_support",
             "training distinct-support mismatch")
    _require(Counter(r.label for r in A) == Counter(r.label for r in B), "class_histogram",
             "training class histogram mismatch")
    _require(distinct_A & test_ids == {a.trial_id}, "arm_a_exposure", "arm A has extra test identity")
    _require(distinct_B & test_ids == {b.trial_id}, "arm_b_exposure", "arm B has extra test identity")
    return out


# ------------------------------------------------------------------------------- pre-declared controls
def _check_training_copy(cb: CheckedBase, src_trial: Optional[TrialRow], al: TrialRow) -> None:
    """Shared checks for a training-side non-canonical copy row added on top of the base."""
    _row_sanity(al)
    _require(al.partition == "train" and not al.canonical, "alias_canonical",
             "control row must be a training-side non-canonical copy")
    if src_trial is not None:
        _require(_row_identical(src_trial, al) and _provenance(src_trial) == _provenance(al),
                 "alias_not_identical", "control copy must be bit-identical to its source trial")
    _require(al.file_run > al.origin_run, "alias_not_later", "control copy must sit in a later file")
    _require(al.row_id not in cb.base_row_ids and al.row_id not in cb.test_row_ids, "alias_in_base",
             "control row already in base or scored set")
    _require(cb.id_to_hash.get(al.trial_id) == al.waveform_sha256, "id_hash_conflict",
             "control copy id / waveform inconsistent")
    _require(cb.hash_to_id.get(al.waveform_sha256) == al.trial_id, "hash_alternate_id",
             "control copy waveform registered under another id")
    if cb.test_fold is not None:
        _require(al.fold is not None and al.fold != cb.test_fold, "alias_in_test_fold",
                 "control copy file belongs to the test fold")
        known = cb.file_fold.get(_file_key(al))
        _require(known is None or known == al.fold, "split_not_whole_file", "control file fold mismatch")


def check_neighbour_arm(cb: CheckedBase, *, a: TrialRow, alias_a: TrialRow, g: TrialRow,
                        alias_g: TrialRow) -> int:
    """Support-matched different-run neighbour arm C+[g'] for a PMIS pair whose alias file holds a'.

    g is a scored trial of the same participant, session and class as a but from ANOTHER acquisition
    run; g' is its bit-identical copy in the SAME later file as a'. C+[g'] then has the same rows,
    distinct support (|C|+1) and class histogram as C+[a'] and C+[b'], and exposes only g."""
    _require(a in cb.test_rows and g in cb.test_rows, "not_scored", "a or g absent from scored set")
    _require(g.trial_id != a.trial_id and g.waveform_sha256 != a.waveform_sha256, "same_trial",
             "neighbour must be a different physical trial")
    _require((g.participant, g.session, g.label) == (a.participant, a.session, a.label), "neighbour_scope",
             "neighbour must share participant, session and class")
    _require(g.origin_run != a.origin_run, "neighbour_same_run", "neighbour must come from another run")
    _check_training_copy(cb, g, alias_g)
    _require(_file_key(alias_g) == _file_key(alias_a), "alias_file", "neighbour copy must sit in the pair's file")
    _require({alias_g.trial_id} & cb.test_trial_ids == {g.trial_id}, "arm_g_exposure", "arm G must expose only g")
    return len(cb.base_trial_ids) + 1


def check_placebo_arms(cb: CheckedBase, *, c_copy: TrialRow, e_copy: TrialRow, alias_file: tuple) -> int:
    """Placebo replacement arms C+[c'] and C+[e'] of two NON-test base trials (canonical base only).

    c, e share session and class and have copies in the pair's alias file; c', e' are copies of base
    identities, so both arms have rows |C|+1 but support |C| (one less than the PMIS arms; declared).
    On a multiset base the copies are already base rows, so the placebo is undefined there."""
    _require(cb.canonical_base, "placebo_requires_canonical_base", "placebo arms need the canonical base")
    for al in (c_copy, e_copy):
        _check_training_copy(cb, None, al)
        _require(al.trial_id in cb.base_trial_ids, "placebo_not_base_identity",
                 "placebo copy must duplicate a base (non-test) identity")
        _require(_file_key(al) == tuple(alias_file), "alias_file", "placebo copy must sit in the pair's file")
    _require(c_copy.trial_id != e_copy.trial_id and c_copy.row_id != e_copy.row_id, "same_trial",
             "placebo copies must be two different trials")
    _require((c_copy.session, c_copy.label) == (e_copy.session, e_copy.label), "placebo_scope",
             "placebo copies must share session and class")
    return len(cb.base_trial_ids)
