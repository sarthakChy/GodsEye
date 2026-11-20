"""
ExamGuard-Causal Rule Engine
=============================
Ingests a list of TemporalRelation intervals (the output of TemporalAggregator)
and flags four categories of academic malpractice with timestamps, duration,
involved parties, and a numeric confidence score.

Rule taxonomy
-------------
RULE-1  Chit / Material Transfer
        An object (book | paper | notebook) first observed as held by student_A,
        then observed as held by student_B, within a hand-off window.
        Prerequisite: both holding intervals must partially overlap or lie within
        TRANSFER_WINDOW_SEC of each other and must share the SAME object_id.

RULE-2  Forbidden Electronic Device
        Any "holding" or "using" relation whose object_id class is in
        FORBIDDEN_DEVICES (phone, cell_phone, mobile, smartphone).

RULE-3  Desk Encroachment / Leaning
        "near" or "reaching toward" relation from student_A to desk_B
        (or any non-student object at a desk position) persisting for
        >= ENCROACHMENT_MIN_SEC seconds.

RULE-4  Unauthorized Direct Communication
        "talking to" predicate between two persons, or a "looking at"
        predicate that persists >= GAZE_MIN_SEC seconds between two persons
        during the exam.

Each violation is returned as a MalpracticeEvent dataclass that is fully
JSON-serializable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from temporal.schema import TemporalRelation

# ---------------------------------------------------------------------------
# Tuneable thresholds (all units: seconds unless stated)
# ---------------------------------------------------------------------------

TRANSFER_WINDOW_SEC: float = 8.0    # max gap between holding-A and holding-B
ENCROACHMENT_MIN_SEC: float = 3.0   # desk-near dwell time to flag
GAZE_MIN_SEC: float = 4.0           # looking-at dwell time to flag
MIN_RULE_CONFIDENCE: float = 0.25   # ignore intervals below this raw score

# ---------------------------------------------------------------------------
# Vocabulary sets (all lower-case; matched case-insensitively)
# ---------------------------------------------------------------------------

FORBIDDEN_DEVICES: frozenset[str] = frozenset({
    "phone", "cell_phone", "cellphone", "mobile", "smartphone",
    "mobile phone", "cell phone",
})

MATERIAL_OBJECTS: frozenset[str] = frozenset({
    "book", "notebook", "paper", "chit", "note", "sheet", "document",
    "answer sheet", "question paper",
})

DESK_OBJECTS: frozenset[str] = frozenset({
    "desk", "table", "bench",
})

PERSON_CLASS: str = "person"

HOLDING_PREDICATES: frozenset[str] = frozenset({"holding", "carrying"})
USING_PREDICATES:   frozenset[str] = frozenset({"using", "operating"})
NEAR_PREDICATES:    frozenset[str] = frozenset({"near", "next to", "beside", "reaching toward", "reaching towards"})
TALK_PREDICATES:    frozenset[str] = frozenset({"talking to", "talking with", "speaking to"})
GAZE_PREDICATES:    frozenset[str] = frozenset({"looking at", "watching", "staring at"})

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}


@dataclass
class MalpracticeEvent:
    rule_id: str                        # "RULE-1" … "RULE-4"
    description: str                    # human-readable summary
    start_time: float                   # seconds into video
    end_time: float                     # seconds into video
    duration: float                     # end_time - start_time
    subjects: List[str]                 # semantic IDs of involved persons
    objects: List[str]                  # semantic IDs of involved objects/persons
    confidence: float                   # 0.0–1.0 aggregated score
    severity: int                       # 1=LOW  2=MEDIUM  3=HIGH  4=CRITICAL
    severity_label: str                 # human label
    source_relations: List[str] = field(default_factory=list)  # relation keys

    # convenience ---------
    @property
    def time_range(self) -> str:
        return f"{self.start_time:.2f}s – {self.end_time:.2f}s"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_log_line(self) -> str:
        return (
            f"[{self.severity_label}] {self.rule_id} @ {self.time_range} "
            f"({self.duration:.1f}s) | {self.description} | conf={self.confidence:.2f}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _obj_class(semantic_id: str) -> str:
    """'person_02' -> 'person',  'cell_phone_01' -> 'cell_phone'."""
    parts = semantic_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].lower()
    return semantic_id.lower()


def _is_person(semantic_id: str) -> bool:
    return _obj_class(semantic_id) == PERSON_CLASS


def _in_set(semantic_id: str, vocab: frozenset[str]) -> bool:
    return _obj_class(semantic_id) in vocab


def _intervals_overlap_or_close(a: TemporalRelation, b: TemporalRelation,
                                 window: float) -> bool:
    """True if the two intervals overlap or start within `window` of each other."""
    latest_start = max(a.start_time, b.start_time)
    earliest_end = min(a.end_time, b.end_time)
    if latest_start <= earliest_end:          # actual overlap
        return True
    gap = latest_start - earliest_end
    return gap <= window


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _duration_confidence(duration: float, threshold: float,
                          cap: float = 30.0) -> float:
    """Score that grows with duration relative to threshold."""
    return _clamp01((duration - threshold) / max(cap - threshold, 1.0) + 0.5)


def _relation_key(r: TemporalRelation) -> str:
    return f"{r.subject_id}|{r.predicate}|{r.object_id}@{r.start_time:.2f}"


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def _rule1_material_transfer(
    events: List[TemporalRelation],
    window: float = TRANSFER_WINDOW_SEC,
) -> List[MalpracticeEvent]:
    """
    RULE-1 — Chit / Material Transfer

    Looks for two "holding" intervals on the SAME object_id where:
      - subject of interval A is a person
      - subject of interval B is a *different* person
      - the two intervals are within `window` seconds of each other

    Confidence = average of the two interval mean_scores × handoff recency factor.
    """
    violations: List[MalpracticeEvent] = []

    # group holding relations by object
    held_by_object: Dict[str, List[TemporalRelation]] = {}
    for rel in events:
        if rel.predicate.lower() not in HOLDING_PREDICATES:
            continue
        if rel.mean_score < MIN_RULE_CONFIDENCE:
            continue
        if not _is_person(rel.subject_id):
            continue
        if not _in_set(rel.object_id, MATERIAL_OBJECTS):
            continue
        held_by_object.setdefault(rel.object_id, []).append(rel)

    for obj_id, rels in held_by_object.items():
        # sort by start time
        rels = sorted(rels, key=lambda r: r.start_time)
        seen_pairs: set[Tuple[str, str]] = set()

        for i, ra in enumerate(rels):
            for rb in rels[i + 1:]:
                if ra.subject_id == rb.subject_id:
                    continue
                pair_key = tuple(sorted([ra.subject_id, rb.subject_id]))
                if pair_key in seen_pairs:
                    continue
                if not _intervals_overlap_or_close(ra, rb, window):
                    continue
                seen_pairs.add(pair_key)

                t_start = min(ra.start_time, rb.start_time)
                t_end   = max(ra.end_time,   rb.end_time)
                dur     = t_end - t_start
                # gap penalty: smaller gap → higher conf
                gap = max(0.0, max(ra.start_time, rb.start_time)
                          - min(ra.end_time, rb.end_time))
                gap_factor = _clamp01(1.0 - gap / window)
                conf = _clamp01((ra.mean_score + rb.mean_score) / 2.0 * gap_factor)

                violations.append(MalpracticeEvent(
                    rule_id="RULE-1",
                    description=(
                        f"Material transfer: {_obj_class(obj_id)} passed from "
                        f"{ra.subject_id} to {rb.subject_id}"
                    ),
                    start_time=t_start,
                    end_time=t_end,
                    duration=dur,
                    subjects=[ra.subject_id, rb.subject_id],
                    objects=[obj_id],
                    confidence=conf,
                    severity=4,
                    severity_label=SEVERITY_LABELS[4],
                    source_relations=[_relation_key(ra), _relation_key(rb)],
                ))

    return violations


def _rule2_forbidden_device(
    events: List[TemporalRelation],
) -> List[MalpracticeEvent]:
    """
    RULE-2 — Forbidden Electronic Device

    Flags any "holding" or "using" relation from a person to a device in
    FORBIDDEN_DEVICES that meets the minimum confidence threshold.
    """
    violations: List[MalpracticeEvent] = []
    seen: set[str] = set()

    for rel in events:
        if rel.predicate.lower() not in (HOLDING_PREDICATES | USING_PREDICATES):
            continue
        if rel.mean_score < MIN_RULE_CONFIDENCE:
            continue
        if not _is_person(rel.subject_id):
            continue
        if not _in_set(rel.object_id, FORBIDDEN_DEVICES):
            continue

        key = _relation_key(rel)
        if key in seen:
            continue
        seen.add(key)

        dur = rel.end_time - rel.start_time
        conf = _clamp01(rel.mean_score * (1.0 + min(dur, 10.0) / 20.0))

        violations.append(MalpracticeEvent(
            rule_id="RULE-2",
            description=(
                f"Forbidden device: {rel.subject_id} {rel.predicate} "
                f"{rel.object_id}"
            ),
            start_time=rel.start_time,
            end_time=rel.end_time,
            duration=dur,
            subjects=[rel.subject_id],
            objects=[rel.object_id],
            confidence=conf,
            severity=3,
            severity_label=SEVERITY_LABELS[3],
            source_relations=[key],
        ))

    return violations


def _rule3_desk_encroachment(
    events: List[TemporalRelation],
    min_dwell: float = ENCROACHMENT_MIN_SEC,
) -> List[MalpracticeEvent]:
    """
    RULE-3 — Desk Encroachment / Leaning

    A person "near" / "next to" / "reaching toward" a desk-class object for
    >= min_dwell seconds is flagged.
    """
    violations: List[MalpracticeEvent] = []
    seen: set[str] = set()

    for rel in events:
        if rel.predicate.lower() not in NEAR_PREDICATES:
            continue
        if rel.mean_score < MIN_RULE_CONFIDENCE:
            continue
        if not _is_person(rel.subject_id):
            continue
        if not _in_set(rel.object_id, DESK_OBJECTS):
            continue

        dur = rel.end_time - rel.start_time
        if dur < min_dwell:
            continue

        key = _relation_key(rel)
        if key in seen:
            continue
        seen.add(key)

        conf = _clamp01(rel.mean_score * _duration_confidence(dur, min_dwell, cap=15.0))
        severity = 2 if dur < 6.0 else 3

        violations.append(MalpracticeEvent(
            rule_id="RULE-3",
            description=(
                f"Desk encroachment: {rel.subject_id} {rel.predicate} "
                f"{rel.object_id} for {dur:.1f}s"
            ),
            start_time=rel.start_time,
            end_time=rel.end_time,
            duration=dur,
            subjects=[rel.subject_id],
            objects=[rel.object_id],
            confidence=conf,
            severity=severity,
            severity_label=SEVERITY_LABELS[severity],
            source_relations=[key],
        ))

    return violations


def _rule4_unauthorized_communication(
    events: List[TemporalRelation],
    gaze_min: float = GAZE_MIN_SEC,
) -> List[MalpracticeEvent]:
    """
    RULE-4 — Unauthorized Direct Communication

    (a) Any "talking to" / "talking with" interval between two persons.
    (b) "Looking at" / "watching" interval between two persons that lasts
        >= gaze_min seconds.
    """
    violations: List[MalpracticeEvent] = []
    seen: set[str] = set()

    for rel in events:
        pred = rel.predicate.lower()
        if rel.mean_score < MIN_RULE_CONFIDENCE:
            continue
        if not _is_person(rel.subject_id):
            continue
        if not _is_person(rel.object_id):
            continue

        is_talk = pred in TALK_PREDICATES
        is_gaze = pred in GAZE_PREDICATES

        if not (is_talk or is_gaze):
            continue

        dur = rel.end_time - rel.start_time

        if is_gaze and dur < gaze_min:
            continue

        key = _relation_key(rel)
        if key in seen:
            continue
        seen.add(key)

        if is_talk:
            conf = _clamp01(rel.mean_score)
            severity = 3
        else:
            conf = _clamp01(rel.mean_score * _duration_confidence(dur, gaze_min, cap=20.0))
            severity = 2

        label = "talking" if is_talk else f"prolonged gaze ({dur:.1f}s)"
        violations.append(MalpracticeEvent(
            rule_id="RULE-4",
            description=(
                f"Unauthorized communication: {rel.subject_id} {label} "
                f"→ {rel.object_id}"
            ),
            start_time=rel.start_time,
            end_time=rel.end_time,
            duration=dur,
            subjects=[rel.subject_id, rel.object_id],
            objects=[],
            confidence=conf,
            severity=severity,
            severity_label=SEVERITY_LABELS[severity],
            source_relations=[key],
        ))

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_exam_rules(
    events: List[TemporalRelation],
    *,
    transfer_window: float = TRANSFER_WINDOW_SEC,
    encroachment_min_sec: float = ENCROACHMENT_MIN_SEC,
    gaze_min_sec: float = GAZE_MIN_SEC,
    min_confidence: float = MIN_RULE_CONFIDENCE,
) -> List[MalpracticeEvent]:
    """
    Run all four ExamGuard rules over a list of TemporalRelation intervals.

    Parameters
    ----------
    events : list of TemporalRelation
        All temporal relation intervals for the video (from TemporalAggregator).
    transfer_window : float
        Max gap (seconds) between two holding intervals to infer a transfer.
    encroachment_min_sec : float
        Min dwell time (seconds) near a desk to flag RULE-3.
    gaze_min_sec : float
        Min gaze duration (seconds) between persons to flag RULE-4b.
    min_confidence : float
        Intervals whose mean_score is below this are ignored globally.

    Returns
    -------
    list of MalpracticeEvent, sorted by start_time.
    """
    global MIN_RULE_CONFIDENCE
    MIN_RULE_CONFIDENCE = min_confidence

    violations: List[MalpracticeEvent] = []
    violations += _rule1_material_transfer(events, window=transfer_window)
    violations += _rule2_forbidden_device(events)
    violations += _rule3_desk_encroachment(events, min_dwell=encroachment_min_sec)
    violations += _rule4_unauthorized_communication(events, gaze_min=gaze_min_sec)

    violations.sort(key=lambda v: v.start_time)
    return violations


def violations_at_time(
    violations: List[MalpracticeEvent],
    timestamp: float,
    lookahead: float = 0.0,
) -> List[MalpracticeEvent]:
    """
    Filter violations active at or just before `timestamp`.

    Parameters
    ----------
    violations : pre-computed list from run_exam_rules()
    timestamp  : current playhead (seconds)
    lookahead  : show violations that *start* within this many seconds
                 in the future (useful for dashboard preview).
    """
    return [
        v for v in violations
        if v.start_time <= timestamp + lookahead and v.end_time >= timestamp
    ]


# ---------------------------------------------------------------------------
# CLI helper (python -m temporal.exam_rules <run_dir>)
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run ExamGuard rules on a GodsEye run directory."
    )
    ap.add_argument("run_dir", help="Path to outputs/<run> containing events.json")
    ap.add_argument("--transfer-window",   type=float, default=TRANSFER_WINDOW_SEC)
    ap.add_argument("--encroachment-sec",  type=float, default=ENCROACHMENT_MIN_SEC)
    ap.add_argument("--gaze-sec",          type=float, default=GAZE_MIN_SEC)
    ap.add_argument("--min-confidence",    type=float, default=MIN_RULE_CONFIDENCE)
    ap.add_argument("--out", default=None, help="Write violations.json here (default: run_dir)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    raw = json.loads((run_dir / "events.json").read_text())
    events = [TemporalRelation(**r) for r in raw]
    print(f"Loaded {len(events)} temporal relations from {run_dir/'events.json'}")

    violations = run_exam_rules(
        events,
        transfer_window=args.transfer_window,
        encroachment_min_sec=args.encroachment_sec,
        gaze_min_sec=args.gaze_sec,
        min_confidence=args.min_confidence,
    )

    print(f"\n{'='*60}")
    print(f"ExamGuard Audit: {len(violations)} violation(s) detected")
    print(f"{'='*60}")
    for v in violations:
        print(f"  {v.to_log_line()}")

    out_path = Path(args.out) if args.out else run_dir / "violations.json"
    out_path.write_text(
        json.dumps([v.to_dict() for v in violations], indent=2)
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    _cli()
