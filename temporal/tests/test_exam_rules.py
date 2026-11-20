"""
Unit tests for temporal/exam_rules.py — ExamGuard-Causal Rule Engine
=====================================================================
Run with:
    cd d:\\Major - 1\\GodsEye
    ..\\env\\Scripts\\python -m pytest temporal/tests/test_exam_rules.py -v

Each test builds minimal synthetic TemporalRelation lists that should
(or should NOT) trigger each rule, then asserts on the results.
"""

import pytest
from temporal.schema import TemporalRelation
from temporal.exam_rules import (
    run_exam_rules,
    violations_at_time,
    MalpracticeEvent,
    TRANSFER_WINDOW_SEC,
    ENCROACHMENT_MIN_SEC,
    GAZE_MIN_SEC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(
    subject: str,
    predicate: str,
    obj: str,
    t0: float,
    t1: float,
    score: float = 0.75,
) -> TemporalRelation:
    return TemporalRelation(
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        start_time=t0,
        end_time=t1,
        mean_score=score,
        frame_count=max(1, int((t1 - t0) * 2)),
        confidence=score,
    )


def _run(rels, **kwargs):
    return run_exam_rules(rels, **kwargs)


# ---------------------------------------------------------------------------
# RULE-1 — Material Transfer
# ---------------------------------------------------------------------------

class TestRule1MaterialTransfer:

    def test_basic_transfer_detected(self):
        """Two persons hold the SAME book within window → RULE-1 fires."""
        rels = [
            _rel("person_01", "holding", "book_01", 0.0, 4.0),
            _rel("person_02", "holding", "book_01", 5.0, 9.0),
        ]
        viols = _run(rels)
        r1 = [v for v in viols if v.rule_id == "RULE-1"]
        assert len(r1) == 1, f"Expected 1 RULE-1 violation, got {r1}"
        v = r1[0]
        assert "person_01" in v.subjects or "person_02" in v.subjects
        assert "book_01" in v.objects
        assert v.severity == 4

    def test_non_material_object_ignored(self):
        """Transfer of a 'chair' should NOT trigger RULE-1."""
        rels = [
            _rel("person_01", "holding", "chair_01", 0.0, 3.0),
            _rel("person_02", "holding", "chair_01", 4.0, 7.0),
        ]
        viols = _run(rels)
        r1 = [v for v in viols if v.rule_id == "RULE-1"]
        assert len(r1) == 0

    def test_same_person_no_transfer(self):
        """Same person holding the same paper twice → NOT a transfer."""
        rels = [
            _rel("person_01", "holding", "paper_01", 0.0, 5.0),
            _rel("person_01", "holding", "paper_01", 6.0, 10.0),
        ]
        viols = _run(rels)
        r1 = [v for v in viols if v.rule_id == "RULE-1"]
        assert len(r1) == 0

    def test_transfer_outside_window_ignored(self):
        """Gap > transfer_window → no RULE-1."""
        big_gap = TRANSFER_WINDOW_SEC + 5.0
        rels = [
            _rel("person_01", "holding", "book_01", 0.0, 2.0),
            _rel("person_02", "holding", "book_01", 2.0 + big_gap, 15.0),
        ]
        viols = _run(rels, transfer_window=TRANSFER_WINDOW_SEC)
        r1 = [v for v in viols if v.rule_id == "RULE-1"]
        assert len(r1) == 0

    def test_overlapping_hold_counts_as_transfer(self):
        """Overlapping intervals (snatching) is still a transfer."""
        rels = [
            _rel("person_01", "holding", "notebook_01", 0.0, 6.0),
            _rel("person_02", "holding", "notebook_01", 3.0, 9.0),
        ]
        viols = _run(rels)
        r1 = [v for v in viols if v.rule_id == "RULE-1"]
        assert len(r1) == 1

    def test_low_confidence_suppressed(self):
        """Intervals below min_confidence threshold are skipped."""
        rels = [
            _rel("person_01", "holding", "book_01", 0.0, 4.0, score=0.15),
            _rel("person_02", "holding", "book_01", 5.0, 9.0, score=0.15),
        ]
        viols = _run(rels, min_confidence=0.25)
        r1 = [v for v in viols if v.rule_id == "RULE-1"]
        assert len(r1) == 0


# ---------------------------------------------------------------------------
# RULE-2 — Forbidden Electronic Device
# ---------------------------------------------------------------------------

class TestRule2ForbiddenDevice:

    @pytest.mark.parametrize("obj_id", [
        "phone_01", "cell_phone_01", "mobile_01", "smartphone_01",
    ])
    def test_holding_device_detected(self, obj_id):
        rels = [_rel("person_01", "holding", obj_id, 2.0, 7.0)]
        viols = _run(rels)
        r2 = [v for v in viols if v.rule_id == "RULE-2"]
        assert len(r2) == 1
        assert all(v.severity == 3 for v in r2)

    def test_using_device_detected(self):
        rels = [_rel("person_01", "using", "phone_01", 1.0, 5.0)]
        viols = _run(rels)
        r2 = [v for v in viols if v.rule_id == "RULE-2"]
        assert len(r2) == 1

    def test_holding_non_device_ignored(self):
        rels = [_rel("person_01", "holding", "pen_01", 0.0, 5.0)]
        viols = _run(rels)
        r2 = [v for v in viols if v.rule_id == "RULE-2"]
        assert len(r2) == 0

    def test_non_person_holding_device_ignored(self):
        """A 'desk' holding a phone is not a malpractice event."""
        rels = [_rel("desk_01", "holding", "phone_01", 0.0, 5.0)]
        viols = _run(rels)
        r2 = [v for v in viols if v.rule_id == "RULE-2"]
        assert len(r2) == 0


# ---------------------------------------------------------------------------
# RULE-3 — Desk Encroachment
# ---------------------------------------------------------------------------

class TestRule3DeskEncroachment:

    def test_long_near_desk_detected(self):
        dur = ENCROACHMENT_MIN_SEC + 2.0
        rels = [_rel("person_01", "near", "desk_01", 0.0, dur)]
        viols = _run(rels)
        r3 = [v for v in viols if v.rule_id == "RULE-3"]
        assert len(r3) == 1
        assert r3[0].duration >= ENCROACHMENT_MIN_SEC

    def test_short_near_desk_not_flagged(self):
        dur = ENCROACHMENT_MIN_SEC - 0.5
        rels = [_rel("person_01", "near", "desk_01", 0.0, dur)]
        viols = _run(rels)
        r3 = [v for v in viols if v.rule_id == "RULE-3"]
        assert len(r3) == 0

    def test_reaching_toward_desk_detected(self):
        dur = ENCROACHMENT_MIN_SEC + 1.0
        rels = [_rel("person_01", "reaching toward", "desk_01", 0.0, dur)]
        viols = _run(rels)
        r3 = [v for v in viols if v.rule_id == "RULE-3"]
        assert len(r3) == 1

    def test_near_non_desk_not_flagged(self):
        """Being near a chair should not trigger RULE-3."""
        rels = [_rel("person_01", "near", "chair_01", 0.0, 10.0)]
        viols = _run(rels)
        r3 = [v for v in viols if v.rule_id == "RULE-3"]
        assert len(r3) == 0

    def test_severity_scales_with_duration(self):
        short = ENCROACHMENT_MIN_SEC + 1.0
        long  = ENCROACHMENT_MIN_SEC + 5.0
        rels_short = [_rel("person_01", "near", "desk_01", 0.0, short)]
        rels_long  = [_rel("person_01", "near", "desk_01", 0.0, long)]
        v_short = [v for v in _run(rels_short) if v.rule_id == "RULE-3"]
        v_long  = [v for v in _run(rels_long)  if v.rule_id == "RULE-3"]
        assert v_long[0].severity >= v_short[0].severity


# ---------------------------------------------------------------------------
# RULE-4 — Unauthorized Communication
# ---------------------------------------------------------------------------

class TestRule4UnauthorizedCommunication:

    def test_talking_to_between_persons(self):
        rels = [_rel("person_01", "talking to", "person_02", 5.0, 9.0)]
        viols = _run(rels)
        r4 = [v for v in viols if v.rule_id == "RULE-4"]
        assert len(r4) == 1
        assert r4[0].severity == 3

    def test_long_gaze_between_persons_flagged(self):
        dur = GAZE_MIN_SEC + 2.0
        rels = [_rel("person_01", "looking at", "person_02", 0.0, dur)]
        viols = _run(rels)
        r4 = [v for v in viols if v.rule_id == "RULE-4"]
        assert len(r4) == 1

    def test_short_gaze_not_flagged(self):
        dur = GAZE_MIN_SEC - 1.0
        rels = [_rel("person_01", "looking at", "person_02", 0.0, dur)]
        viols = _run(rels)
        r4 = [v for v in viols if v.rule_id == "RULE-4"]
        assert len(r4) == 0

    def test_looking_at_object_not_flagged(self):
        """Looking at a desk is not a communication violation."""
        rels = [_rel("person_01", "looking at", "desk_01", 0.0, 10.0)]
        viols = _run(rels)
        r4 = [v for v in viols if v.rule_id == "RULE-4"]
        assert len(r4) == 0


# ---------------------------------------------------------------------------
# violations_at_time filter
# ---------------------------------------------------------------------------

class TestViolationsAtTime:

    def _make_viols(self):
        rels = [
            _rel("person_01", "talking to", "person_02", 2.0, 5.0),
            _rel("person_01", "holding",    "phone_01",  8.0, 12.0),
        ]
        return run_exam_rules(rels)

    def test_active_at_midpoint(self):
        viols = self._make_viols()
        active = violations_at_time(viols, 3.5)
        rule_ids = {v.rule_id for v in active}
        assert "RULE-4" in rule_ids

    def test_nothing_active_before_start(self):
        viols = self._make_viols()
        active = violations_at_time(viols, 0.5)
        assert len(active) == 0

    def test_lookahead_includes_upcoming(self):
        viols = self._make_viols()
        # At t=1.5, RULE-4 starts at 2.0 → within lookahead=1.0
        active = violations_at_time(viols, 1.5, lookahead=1.0)
        rule_ids = {v.rule_id for v in active}
        assert "RULE-4" in rule_ids


# ---------------------------------------------------------------------------
# Integration smoke test — multiple rules at once
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_multi_rule_scenario(self):
        """
        Scenario: student_01 passes a book to student_02, also uses a phone,
        and student_02 talks to student_03.
        Expect: RULE-1, RULE-2, RULE-4 all fire.
        """
        rels = [
            # Book transfer
            _rel("person_01", "holding", "book_01",   0.0,  5.0),
            _rel("person_02", "holding", "book_01",   6.0, 10.0),
            # Phone usage
            _rel("person_01", "using",   "phone_01",  3.0,  8.0),
            # Communication
            _rel("person_02", "talking to", "person_03", 7.0, 11.0),
            # Desk encroachment (long enough)
            _rel("person_03", "near",    "desk_01",   1.0, 6.0),
        ]
        viols = run_exam_rules(rels)
        rule_ids = {v.rule_id for v in viols}
        assert "RULE-1" in rule_ids, "Expected RULE-1 (material transfer)"
        assert "RULE-2" in rule_ids, "Expected RULE-2 (forbidden device)"
        assert "RULE-4" in rule_ids, "Expected RULE-4 (communication)"
        assert "RULE-3" in rule_ids, "Expected RULE-3 (desk encroachment)"

    def test_empty_events_no_crash(self):
        viols = run_exam_rules([])
        assert viols == []

    def test_output_is_json_serializable(self):
        import json
        rels = [
            _rel("person_01", "holding", "book_01", 0.0, 4.0),
            _rel("person_02", "holding", "book_01", 5.0, 9.0),
        ]
        viols = run_exam_rules(rels)
        payload = json.dumps([v.to_dict() for v in viols])
        reloaded = json.loads(payload)
        assert len(reloaded) == len(viols)
