"""
PDDL Action Extraction from the Temporal Scene Graph.

Implements GraSP-VLA Algorithm 1 (Neau et al., 2026):
  Given a sliding window ζ, find every Functional relation (e.g. person_holding_X)
  in the temporal graph. For each one, look back ζ seconds for Topological
  relations (e.g. X_on_Y) to define preconditions, and forward ζ seconds for
  new Topological relations to define effects. Emit a PDDL action when the
  precondition set and effect set are both non-empty and differ.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Set, Tuple

from temporal.schema import TemporalRelation
from temporal.compat import is_compatible


# --- Layer classification (see GraSP-VLA §III-A, OpenSearch §3.3) ---

FUNCTIONAL = {
    "holding", "riding", "wearing", "carrying",
    "sitting on", "walking on", "looking at",
    "picking up", "putting down", "using",
    "eating", "drinking", "cutting", "stirring", "pouring",
}

TOPOLOGICAL = {
    "on", "inside", "near", "next to",
    "to the left of", "to the right of", "above", "below",
    "in front of", "behind", "under", "over", "beside",
    "resting on", "lying on", "supported by",
}

SYMMETRIC = {"beside", "near", "next to"}

def classify(predicate: str) -> str:
    p = predicate.lower().strip()
    if p in FUNCTIONAL:
        return "functional"
    if p in TOPOLOGICAL:
        return "topological"
    return "other"


ACTION_VERB = {
    "holding": "Hold", "riding": "Ride", "wearing": "Wear",
    "carrying": "Carry", "sitting on": "SitOn", "walking on": "WalkOn",
    "looking at": "LookAt", "picking up": "PickUp",
    "putting down": "PutDown", "using": "Use",
}

def action_name(predicate: str) -> str:
    return ACTION_VERB.get(predicate.lower(), predicate.title().replace(" ", ""))


def pddl_pred(predicate: str) -> str:
    return predicate.lower().replace(" ", "_").replace("-", "_")


# --- Action dataclass ---

@dataclass
class PDDLAction:
    name: str
    subject: str
    object: str
    preconditions: List[Tuple[str, str, str]]
    positive_effects: List[Tuple[str, str, str]]
    negative_effects: List[Tuple[str, str, str]]

    def to_pddl(self) -> str:
        subj_cls = self.subject.rsplit("_", 1)[0]
        obj_cls = self.object.rsplit("_", 1)[0]
        lines = [f";; binding: {self.subject} <-> ?agent, {self.object} <-> ?target"]
        lines.append(f"(:action {self.name}")
        lines.append(f"  :parameters (?agent - {subj_cls} ?target - {obj_cls})")
        if self.preconditions:
            pc = " ".join(f"({pddl_pred(p)} ?agent ?target)" for _, p, _ in self.preconditions)
            lines.append(f"  :precondition (and {pc})")
        else:
            lines.append("  :precondition ()")
        eff = [f"({pddl_pred(p)} ?agent ?target)" for _, p, _ in self.positive_effects]
        eff += [f"(not ({pddl_pred(p)} ?agent ?target))" for _, p, _ in self.negative_effects]
        if eff:
            lines.append(f"  :effect (and {' '.join(eff)})")
        else:
            lines.append("  :effect ()")
        lines.append(")")
        return "\n".join(lines)


# --- Extraction ---

def _topo_relations(
    events: List[TemporalRelation], object_id: str, t_lo: float, t_hi: float,
) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    for ev in events:
        if classify(ev.predicate) != "topological":
            continue
        if ev.end_time < t_lo or ev.start_time > t_hi:
            continue
        if ev.object_id == object_id or ev.subject_id == object_id:
            out.add((ev.subject_id, ev.predicate, ev.object_id))
    return out


def extract_actions(
    events: List[TemporalRelation],
    window_sec: float = 5.0,
    min_confidence: float = 0.5,
) -> List[PDDLAction]:
    filtered = []
    for e in events:
        conf = getattr(e, "confidence", 0.0) or e.mean_score
        if conf < min_confidence:
            continue
        if not is_compatible(e.subject_id, e.predicate):
            continue
        filtered.append(e)
    events = sorted(filtered, key=lambda e: e.start_time)

    actions: List[PDDLAction] = []
    for ev in events:
        if classify(ev.predicate) != "functional":
            continue
        t = ev.start_time
        patient = ev.object_id
        agent = ev.subject_id

        pre = _topo_relations(events, patient, t - window_sec, t)
        eff_pos = _topo_relations(events, patient, t, t + window_sec)

        pre = {r for r in pre if r[1] not in SYMMETRIC}
        eff_pos = {r for r in eff_pos if r[1] not in SYMMETRIC}
        pre_set = set(pre)
        eff_set = set(eff_pos)
        new_eff = eff_set - pre_set
        removed = pre_set - eff_set

        if not pre or (not new_eff and not removed):
            continue

        actions.append(PDDLAction(
            name=action_name(ev.predicate),
            subject=agent, object=patient,
            preconditions=sorted(pre),
            positive_effects=sorted(new_eff),
            negative_effects=sorted(removed),
        ))
    return actions


# --- CLI ---

def main():
    ap = argparse.ArgumentParser(description="Extract PDDL actions from a GodsEye run dir")
    ap.add_argument("run_dir", help="outputs/<run>")
    ap.add_argument("--window", type=float, default=5.0, help="sliding window zeta (seconds)")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    raw = json.loads((run_dir / "events.json").read_text())
    events = [TemporalRelation(**r) for r in raw]
    print(f"loaded {len(events)} temporal relations")

    actions = extract_actions(events, args.window, args.min_confidence)
    print(f"extracted {len(actions)} PDDL actions")

    (run_dir / "actions.pddl").write_text(
        ";; Auto-extracted by GodsEye (GraSP-VLA Algorithm 1)\n"
        f";; {len(actions)} actions\n\n"
        + "\n\n".join(a.to_pddl() for a in actions) + "\n"
    )
    (run_dir / "actions.json").write_text(json.dumps([asdict(a) for a in actions], indent=2))

    print(f"wrote {run_dir / 'actions.pddl'}")
    print(f"wrote {run_dir / 'actions.json'}")
    for a in actions:
        print(f"  {a.name}({a.subject}, {a.object})  "
              f"pre={len(a.preconditions)} eff+={len(a.positive_effects)} "
              f"eff-={len(a.negative_effects)}")


if __name__ == "__main__":
    main()
