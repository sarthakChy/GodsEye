#!/usr/bin/env python3
"""
GodsEye demo entry point.

Usage:
    python demo.py dashboard          # launch the Gradio dashboard
    python demo.py script <run_dir>   # print a single run's summary
    python demo.py script-all         # print all demo runs' summaries
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEMO_RUNS = ["demo_chopping", "demo_pouring", "demo_combined"]
OUTPUTS = Path("outputs")


# --- ANSI colours (safe fallback if not a tty) ---
def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""

BOLD   = _c("\033[1m")
DIM    = _c("\033[2m")
GREEN  = _c("\033[32m")
YELLOW = _c("\033[33m")
BLUE   = _c("\033[34m")
RESET  = _c("\033[0m")


def summarise_run(run_dir: Path) -> dict:
    events_path = run_dir / "events.json"
    actions_path = run_dir / "actions.json"
    if not events_path.exists():
        return {"error": f"no events.json in {run_dir}"}

    events = json.loads(events_path.read_text())
    actions = json.loads(actions_path.read_text()) if actions_path.exists() else []

    preds = {}
    for e in events:
        preds[e["predicate"]] = preds.get(e["predicate"], 0) + 1

    dur_total = sum(e["end_time"] - e["start_time"] for e in events)

    return {
        "run": run_dir.name,
        "events": len(events),
        "total_duration_sec": round(dur_total, 1),
        "predicates": preds,
        "actions": actions,
    }


def print_summary(s: dict) -> None:
    if "error" in s:
        print(f"{YELLOW}  {s['error']}{RESET}")
        return

    print(f"{BOLD}━━━ {s['run']} ━━━{RESET}")
    print(f"  temporal relations : {BOLD}{s['events']}{RESET}")
    print(f"  total tracked time : {s['total_duration_sec']}s")

    if s["predicates"]:
        print(f"  {DIM}by predicate:{RESET}")
        for p, c in sorted(s["predicates"].items(), key=lambda x: -x[1])[:6]:
            print(f"    {p:<20} {c}")

    n = len(s["actions"])
    print(f"  {GREEN}PDDL actions{RESET}      : {BOLD}{n}{RESET}")
    for a in s["actions"][:6]:
        name = a["name"]
        subj = a["subject"]
        obj  = a["object"]
        pre  = len(a["preconditions"])
        effp = len(a["positive_effects"])
        effn = len(a["negative_effects"])
        print(f"    {BLUE}{name:<8}{RESET}({subj}, {obj})  "
              f"pre={pre} eff+={effp} eff-={effn}")
    if n > 6:
        print(f"    {DIM}... and {n - 6} more{RESET}")
    print()


# --- Modes ---

def mode_dashboard() -> None:
    print(f"{BOLD}Launching GodsEye dashboard…{RESET}")
    print(f"  Runs available: {', '.join(DEMO_RUNS)}")
    print(f"  Open the URL below, pick a run from the dropdown, drag the slider.")
    print()
    subprocess.run(
        [sys.executable, "-m", "temporal.viz.dashboard_ui"],
        check=False,
    )


def mode_script(run: str) -> None:
    run_dir = Path(run) if Path(run).exists() else OUTPUTS / run
    print_summary(summarise_run(run_dir))


def mode_script_all() -> None:
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  GodsEye — Temporal Scene Graph + PDDL Extraction{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")
    for r in DEMO_RUNS:
        d = OUTPUTS / r
        if d.exists():
            print_summary(summarise_run(d))
    print(f"{BOLD}{'=' * 60}{RESET}")


# --- CLI ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dashboard", "script", "script-all"])
    ap.add_argument("run", nargs="?", help="run directory (for 'script' mode)")
    args = ap.parse_args()

    if args.mode == "dashboard":
        mode_dashboard()
    elif args.mode == "script":
        if not args.run:
            ap.error("'script' mode requires a run directory")
        mode_script(args.run)
    elif args.mode == "script-all":
        mode_script_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
