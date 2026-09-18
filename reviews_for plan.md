Read it end to end. The plan is well-organized and shows they've actually read the RelateAnything docs — the dependency argument is sound, the sprint structure is reasonable, and the module breakdown is clean. But there are **three technical errors** that will cost real time if not caught now, plus several gaps.

## The three that matter

### 1. The Kalman filter dismissal is wrong, and it's the central design decision

They wrote:

> **Why not a Kalman filter (like RelateAnything's `render_video.py`)?** Their Kalman filter smooths *scores* for display purposes. We need to smooth *discrete states* (predicate labels) for semantic reasoning. Majority voting is simpler, interpretable, and directly produces the state transitions we need.

This misreads what the Kalman filter in `render_video.py` does. Its job is not smoothing for display — it's distinguishing **"measured weak" from "not measurable"**. When an endpoint goes undetected for a frame, the pair is not "score dropped to zero" — it's "no observation this frame." The Kalman filter does a predict-only step, belief stays put, and the edge survives the occlusion.

Majority voting over a 5-frame window has no way to represent that distinction. A person briefly occluded for 2 of 5 frames will get outvoted by whatever noise filled the gap. The README quotes the measured consequence: the Kalman approach "cut blinks by 39% and raised the mean run of an edge from 21 to 32 frames." A 5-frame majority vote will be worse than the per-frame output on any clip with occlusion.

The fix isn't to adopt their code — it's to fix the reasoning. Either:
- **Keep the Kalman filter**, on calibrated log-odds, with the predict-only step for missing endpoints. This is the mechanically correct answer and it's not a lot of code (`pykalman` or a 30-line manual implementation). The "discrete states" argument doesn't hold because you can discretize *after* filtering.
- **Or do proper state tracking** with explicit `unknown` states — three-valued (held / not-held / no-evidence), with transitions gated on the third. This is more work and loses the score smoothing.

The current plan's majority vote is the one approach that will visibly fail on their own test videos.

### 2. The curated vocabulary isn't checked against the model

The list includes `"outside"`, which is **not** in `deploy/pipeline._default_predicates()` or `relsgg/vocabulary.py::DEFAULT_PREDICATES`. Adding strings the model wasn't trained against is allowed — the text student will encode them — but the quality is unpredictable, and `outside` lands in a sparsely-populated region of predicate space.

More importantly, the list mixes predicates the model handles well (`holding`, `sitting on`, `on`, `in front of`) with ones it's weak on (`approaching`, `moving away from` — both rare in RA-4M). If a pattern like `APPROACH` depends on `approaching` firing, the pattern will almost never match.

They should:
- Start from `_default_predicates()` (the 22 the pipeline ships with), plus `relsgg/vocabulary.py::DEFAULT_PREDICATES` (35 the released models were trained against). Anything else is a stretch.
- For each predicate in the curated list, run a quick test: does it produce sane output on the demo images? The `deploy/gradio_app_styled.py` you've been hacking on is a ready-made harness for exactly this.


Gradio-based temporal inspection dashboard

The dashboard uses a timestamp/frame slider rather than requiring continuous synchronized video playback. Processing is performed offline by process_video.py; the dashboard reads precomputed frames, relations, scene graphs, and events from a run directory. Changing the timestamp updates the rendered frame, current relations, scene-graph state, and timeline marker.

## What's good

- The dependency-not-plagiarism argument is correct. The PyTorch analogy is apt, and the table they built is fine even if it feels defensive.
- ByteTrack over DeepSORT is the right call for indoor scenes with few objects.
- Rule-based event patterns over an LLM is the right call for a project with 15 event types and no training data.
- Allen's interval algebra for the event graph is standard and correct.
- The 5-sprint consolidation from 10 phases is sensible.

## What's missing

**Re-identification across track breaks.** ByteTrack will lose and re-acquire objects. `registry.py` is mentioned but doesn't say how it handles this. If `person_01` becomes `person_03` at frame 240, you get two disjoint state histories for the same physical person. Need either appearance-based re-ID (adds a CNN) or IoU-based merging over a short window.

**Frame-level streaming.** At 2 Hz × 10 min × 1280px, they're decoding 1,200 frames. If each frame is 3 MB, that's 3.6 GB of transient data. They need to say whether they hold all frames in memory, stream from disk, or use a sliding window. This matters on the user's 4 GB machine.

**VRAM on the user's machine.** They wrote "ideally ≥8 GB VRAM." The user has 4 GB. RelateAnything ViT-S/16+ (53M) + YOLOv8m + ByteTrack + Streamlit + PyTorch CUDA context is easily 3–4 GB at load. This will OOM. The plan should either:
- Recommend the ViT-S/16 checkpoint (46M), not `vits16plus`.
- Suggest CPU for the relation model, GPU for the detector.
- Or admit that 4 GB is very tight and suggest batch-only processing (no live preview).

**The evaluation section is a stub.** Four experiments listed, but no ground truth source. Where do labeled events come from? Manual annotation of their own test videos? That's fine but needs to be spelled out — it's probably a full sprint of its own. Otherwise the "P/R/F1" numbers have no meaning.

**Score semantics.** The plan uses `confidence` as if it's a probability. It's not, unless calibration is loaded. `from_checkpoint` picks up `calibration.json` when present — the released models ship one — but the plan doesn't mention this at all. If they build a temporal state machine on `if score > 0.3` without understanding that score is `sigmoid(a·(pred + pair) + b)`, the thresholds will be wrong.

**The PICK_UP pattern needs an intermediate the model may not emit.** It requires `near → touching → holding`. On an actual pick-up, the model often jumps straight from `near` to `holding` without an intermediate `touching` frame. The `alt_sequence` handles this — but `near` and `touching` both need to actually fire, and `touching` is a predicate the model is weak on. Worth testing the pattern against real video before committing to it.

Fair pushback. I overcorrected. **Gradio is fine for this** — and given what you've already built, it's the right call. Let me correct that.


## What you actually need

Three panels that stay consistent:

1. A way to pick a time `t`
2. A frame image at `t` with boxes + relation edges drawn
3. A scene graph and timeline for the whole video, with the current time marked

**None of that requires video playback.** It requires a scrubber. And a scrubber is a slider, which Gradio does perfectly.

```
┌──────────────────────┬──────────────────────┐
│  [ slider: 0 ──●──── 10.4s ]                 │
│  ┌────────────────┐  │  SCENE GRAPH          │
│  │  frame @ 10.4s │  │  (gr.Plot, Plotly)    │
│  │  with overlays │  │                       │
│  └────────────────┘  │                       │
│                      ├───────────────────────┤
│  RELATIONS @ 10.4s   │  TIMELINE             │
│  (relation cards)    │  (gr.Plot, Gantt)     │
└──────────────────────┴──────────────────────┘
```

Move the slider → the three panels update. That's a Gradio callback. You already have the styled panel infrastructure and the overlay renderer. This is a few hundred lines on top of what exists.

## Why this is better than video playback for your project

- **Exact frame control.** For debugging and for the eval, you want to land on `t=10.4` and inspect. Video playback is imprecise.
- **Deterministic.** No browser codec variance, no autoplay policies, no "why did the video stop?"
- **Cheap.** You already have `render(res, ...)` from the styled app. You feed it the frame at `t`.
- **Pre-computable.** Run the pipeline once over the whole video, save frames + relations to disk, then the dashboard just reads from that store. No live inference in the UI at all.

That last point is the design that actually works. Offline process, online browse. The Streamlit plan implies live inference per interaction, which is slow and fragile. Your Gradio version should:

1. `scripts/process_video.py` runs the whole pipeline, writes `outputs/<video>/frames/*.jpg`, `relations.jsonl`, `scene_graph.json`, `events.json`
2. `run_dashboard.py` loads that directory and only does rendering.

## What you lose vs Dash

- **No autoplay video.** Users scrub instead of watching. For a research demo, usually fine.
- **Slightly less polished multi-panel layout.** Gradio's grid is `gr.Row`/`gr.Column`, less flexible than Dash's flexbox. You've already proven you can CSS around it in `gradio_app_styled.py`.

If the supervisor specifically wants "it plays like YouTube with overlays," you'd need the Dash route or a custom HTML canvas. If they want "I can inspect the temporal graph," Gradio wins.

## What this means for the plan

Tell your friend to change the plan:

1. **Sprint 3 dashboard**: Gradio, not Streamlit. Reuse the styled app. Slider-driven, not video-driven.
2. **Architecture**: `process_video.py` writes a **run directory**; the dashboard reads it. Two-phase, not live.
3. **Data format**: a JSON manifest per video, plus a frames directory. The dashboard is a viewer.

Concrete module changes:

| Plan says | Should say |
|---|---|
| `viz/dashboard.py` (Streamlit) | `viz/dashboard.py` (Gradio, extends the styled app) |
| Live inference in dashboard | Pre-computed run dir, read-only viewer |
| "Video player with bbox overlay" | "Frame slider + overlay image" |
| `requirements.txt`: streamlit, plotly | gradio, plotly |

Plotly still works — `gr.Plot` wraps it. So the timeline and scene graph panels are unchanged.

## What this doesn't fix

The Kalman filter issue is still the biggest technical error in the plan. Slider vs video is a UI choice; the temporal smoothing is the correctness question. Don't let the framework discussion distract from that.

But yes — **stay in Gradio**. You've paid the framework tax already. Don't pay it twice.

## What I'd tell them to change

1. **Fix the temporal smoothing section.** Kalman filter on log-odds, with predict-only for missing endpoints. This is the single most important change — it's the difference between a working demo and one that blinks.
2. **Check the curated vocabulary** against `_default_predicates()` and drop anything the model handles poorly.
3. **Downgrade the dashboard scope**, or commit to Dash instead of Streamlit and accept the extra setup.
5. **Flesh out the evaluation ground truth** — where do labeled events come from?
6. **Add re-ID across track breaks** to the registry design.

The rest is solid. It's a good plan with one design error that happens to be the one the project is actually about.
