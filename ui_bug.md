The symptom was "nothing renders when I move the slider." The chain of causes, in the order they actually blocked you:

The one real bug
The dropdown's .change event never fired, so load_run was never called, so STATE.manifest stayed None, so update_frame hit its early-return guard and returned (None, None) on every slider move.

We proved this directly with the instrumented run. The terminal showed:

text
[slider] called with 2 (type=int)
…and no [dropdown] line above it. If load_run had run, we'd have seen [dropdown] 'whatsapp_test' first. We didn't. So the state was empty, and everything downstream of it correctly rendered nothing.

Why didn't .change fire? Because your outputs/ directory had exactly one run in it. The dropdown's only option was whatsapp_test, and Gradio had it pre-selected. Re-selecting the same value doesn't count as a change, so .change never fired. This is a known Gradio behavior with single-option dropdowns.

The chain of contributing bugs
Each of these was real, and each one masked the next:

No if __name__ == "__main__": block. Running python temporal/viz/dashboard_ui.py exited silently because the file only defined functions. So the first time you tried to run it, it just… stopped. Nothing was broken yet; there was simply no demo.launch() call.

load_run returned gr.Slider(...) instead of gr.update(...). Returning a component as an output value is not how you update an already-rendered component. Even if the dropdown had fired, this would have aborted the callback and blanked all four outputs.

refresh_btn.click(lambda: gr.Dropdown(choices=list_runs()), None, run_dropdown) had the same problem — returning a bare component instead of gr.update(choices=...).

No auto-load on startup. Even with a working dropdown, you had to explicitly pick a run. If you loaded the page and dragged the slider without touching the dropdown, STATE was empty.

What the fix changed
The patch that finally worked did three things:

Added a Load Run button whose .click event explicitly calls load_run. Clicking is unambiguous — no dependence on whether the dropdown value "changed."

Added demo.load(...) on startup, so the default run (first entry in outputs/) is loaded automatically before you touch anything. STATE is never empty.

Fixed the component-update returns (gr.update(...) for slider and refresh), and clamped frame_idx in case Gradio's slider max didn't update from 100 to 33.

Any one of the four on its own would have produced the same "nothing renders" symptom. The dropdown-with-one-option one is what actually triggered it in your session; the others would have broken it the moment you added a second run to outputs/.

