# lightcraft/main_view.py

"""
LightCraft — main view / runner
- CLI mode (default): prints a launch banner.
- Pyto UI mode (optional): if pyto_ui is available, shows a simple view
  with a button and console log to prove UI wiring works.
"""

from __future__ import annotations
import os
import sys
from typing import Optional

# --- Optional Pyto UI integration (iOS) ---
try:
    import pyto_ui as ui  # type: ignore
except Exception:
    ui = None  # Not in Pyto / UI not available


BANNER = "Launching LightCraft Interface... 🜂"


def _run_cli() -> int:
    """Fallback/standard CLI behavior."""
    print(BANNER)
    return 0


def _run_pyto_ui() -> int:
    """If Pyto is available, present a minimal UI window."""
    if ui is None:
        return _run_cli()

    class MainView(ui.View):

        def ib_init(self):
            # Called when created from .pytoui; safe to leave empty.
            pass

        def did_appear(self):
            # Called when the view is shown.
            print(BANNER)

        @ui.ib_action
        def button_pressed(self, sender: ui.Button):
            print("🔥 LightCraft ping — ritual circuit complete.")
            self.close()

    # Build a simple view programmatically (works even without .pytoui)
    view = MainView()
    view.background_color = ui.COLOR_SYSTEM_BACKGROUND
    view.title = "LightCraft"

    label = ui.Label()
    label.text = "LightCraft"
    label.text_alignment = ui.TextAlignment.CENTER
    label.font = ui.Font.system_font_of_size(22)
    label.frame = (0, 40, 0, 30)  # x,y,width,height; width auto by autoresizing
    view.add_subview(label)

    button = ui.Button()
    button.title = "Begin Ritual"
    button.frame = (0, 100, 0, 44)
    button.autoresizing = ui.AutoResizing.FLEXIBLE_WIDTH
    # Wire to the same action as Interface Builder would:
    button.action = view.button_pressed
    view.add_subview(button)

    ui.show_view(view, ui.PRESENTATION_MODE_SHEET)
    return 0


def run(argv: Optional[list[str]] = None) -> int:
    """
    Entrypoint used by console_scripts (lightcraft).
    Selects UI mode if Pyto is available, else CLI mode.
    """
    _ = argv or sys.argv[1:]
    # Allow forcing modes via env var if needed:
    mode = os.getenv("LIGHTCRAFT_MODE", "").lower()
    if mode == "cli":
        return _run_cli()
    if mode == "ui":
        return _run_pyto_ui()

    # Auto-detect: prefer Pyto UI if available
    if ui is not None:
        return _run_pyto_ui()
    return _run_cli()


if __name__ == "__main__":
    raise SystemExit(run())
