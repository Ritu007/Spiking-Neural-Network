"""
actions.py — OS Command Dispatcher for SpikeVoice OS
=====================================================
Maps detected keyword labels to OS-level actions.
Enforces a 2-second cooldown to prevent double-triggering.
"""

import os
import time
import pyautogui

# Disable pyautogui fail-safe pause for low-latency triggering
pyautogui.PAUSE = 0

# ── Cooldown state ────────────────────────────────────────────────────────────
_last_trigger_time: float = 0.0
COOLDOWN_SECONDS          = 2.0


# ── Action map ────────────────────────────────────────────────────────────────
ACTIONS: dict = {
    "mute":  lambda: pyautogui.press("volumemute"),
    "up":    lambda: [pyautogui.press("volumeup")   for _ in range(3)],
    "down":  lambda: [pyautogui.press("volumedown") for _ in range(3)],
    "stop":  lambda: pyautogui.press("playpause"),
    "bed":   lambda: None,   # placeholder — maps to a user-defined shortcut
    "sleep": lambda: os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0"),
}


def dispatch(label: str) -> bool:
    """
    Execute the OS action for the given keyword label, respecting the cooldown.

    Args:
        label: Detected keyword string (e.g. "mute", "up", "silence")

    Returns:
        True if the action was executed, False if cooled down or label unknown.
    """
    global _last_trigger_time

    if label == "silence" or label not in ACTIONS:
        return False

    now = time.monotonic()
    if (now - _last_trigger_time) < COOLDOWN_SECONDS:
        return False   # still in cooldown window

    _last_trigger_time = now
    action = ACTIONS[label]
    try:
        action()
        print(f"⚡ Action dispatched: [{label.upper()}]")
    except Exception as e:
        print(f"⚠️  Action failed for '{label}': {e}")

    return True


def time_since_last_action() -> float:
    """Returns seconds elapsed since the last triggered action."""
    return time.monotonic() - _last_trigger_time


def is_in_cooldown() -> bool:
    return (time.monotonic() - _last_trigger_time) < COOLDOWN_SECONDS
