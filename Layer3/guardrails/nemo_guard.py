"""
Optional NeMo Guardrails integration.

When `nemoguardrails` is installed (pip install nemoguardrails), this module
wraps the colang rails defined in nemo_config/ and exposes a check() that
mirrors InputResult so it can be used as a drop-in layer alongside the
custom regex-based guards.

If nemoguardrails is not installed, get_nemo_rails() returns None and all
callers should fall back to the custom InputGuardrail / OutputGuardrail.

Install note: nemoguardrails requires the `annoy` C++ extension which must be
compiled. On Linux/Docker it installs cleanly via pip. On Windows you need
Microsoft C++ Build Tools.
"""

from __future__ import annotations
from pathlib import Path

_NEMO_CONFIG_DIR = str(Path(__file__).parent / "nemo_config")
_rails = None
_tried = False


def get_nemo_rails():
    """Return a compiled RailsConfig, or None if nemoguardrails is not available."""
    global _rails, _tried
    if _tried:
        return _rails
    _tried = True
    try:
        from nemoguardrails import RailsConfig, LLMRails
        cfg    = RailsConfig.from_path(_NEMO_CONFIG_DIR)
        _rails = LLMRails(cfg)
    except ImportError:
        _rails = None
    except Exception:
        _rails = None
    return _rails


def nemo_check_input(text: str) -> bool:
    """
    Returns True if NeMo rails allow the input, False if blocked.
    Falls back to True (allow) if nemoguardrails is not installed.
    """
    rails = get_nemo_rails()
    if rails is None:
        return True
    try:
        response = rails.generate(messages=[{"role": "user", "content": text}])
        content  = response.get("content", "")
        blocked_markers = ["נחסמה", "נחסם", "blocked", "refuse"]
        return not any(m in content.lower() for m in blocked_markers)
    except Exception:
        return True
