import os
from unittest import mock
import pytest

def test_antigravity_hooks_windows_extension():
    """
    Test that the antigravity hooks use the correct extension (.ps1 on Windows, .sh on POSIX).
    We test this by mocking platform.system and reloading the module.
    """
    import importlib
    from iai_mcp.cli import _antigravity_hooks
    
    # Test POSIX behavior
    with mock.patch("platform.system", return_value="Linux"):
        importlib.reload(_antigravity_hooks)
        assert _antigravity_hooks._EXT == ".sh"
        
        # Verify wiring picks up .sh
        for event, marker, timeout in _antigravity_hooks._EVENT_WIRING:
            assert marker.endswith(".sh")

    # Test Windows behavior
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        assert _antigravity_hooks._EXT == ".ps1"
        
        # Verify wiring picks up .ps1
        for event, marker, timeout in _antigravity_hooks._EVENT_WIRING:
            assert marker.endswith(".ps1")
