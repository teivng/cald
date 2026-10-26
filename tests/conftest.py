import os

import pytest


def ckpt_path():
    """Local official TabICLv2 checkpoint, or None (tests then download/skip)."""
    p = os.environ.get("CALD_TEST_CKPT")
    return p if p and os.path.isfile(p) else None


@pytest.fixture(scope="session")
def nano_adapter():
    """A CPU NanoTabICLv2Adapter, shared across the session (110 MB weights)."""
    pytest.importorskip("torch")
    from cald.adapters.nanotabiclv2 import NanoTabICLv2Adapter

    try:
        return NanoTabICLv2Adapter(device="cpu", checkpoint_path=ckpt_path())
    except Exception as e:  # no local ckpt and no network
        pytest.skip(f"TabICLv2 checkpoint unavailable: {e}")
