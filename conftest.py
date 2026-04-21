# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal pytest fixtures for tt-Gaze-LLE.

Provides a single-device ``device`` fixture so the gaze_lle tests can be run
without the tt-metal monorepo's own conftest.py. The underlying device is
selected with ``--device-id N`` on the pytest CLI or with the environment
variable ``GAZE_LLE_DEVICE`` (default 0). Set ``TT_VISIBLE_DEVICES`` to
constrain which PCIe chips ttnn enumerates at all.
"""

from __future__ import annotations

import gc
import os

import pytest


@pytest.fixture(autouse=True)
def _gc_between_tests():
    gc.collect()


@pytest.fixture(scope="session")
def device():
    import ttnn

    device_id = int(os.environ.get("GAZE_LLE_DEVICE", "0"))
    dev = ttnn.open_device(device_id=device_id)
    yield dev
    ttnn.close_device(dev)
