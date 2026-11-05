#!/usr/bin/env python3
"""
test_protocol.py — Protocol binary layout and serialization tests (Section 11.1)
"""

import sys
import os
import struct

# Add inference directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../inference")))

from protocol import (
    MSG_WINDOW,
    MSG_DONE,
    MSG_RESULT,
    PROTOCOL_VERSION,
    WINDOW_MSG_SIZE,
    DONE_MSG_SIZE,
    RESULT_HEADER_SIZE,
    FRAME_OUT_SIZE,
    unpack_window_msg,
    pack_done_msg,
    pack_result,
)


def test_struct_sizes():
    # As per Section 6 specification:
    # WindowMsg = 64 bytes
    # DoneMsg = 40 bytes
    # ResultHeader = 64 bytes
    # FrameOut = 8 bytes
    assert WINDOW_MSG_SIZE == 64, f"WindowMsg size {WINDOW_MSG_SIZE} != 64"
    assert DONE_MSG_SIZE == 40, f"DoneMsg size {DONE_MSG_SIZE} != 40"
    assert RESULT_HEADER_SIZE == 64, f"ResultHeader size {RESULT_HEADER_SIZE} != 64"
    assert FRAME_OUT_SIZE == 8, f"FrameOut size {FRAME_OUT_SIZE} != 8"
    print("[PASS] Protocol binary struct sizes match specification.")


def test_done_msg_roundtrip():
    b = pack_done_msg(
        window_id=42,
        t_infer_begin_ns=1000000000,
        t_infer_end_ns=1050000000,
        status=0,
    )
    assert len(b) == 40
    fields = struct.unpack("<IIQqqII", b)
    assert fields[0] == MSG_DONE
    assert fields[1] == PROTOCOL_VERSION
    assert fields[2] == 42
    assert fields[3] == 1000000000
    assert fields[4] == 1050000000
    assert fields[5] == 0
    print("[PASS] DoneMsg serialization round-trip.")


def test_result_roundtrip():
    frames = [(1, 0.95), (2, 0.88), (41, 0.99)]
    b = pack_result(
        window_id=10,
        start_sample=3200,
        t_start_ns=500000,
        t_infer_begin_ns=600000,
        t_infer_end_ns=650000,
        n_frames=3,
        commit_begin=1,
        commit_end=2,
        flags=0,
        frames_array=frames,
    )
    expected_len = RESULT_HEADER_SIZE + 3 * FRAME_OUT_SIZE
    assert len(b) == expected_len

    hdr = struct.unpack_from("<IIQQqqqIIII", b, 0)
    assert hdr[0] == MSG_RESULT
    assert hdr[1] == PROTOCOL_VERSION
    assert hdr[2] == 10
    assert hdr[3] == 3200
    assert hdr[7] == 3 # n_frames
    assert hdr[8] == 1 # commit_begin
    assert hdr[9] == 2 # commit_end

    f1 = struct.unpack_from("<HHf", b, RESULT_HEADER_SIZE)
    assert f1[0] == 1
    assert abs(f1[2] - 0.95) < 1e-5
    print("[PASS] ResultHeader + FrameOut array round-trip.")


if __name__ == "__main__":
    test_struct_sizes()
    test_done_msg_roundtrip()
    test_result_roundtrip()
    print("[ALL PASS] Python protocol tests passed.")

