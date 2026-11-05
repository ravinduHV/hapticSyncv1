#!/usr/bin/env python3
"""
protocol.py — Python mirror of common/protocol.h (Section 6)
All structs are packed little-endian matching C++ structs byte-for-byte.
"""

import struct

MSG_WINDOW = 1
MSG_DONE   = 2
MSG_RESULT = 3

PROTOCOL_VERSION = 1

# WindowMsg format: type(I), version(I), window_id(Q), slot(I), n_samples(I),
# start_sample(Q), t_start_ns(q), t_sent_ns(q), n_frames(I), commit_begin(I),
# commit_end(I), flags(I)
WINDOW_MSG_FMT = "<IIQIIQqqIIII"
WINDOW_MSG_SIZE = struct.calcsize(WINDOW_MSG_FMT)  # 64 bytes

# DoneMsg format: type(I), version(I), window_id(Q), t_infer_begin(q),
# t_infer_end(q), status(I), pad(I)
DONE_MSG_FMT = "<IIQqqII"
DONE_MSG_SIZE = struct.calcsize(DONE_MSG_FMT)      # 40 bytes

# ResultHeader format: type(I), version(I), window_id(Q), start_sample(Q),
# t_start_ns(q), t_infer_begin(q), t_infer_end(q), n_frames(I),
# commit_begin(I), commit_end(I), flags(I)
RESULT_HEADER_FMT = "<IIQQqqqIIII"
RESULT_HEADER_SIZE = struct.calcsize(RESULT_HEADER_FMT)  # 64 bytes

# FrameOut format: label(H), pad(H), conf(f)
FRAME_OUT_FMT = "<HHf"
FRAME_OUT_SIZE = struct.calcsize(FRAME_OUT_FMT)    # 8 bytes


def unpack_window_msg(data: bytes) -> dict:
    """Unpack a WindowMsg received from the assembler."""
    if len(data) < WINDOW_MSG_SIZE:
        raise ValueError(f"WindowMsg too short: {len(data)} < {WINDOW_MSG_SIZE}")

    fields = struct.unpack_from(WINDOW_MSG_FMT, data)
    return {
        "type": fields[0],
        "version": fields[1],
        "window_id": fields[2],
        "slot": fields[3],
        "n_samples": fields[4],
        "start_sample": fields[5],
        "t_start_ns": fields[6],
        "t_sent_ns": fields[7],
        "n_frames": fields[8],
        "commit_begin": fields[9],
        "commit_end": fields[10],
        "flags": fields[11],
    }


def pack_done_msg(window_id: int, t_infer_begin_ns: int, t_infer_end_ns: int, status: int = 0) -> bytes:
    """Pack a DoneMsg to send to the assembler."""
    return struct.pack(
        DONE_MSG_FMT,
        MSG_DONE,
        PROTOCOL_VERSION,
        window_id,
        t_infer_begin_ns,
        t_infer_end_ns,
        status,
        0,  # pad
    )


def pack_result(
    window_id: int,
    start_sample: int,
    t_start_ns: int,
    t_infer_begin_ns: int,
    t_infer_end_ns: int,
    n_frames: int,
    commit_begin: int,
    commit_end: int,
    flags: int,
    frames_array: list,  # list of tuples: (label: int, conf: float)
) -> bytes:
    """Pack ResultHeader + n_frames * FrameOut to send to post-processing."""
    hdr = struct.pack(
        RESULT_HEADER_FMT,
        MSG_RESULT,
        PROTOCOL_VERSION,
        window_id,
        start_sample,
        t_start_ns,
        t_infer_begin_ns,
        t_infer_end_ns,
        n_frames,
        commit_begin,
        commit_end,
        flags,
    )
    payload = bytearray(hdr)
    for label, conf in frames_array:
        payload.extend(struct.pack(FRAME_OUT_FMT, label, 0, conf))
    return bytes(payload)
