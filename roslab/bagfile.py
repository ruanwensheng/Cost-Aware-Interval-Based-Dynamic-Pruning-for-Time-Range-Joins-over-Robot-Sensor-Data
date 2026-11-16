"""Minimal ROS 1 bag (format 2.0) reader that never loads the whole file.

Layout of a bag file:
    b"#ROSBAG V2.0\\n"
    BAG HEADER (op 0x03)        index_pos, conn_count, chunk_count
    CHUNK (op 0x05)             compressed block of CONNECTION + MESSAGE DATA records
    INDEX DATA (op 0x04) x n    one per connection in that chunk: (time, offset) per message
    ... more CHUNK + INDEX DATA ...
    CONNECTION (op 0x07) x c    at index_pos: the index section
    CHUNK INFO (op 0x06) x k    chunk_pos, start/end time, message count per connection

`BagFile.open()` reads only the bag header and the index section at the end.
Chunks are then read and decompressed one at a time (`read_chunk`).
"""
from __future__ import annotations

import bz2
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

OP_MSG_DATA, OP_BAG_HEADER, OP_INDEX_DATA = 0x02, 0x03, 0x04
OP_CHUNK, OP_CHUNK_INFO, OP_CONNECTION = 0x05, 0x06, 0x07
MAGIC = b"#ROSBAG V2.0\n"

INDEX_ENTRY = np.dtype([("sec", "<u4"), ("nsec", "<u4"), ("offset", "<u4")])


class BagError(Exception):
    pass


def parse_header(buf) -> dict[str, bytes]:
    """Header = repeated (field_len uint32, b'name=value')."""
    fields, pos, n_buf = {}, 0, len(buf)
    while pos < n_buf:
        n = struct.unpack_from("<I", buf, pos)[0]
        item = bytes(buf[pos + 4:pos + 4 + n])
        name, _, value = item.partition(b"=")
        fields[name.decode()] = value
        pos += 4 + n
    return fields


def _u32(b: bytes) -> int:
    return struct.unpack("<I", b)[0]


def _u64(b: bytes) -> int:
    return struct.unpack("<Q", b)[0]


def _time_ns(b: bytes) -> int:
    s, ns = struct.unpack("<II", b)
    return s * 1_000_000_000 + ns


@dataclass
class Connection:
    conn: int
    topic: str
    msgtype: str          # ROS 1 style, e.g. "sensor_msgs/Imu"
    md5sum: str
    msgdef: str

    @property
    def typestore_name(self) -> str:
        """rosbags typestore name, e.g. "sensor_msgs/msg/Imu"."""
        pkg, _, name = self.msgtype.partition("/")
        return f"{pkg}/msg/{name}"


@dataclass
class ChunkInfo:
    idx: int              # 0-based chunk number, in file order
    chunk_pos: int        # file offset of the CHUNK record
    start_ns: int
    end_ns: int
    counts: dict[int, int] = field(default_factory=dict)   # conn -> message count

    @property
    def n_messages(self) -> int:
        return sum(self.counts.values())


@dataclass
class ChunkHeader:
    compression: str
    uncompressed_size: int
    data_pos: int         # file offset of the compressed bytes
    data_len: int         # compressed size in bytes

    @property
    def end_pos(self) -> int:
        """File offset right after the chunk data (the INDEX DATA records start here)."""
        return self.data_pos + self.data_len


class BagFile:
    """Index-only view of a bag. Use as a context manager."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.file_size = self.path.stat().st_size
        self._f = open(self.path, "rb")
        self.index_pos = 0
        self.conn_count = 0
        self.chunk_count = 0
        self.connections: dict[int, Connection] = {}
        self.chunks: list[ChunkInfo] = []
        self.index_bytes = 0          # size of the index section that was read

    # ---------- context manager ----------
    def __enter__(self) -> "BagFile":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._f.close()

    # ---------- low-level record reading ----------
    def _read_record_header(self) -> tuple[dict[str, bytes], int] | None:
        """Read header + data_len at the current position. Data is NOT read."""
        raw = self._f.read(4)
        if len(raw) < 4:
            return None
        hlen = _u32(raw)
        hbuf = self._f.read(hlen)
        dl = self._f.read(4)
        if len(hbuf) < hlen or len(dl) < 4:
            raise BagError(f"truncated record at offset {self._f.tell()} (download not finished?)")
        return parse_header(hbuf), _u32(dl)

    def _read_exact(self, n: int) -> bytes:
        buf = self._f.read(n)
        if len(buf) < n:
            raise BagError(f"file ends early at offset {self._f.tell()} (download not finished?)")
        return buf

    # ---------- index ----------
    def open(self) -> None:
        f = self._f
        if f.read(len(MAGIC)) != MAGIC:
            raise BagError(f"{self.path.name}: not a ROS bag v2.0 file")
        rec = self._read_record_header()
        if rec is None or rec[0].get("op", b"\0")[0] != OP_BAG_HEADER:
            raise BagError("first record is not a BAG HEADER")
        h, _ = rec
        self.index_pos = _u64(h["index_pos"])
        self.conn_count = _u32(h["conn_count"])
        self.chunk_count = _u32(h["chunk_count"])
        if self.index_pos == 0:
            raise BagError("index_pos = 0: bag was not closed properly (unindexed)")
        if self.index_pos >= self.file_size:
            raise BagError(f"index_pos={self.index_pos} is beyond the file size {self.file_size} "
                           f"({self.file_size / self.index_pos:.1%} present): download not finished")

        f.seek(self.index_pos)
        while True:
            rec = self._read_record_header()
            if rec is None:
                break
            h, dlen = rec
            op = h["op"][0]
            data = self._read_exact(dlen)
            if op == OP_CONNECTION:
                ch = parse_header(data)
                c = Connection(conn=_u32(h["conn"]), topic=h["topic"].decode(),
                               msgtype=ch.get("type", b"?").decode(),
                               md5sum=ch.get("md5sum", b"").decode(),
                               msgdef=ch.get("message_definition", b"").decode(errors="replace"))
                self.connections[c.conn] = c
            elif op == OP_CHUNK_INFO:
                pairs = struct.iter_unpack("<II", data)
                self.chunks.append(ChunkInfo(idx=len(self.chunks), chunk_pos=_u64(h["chunk_pos"]),
                                             start_ns=_time_ns(h["start_time"]),
                                             end_ns=_time_ns(h["end_time"]),
                                             counts={c: n for c, n in pairs}))
        self.index_bytes = f.tell() - self.index_pos
        self.chunks.sort(key=lambda c: c.chunk_pos)
        for i, c in enumerate(self.chunks):
            c.idx = i
        if len(self.chunks) != self.chunk_count:
            raise BagError(f"bag header says {self.chunk_count} chunks, index has {len(self.chunks)}")

    @property
    def start_ns(self) -> int:
        return min(c.start_ns for c in self.chunks)

    @property
    def end_ns(self) -> int:
        return max(c.end_ns for c in self.chunks)

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def conns_for_topic(self, topic: str) -> list[int]:
        return [c.conn for c in self.connections.values() if c.topic == topic]

    def topic_counts(self) -> dict[int, int]:
        total: dict[int, int] = {}
        for ch in self.chunks:
            for c, n in ch.counts.items():
                total[c] = total.get(c, 0) + n
        return total

    # ---------- chunks ----------
    def read_chunk_header(self, chunk: ChunkInfo) -> ChunkHeader:
        self._f.seek(chunk.chunk_pos)
        rec = self._read_record_header()
        if rec is None or rec[0]["op"][0] != OP_CHUNK:
            raise BagError(f"no CHUNK record at offset {chunk.chunk_pos}")
        h, dlen = rec
        return ChunkHeader(compression=h["compression"].decode(), uncompressed_size=_u32(h["size"]),
                           data_pos=self._f.tell(), data_len=dlen)

    def read_chunk_raw(self, chunk: ChunkInfo) -> tuple[ChunkHeader, bytes]:
        """Compressed bytes of one chunk (one chunk in RAM, never the whole file)."""
        ch = self.read_chunk_header(chunk)
        return ch, self._read_exact(ch.data_len)

    @staticmethod
    def decompress(ch: ChunkHeader, raw: bytes) -> bytes:
        if ch.compression == "none":
            out = raw
        elif ch.compression == "bz2":
            out = bz2.decompress(raw)
        elif ch.compression == "lz4":
            import lz4.frame   # ROS 1 roslz4 writes the LZ4 frame format
            out = lz4.frame.decompress(raw)
        else:
            raise BagError(f"unknown chunk compression {ch.compression!r}")
        if len(out) != ch.uncompressed_size:
            raise BagError(f"chunk at {ch.data_pos}: got {len(out)} bytes, header says {ch.uncompressed_size}")
        return out

    def read_chunk(self, chunk: ChunkInfo) -> tuple[ChunkHeader, bytes]:
        ch, raw = self.read_chunk_raw(chunk)
        return ch, self.decompress(ch, raw)

    def read_index_data(self, chunk: ChunkInfo, ch: ChunkHeader | None = None) -> dict[int, np.ndarray]:
        """Per-connection message times (ns, int64) of one chunk, WITHOUT decompressing it.

        The INDEX DATA records sit right after the chunk data, one per connection in the chunk.
        """
        if ch is None:
            ch = self.read_chunk_header(chunk)
        self._f.seek(ch.end_pos)
        out: dict[int, np.ndarray] = {}
        for _ in range(len(chunk.counts)):
            rec = self._read_record_header()
            if rec is None or rec[0]["op"][0] != OP_INDEX_DATA:
                raise BagError(f"expected INDEX DATA after chunk {chunk.idx}")
            h, dlen = rec
            entries = np.frombuffer(self._read_exact(dlen), dtype=INDEX_ENTRY)
            out[_u32(h["conn"])] = entries["sec"].astype(np.int64) * 1_000_000_000 + entries["nsec"]
        return out


def iter_messages(buf: bytes, conns: set[int] | None = None) -> Iterator[tuple[int, int, memoryview]]:
    """Walk the records of one decompressed chunk.

    Yields (conn, time_ns, data) for MESSAGE DATA records whose conn is in `conns`
    (all if None). CONNECTION records inside the chunk are skipped. `data` is a
    memoryview into `buf` (no copy).
    """
    mv = memoryview(buf)
    pos, n = 0, len(buf)
    unpack_u32 = struct.Struct("<I").unpack_from
    while pos < n:
        hlen = unpack_u32(buf, pos)[0]
        h = parse_header(mv[pos + 4:pos + 4 + hlen])
        dpos = pos + 8 + hlen
        dlen = unpack_u32(buf, pos + 4 + hlen)[0]
        if h["op"][0] == OP_MSG_DATA:
            conn = _u32(h["conn"])
            if conns is None or conn in conns:
                yield conn, _time_ns(h["time"]), mv[dpos:dpos + dlen]
        pos = dpos + dlen


def iter_record_sizes(buf: bytes) -> Iterator[tuple[int, int]]:
    """(conn, data_len) for every MESSAGE DATA record of one decompressed chunk."""
    for conn, _, data in iter_messages(buf):
        yield conn, len(data)
