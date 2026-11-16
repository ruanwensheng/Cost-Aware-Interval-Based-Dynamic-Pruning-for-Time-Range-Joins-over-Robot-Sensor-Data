"""Print the records of a ROS 1 bag file (format 2.0), one by one.

Usage:
    python inspect_bag.py my.bag              # top-level records (first 20)
    python inspect_bag.py my.bag --limit 50   # more records
    python inspect_bag.py my.bag --chunk 0    # also open chunk number 0 and list what is inside
    python inspect_bag.py my.bag --index      # jump to the end: connection and chunk info records

Only the Python standard library is needed (plus `pip install lz4` for lz4 bags).
The file is read record by record, so it also works on a 30 GB bag.
"""
import argparse, bz2, struct

OPS = {0x03: "BAG HEADER", 0x05: "CHUNK", 0x07: "CONNECTION",
       0x02: "MESSAGE DATA", 0x04: "INDEX DATA", 0x06: "CHUNK INFO"}


def read_record(f):
    """One record = header_len (4 bytes) + header + data_len (4 bytes) + data."""
    raw = f.read(4)
    if len(raw) < 4:
        return None
    header_len = struct.unpack("<I", raw)[0]
    header = parse_header(f.read(header_len))
    data_len = struct.unpack("<I", f.read(4))[0]
    return header, data_len


def parse_header(buf):
    """The header is a list of fields: field_len (4 bytes) + b'name=value'."""
    fields, pos = {}, 0
    while pos < len(buf):
        n = struct.unpack_from("<I", buf, pos)[0]
        name, value = buf[pos + 4:pos + 4 + n].split(b"=", 1)
        fields[name.decode()] = value
        pos += 4 + n
    return fields


def as_time(b):            # ROS time = uint32 seconds + uint32 nanoseconds
    s, ns = struct.unpack("<II", b)
    return f"{s}.{ns:09d}"


def show(header, data_len, f=None, data=None):
    op = header["op"][0]
    name = OPS.get(op, f"op {op:#x}")
    h = header
    if op == 0x03:
        info = (f"index_pos={struct.unpack('<Q', h['index_pos'])[0]} "
                f"conn_count={struct.unpack('<I', h['conn_count'])[0]} "
                f"chunk_count={struct.unpack('<I', h['chunk_count'])[0]}")
    elif op == 0x05:
        info = f"compression={h['compression'].decode()} uncompressed_size={struct.unpack('<I', h['size'])[0]}"
    elif op == 0x07:
        info = f"conn={struct.unpack('<I', h['conn'])[0]} topic={h['topic'].decode()}"
        if data is not None:   # the data of a connection record is another header: type, md5sum, definition
            ch = parse_header(data)
            info += f" type={ch.get('type', b'?').decode()}"
    elif op == 0x02:
        info = f"conn={struct.unpack('<I', h['conn'])[0]} time={as_time(h['time'])}"
    elif op == 0x04:
        info = f"conn={struct.unpack('<I', h['conn'])[0]} count={struct.unpack('<I', h['count'])[0]}"
        if data is not None:   # data = count x (time 8 bytes, offset in the uncompressed chunk 4 bytes)
            entries = [(as_time(data[i:i + 8]), struct.unpack_from('<I', data, i + 8)[0])
                       for i in range(0, min(len(data), 36), 12)]
            info += f" first entries (time, offset)={entries}"
    elif op == 0x06:
        info = (f"chunk_pos={struct.unpack('<Q', h['chunk_pos'])[0]} "
                f"start={as_time(h['start_time'])} end={as_time(h['end_time'])} "
                f"connections={struct.unpack('<I', h['count'])[0]}")
        if data is not None:   # data = per connection: (conn 4 bytes, message count 4 bytes)
            pairs = [struct.unpack_from('<II', data, i) for i in range(0, len(data), 8)]
            info += f" (conn, messages)={pairs}"
    else:
        info = ""
    print(f"  {name:<13} data={data_len:>9} bytes  {info}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=None, help="open this chunk (0 = first) and list its records")
    ap.add_argument("--index", action="store_true", help="skip the chunks and show the index section at the end")
    a = ap.parse_args()

    with open(a.bag, "rb") as f:
        print("magic line:", f.readline())                 # b'#ROSBAG V2.0\n'
        shown, chunk_no = 0, -1
        while True:
            pos = f.tell()
            rec = read_record(f)
            if rec is None:
                break
            header, data_len = rec
            op = header["op"][0]
            small = op in (0x07, 0x04, 0x06)               # small records: read their data to show it
            data = f.read(data_len) if small else None
            if shown < a.limit:
                print(f"@{pos:>11}", end="")
                show(header, data_len, data=data)
                shown += 1
            if op == 0x03 and a.index:                      # jump to the index section at the end
                f.seek(struct.unpack("<Q", header["index_pos"])[0])
                print("  ... jumping to the index section at the end of the file ...")
                continue
            if op == 0x05:
                chunk_no += 1
                if a.chunk == chunk_no:
                    raw = f.read(data_len)
                    open_chunk(header, raw)
                    continue
            if not small:
                f.seek(data_len, 1)                         # skip big data (chunks, padding)


def open_chunk(header, raw):
    comp = header["compression"].decode()
    if comp == "bz2":
        raw = bz2.decompress(raw)
    elif comp == "lz4":
        import lz4.frame
        raw = lz4.frame.decompress(raw)
    print(f"\n  ---- inside this chunk ({comp}, {len(raw)} bytes after decompression) ----")
    pos = 0
    while pos < len(raw):
        header_len = struct.unpack_from("<I", raw, pos)[0]
        header = parse_header(raw[pos + 4:pos + 4 + header_len])
        data_len = struct.unpack_from("<I", raw, pos + 4 + header_len)[0]
        data = raw[pos + 8 + header_len:pos + 8 + header_len + data_len]
        print(f"    offset {pos:>8}", end="")
        show(header, data_len, data=data)
        pos += 8 + header_len + data_len
    print("  ---- end of chunk ----\n")


if __name__ == "__main__":
    main()
