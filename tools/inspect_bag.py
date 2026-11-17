"""
A tool to print the records of a ROS 1 bag file (format 2.0), one by one.
NOT USED IN THE EDA_BAG code. 

Usage:
    python inspect_bag.py my.bag              # top-level records (first 20)
    python inspect_bag.py my.bag --limit 50   # more records
    python inspect_bag.py my.bag --chunk 0    # also open chunk number 0 and list what is inside
    python inspect_bag.py my.bag --index      # jump to the end: connection and chunk info records

"""
import argparse, bz2, struct

# struct module: covert between python val and C binary data
# struct.pack(), struct.unpack(format_string, string)

OPS = {0x03: "BAG HEADER", 
       0x05: "CHUNK", 
       0x07: "CONNECTION",
       0x02: "MESSAGE DATA", 
       0x04: "INDEX DATA", 
       0x06: "CHUNK INFO"}

"""
read_record <-- parse_header <--"""
"""
    One record = 
    header_len (4 bytes) + 
    header (N bytes) + 
    data_len (4 bytes) + 
    data (M bytes)."""

def read_record(f):
    """
    TO DECOMPRESS HEADER AND DATA LENGTH
    input: file object (f) opened in rb mode (read binary)
    process: 
        1. read the header's length (first 4 bytes)
        2. decompress header's length (unpack, [0] to get the first of the tuple)
        3. read the header (the next N = header's length bytes) by a 
        header's parser (this parser output a dictionary of metadata of the bag file)
        4. read and decompress the data's length (decpress the next 4 bytes, get the 0 id of the tuple)
    output: 3 (decompressed header in dict format) and 4 (data's length)"""
    raw = f.read(4)                   # 1
    if len(raw) < 4:                  # 1 
        return None
    header_len = struct.unpack("<I", raw)[0]      # 2

    header = parse_header(f.read(header_len))    # 3 
    data_len = struct.unpack("<I", f.read(4))[0]  # 4


    return header, data_len


def parse_header(buf):
    """
    TO DECOMPRESS HEADER INTO DICT FORMAT
    The header is a list of fields: field_len (4 bytes) + b'name=value'."""
    fields, pos = {}, 0
    while pos < len(buf):
        n = struct.unpack_from("<I", buf, pos)[0]
        name, value = buf[pos + 4:pos + 4 + n].split(b"=", 1)
        fields[name.decode()] = value
        pos += 4 + n
    return fields


def as_time(b):           
    """
    TO DECOMPRESS TIME DATA TO SECOND AND NANOSEC
    input: b is an 8 byte seq, get from header (in read_record)
    process: 
         1. decompress 8 binary byte --> 2 ints: s and ns.

    output: combine seconds and nanoseconds (ns forced to show 9 number)"""
    s, ns = struct.unpack("<II", b) # 1
    return f"{s}.{ns:09d}"


def show(header, data_len, f=None, data=None):
    """
    TO SHOW DATA INFO 
    input: header (from read_record): dict, data_len (read_record): int
           f: file obj, data: bytes
    process:
        1. Get the type of record in the header 
        2. Decompress the header (depends on the type found)
    output: info
    """
    op = header["op"][0]                        #1
    name = OPS.get(op, f"op {op:#x}")           #1
    h = header                                  # 1

    #2
    if op == 0x03:  # BAG HEADER
        info = (f"index_pos={struct.unpack('<Q', h['index_pos'])[0]} "
                f"conn_count={struct.unpack('<I', h['conn_count'])[0]} "
                f"chunk_count={struct.unpack('<I', h['chunk_count'])[0]}")
    elif op == 0x05:  # CHUNK
        info = f"compression={h['compression'].decode()} uncompressed_size={struct.unpack('<I', h['size'])[0]}"
    elif op == 0x07: # CONNECTION
        info = f"conn={struct.unpack('<I', h['conn'])[0]} topic={h['topic'].decode()}"
        if data is not None:   # the data of a connection record is another header: type, md5sum, definition
            ch = parse_header(data)
            info += f" type={ch.get('type', b'?').decode()}"
    elif op == 0x02: # MESSAGE DATA
        info = f"conn={struct.unpack('<I', h['conn'])[0]} time={as_time(h['time'])}"
    elif op == 0x04: # INDEX DATA
        info = f"conn={struct.unpack('<I', h['conn'])[0]} count={struct.unpack('<I', h['count'])[0]}"
        if data is not None:   # data = count x (time 8 bytes, offset in the uncompressed chunk 4 bytes)
            entries = [(as_time(data[i:i + 8]), struct.unpack_from('<I', data, i + 8)[0])
                       for i in range(0, min(len(data), 36), 12)]
            info += f" first entries (time, offset)={entries}"
    elif op == 0x06:  # CHUNK INFO
        info = (f"chunk_pos={struct.unpack('<Q', h['chunk_pos'])[0]} "
                f"start={as_time(h['start_time'])} end={as_time(h['end_time'])} "
                f"connections={struct.unpack('<I', h['count'])[0]}")
        if data is not None:   # data = per connection: (conn 4 bytes, message count 4 bytes)
            pairs = [struct.unpack_from('<II', data, i) for i in range(0, len(data), 8)]
            info += f" (conn, messages)={pairs}"
    else: # STH WRONG ?????
        info = ""
    print(f"  {name:<13} data={data_len:>9} bytes  {info}")


def open_chunk(header, raw):
    """
    TO DECOMPRESS A CHUNK
    input: header (dict), raw
    process:
            1. get the compression method name in header dict
            2. decompress raw chunk with the corresponding method
    output:  comp, no of bytes after decompression, 
    """
    comp = header["compression"].decode()   #1
    if comp == "bz2":                    #2
        raw = bz2.decompress(raw)
    elif comp == "lz4":              #2 (this actually not used in bag)
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


def main():
    """
    TO SHOW INFORMATION ABOUT THE WHOLE BAG FILE
    input: --
    process:
        1. define a argument parser object and add arguments for it (positional: bắt buộc
        & optional : tùy chọn). bag: đường dẫn file bag; --limit: giới hạn số lượng record
        -- chunk: id of chunk; --index: action = True nếu include else False).
        Sau cùng thực hiện parse (truyền tham số vào các hàm)
        2. read the magic line 
        3. shown: đếm số bản ghi đã ghi ra màn hình; chunk_no: đếm stt chunk
        4. Loop (get byte offset -> read record ->   )
    output: 
    """
    ap = argparse.ArgumentParser()   #1
    ap.add_argument("bag")          #1
    ap.add_argument("--limit", type=int, default=20) #1
    ap.add_argument("--chunk", type=int, default=None, help="open this chunk (0 = first) and list its records")
    ap.add_argument("--index", action="store_true", help="skip the chunks and show the index section at the end")
    a = ap.parse_args() #1

    with open(a.bag, "rb") as f:
        print("magic line:", f.readline())       #2           # b'#ROSBAG V2.0\n'
        
        shown, chunk_no = 0, -1       #3

        # the loop 4
        while True:
            if shown >= a.limit and (a.chunk is None or chunk_no >= a.chunk):
                break                                       # printed enough: do not scan the rest of the file
            pos = f.tell()  # start offset of the record    
            rec = read_record(f) # the file pointer now at the start of the file

            if rec is None:
                break
            header, data_len = rec
            op = header["op"][0]

            # SMALL RECORDS (CONNECTION, INDEX DATA, CHUNKINFO)
            small = op in (0x07, 0x04, 0x06)               # small records: read their data to show it
            data = f.read(data_len) if small else None
            if shown < a.limit:
                print(f"@{pos:>11}", end="")
                show(header, data_len, data=data)
                shown += 1

            # BAG HEADER
            if op == 0x03 and a.index:                      # jump to the index section at the end
                f.seek(struct.unpack("<Q", header["index_pos"])[0])
                print("  ... jumping to the index section at the end of the file ...")
                continue

            # CHUNK
            if op == 0x05:
                chunk_no += 1
                if a.chunk == chunk_no:
                    raw = f.read(data_len)
                    open_chunk(header, raw)
                    continue
            # skip ...
            if not small:
                f.seek(data_len, 1)                         # skip big data (chunks, padding)

if __name__ == "__main__":
    main()
