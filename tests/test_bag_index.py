"""BagFile on a file-like object, and the 2-request index read of bag_engine.

The full engine tests (fake bag, all modes, S3) are in tests/test_engines.py.
Run:  python -m unittest discover -s tests -v
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roslab.bag_engine import SparseFile, read_index  # noqa: E402
from roslab.bagfile import BagError, BagFile  # noqa: E402
from roslab.storage import LocalStore  # noqa: E402


def write_tiny_bag(path: Path) -> None:
    ts = get_typestore(Stores.ROS1_NOETIC)
    F = ts.types["std_msgs/msg/Float64"]
    w = Writer(path)
    w.set_compression(Writer.CompressionFormat.BZ2)
    w.chunk_threshold = 2048
    with w:
        a = w.add_connection("/a", F.__msgtype__, typestore=ts)
        b = w.add_connection("/b", F.__msgtype__, typestore=ts)
        for i in range(1500):
            c = a if i % 3 else b
            w.write(c, 10**18 + i * 10**7, ts.serialize_ros1(F(data=float(i)), c.msgtype))


class TestBagIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "tiny.bag"
        write_tiny_bag(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def summary(self, bag):
        return [(c.chunk_pos, c.start_ns, c.end_ns, c.counts) for c in bag.chunks], bag.index_pos

    def test_fileobj_same_as_path(self):
        with BagFile(self.path) as ref:
            want = self.summary(ref)
        f = io.BytesIO(self.path.read_bytes())
        with BagFile("any-name.bag", fileobj=f) as bag:
            self.assertEqual(self.summary(bag), want)
            self.assertGreater(len(bag.chunks), 5)
            _, raw = bag.read_chunk(bag.chunks[1])          # chunks readable through the file-like
            self.assertGreater(len(raw), 0)
        self.assertFalse(f.closed)                          # caller keeps ownership

    def test_read_index_two_requests(self):
        store = LocalStore(self.path.parent)
        try:
            bag = read_index(store, "tiny.bag")
            self.assertEqual(store.requests, 2)
            self.assertEqual(store.bytes, 4096 + self.path.stat().st_size - bag.index_pos)
            with BagFile(self.path) as ref:
                self.assertEqual(self.summary(bag), self.summary(ref))
        finally:
            store.close()

    def test_sparse_file(self):
        f = SparseFile({0: b"abcd", 10: b"xyz"}, size=13)
        self.assertEqual(f.read(2), b"ab")
        f.seek(11)
        self.assertEqual((f.read(), f.tell()), (b"yz", 13))
        self.assertEqual(f.read(), b"")
        f.seek(5)
        with self.assertRaises(BagError):
            f.read(1)


if __name__ == "__main__":
    unittest.main()
