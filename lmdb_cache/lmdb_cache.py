__author__ = "Vadym Stupakov"
__maintainer__ = "Vadym Stupakov"
__email__ = "vadim.stupakov@gmail.com"

from functools import partial
import shutil
from typing import Any, Iterable
import lmdb
from pathlib import Path
import dill
import brotli
from more_itertools import chunked


def lmdb_exists(p: Path) -> bool:
    if p.exists():
        if p.is_dir():
            files = {f.name for f in p.iterdir() if f.is_file()}
            db_files = {"data.mdb", "lock.mdb"}
            return db_files.issubset(files)
        else:
            raise NotADirectoryError(p)
    else:
        return False


class SerializeMixIn:
    @classmethod
    def serialize(cls, obj) -> bytes:
        data = dill.dumps(obj, protocol=dill.HIGHEST_PROTOCOL)
        return data

    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        obj = dill.loads(data)
        return obj

    @classmethod
    def get_data(cls, key: int, value: Any):
        return str(key).encode(), cls.serialize(value)

    @classmethod
    def get_size(cls, key: bytes, value: bytes):
        return len(key) + len(value)


class LMDBCache(SerializeMixIn):
    """
    A read-optimized LMDB-backed key-value store that supports:
    - write-once from an iterable
    - safe multiprocess reads (e.g., in PyTorch DataLoader)
    """

    def __init__(self, db_path: Path) -> None:
        """
        Args:
            db_path: Path to directory containing the LMDB database (data.mdb, lock.mdb)
        """
        super().__init__()

        if not lmdb_exists(db_path):
            raise RuntimeError(f"LMDB does not exist: {db_path}")

        self._db_path = db_path.expanduser().resolve()
        self._env = None  # Lazy init to support fork-safe multiprocessing

    def __getstate__(self) -> Path:
        # Used for pickling — save only the DB path
        return self._db_path

    def __setstate__(self, path: Path) -> None:
        # Restore path and defer env creation (lazy init)
        self._db_path = path
        self._env = None

    @property
    def env(self) -> lmdb.Environment:
        """
        Fork-safe LMDB environment accessor.
        Each process gets its own LMDB handle.
        """
        if self._env is None:
            self._env = self.get_read_env(self._db_path)
        return self._env

    def __len__(self) -> int:
        # Return number of entries in the database
        return self.env.stat()["entries"]

    def __getitem__(self, index: int) -> Any:
        """
        Fetch and deserialize a single item by index.

        Args:
            index: Integer index (converted to byte key)
        Raises:
            KeyError if index is not found
        """
        key = str(index).encode()
        with self.env.begin(write=False) as txn:
            raw = txn.get(key) or b""  # Avoid None
        if not raw:
            raise KeyError(index)

        return self.deserialize(raw)

    @staticmethod
    def get_read_env(db_path: Path, **kw) -> lmdb.Environment:
        """
        Returns a read-only LMDB environment.
        Safe for multi-process access (e.g., PyTorch DataLoader).
        """
        return lmdb.open(
            db_path.as_posix(),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
            **kw,
        )

    @staticmethod
    def get_write_env(
        db_path: Path, map_size: int = 10 * 1024**2, **kw
    ) -> lmdb.Environment:
        """
        Returns a writable LMDB environment.
        Should only be used in single-process setup when creating the database.
        """
        return lmdb.open(
            db_path.as_posix(),
            subdir=True,
            map_size=map_size,
            metasync=False,
            sync=False,
            writemap=True,
            map_async=True,
            **kw,
        )

    @classmethod
    def from_iterable(
        cls,
        db_path: Path,
        iterable: Iterable,
        size_multiplier: int = 100,
        block_size: int = 1024**2,
        batch_size: int = 128,
    ) -> "LMDBCache":
        """
        Create and populate an LMDB from a batched iterable.

        Args:
            db_path: Output directory for LMDB
            iterable: Iterable yielding raw values
            size_multiplier: Resize multiplier on MapFullError
            block_size: Minimum expansion block size in bytes
            batch_size: Items per write batch
        """
        db_path.mkdir(parents=True)
        env = None
        all_size = 0
        open_lmdb = partial(cls.get_write_env, db_path=db_path)

        try:
            env = open_lmdb()
            i = 0
            for batch in chunked(iterable, batch_size, strict=False):
                # Build {key: serialized_value} map for the batch
                batch_dict = {
                    cls.get_data(i + j, val)[0]: cls.get_data(i + j, val)[1]
                    for j, val in enumerate(batch)
                }
                i += len(batch)
                # Write batch into LMDB
                env, all_size = cls.write_batch(
                    env, batch_dict, all_size, block_size, size_multiplier, open_lmdb
                )
        except Exception:
            # Clean up partially written DB
            if env:
                env.close()
            shutil.rmtree(db_path)
            raise

        env.close()
        assert lmdb_exists(db_path)
        return cls(db_path)

    @classmethod
    def write_batch(
        cls,
        env: lmdb.Environment,
        batch: dict[bytes, bytes],
        all_size: int,
        block_size: int,
        size_multiplier: int,
        open_lmdb,
    ) -> tuple[lmdb.Environment, int]:
        """
        Write a batch of items into the LMDB, expanding map_size if needed.

        Args:
            env: Current LMDB environment
            batch: Dict of serialized key-value pairs
            all_size: Estimated total DB size so far
            block_size: Size of map increase chunk
            size_multiplier: Expansion multiplier
            open_lmdb: Callable to reopen the DB
        Returns:
            (new_env, new_total_size)
        """
        for _ in range(2):
            try:
                with env.begin(write=True) as txn:
                    for key, value in batch.items():
                        all_size += cls.get_size(key, value)
                        txn.put(key, value)
            except lmdb.MapFullError:
                # Increase map size and retry
                all_size += block_size * size_multiplier
                env.close()
                env = open_lmdb(map_size=all_size)
                continue
            break
        return env, all_size

    @classmethod
    def get_data(cls, index: int, value: Any) -> tuple[bytes, bytes]:
        """
        Convert index and value to (key, serialized_value) pair.
        """
        return str(index).encode(), cls.serialize(value)

    @staticmethod
    def get_size(key: bytes, value: bytes) -> int:
        """
        Return total byte size of key + value pair.
        Used for map size estimation.
        """
        return len(key) + len(value)


class SerializeWithCompressionMixIn(SerializeMixIn):
    @classmethod
    def serialize(cls, obj) -> bytes:
        data = super().serialize(obj)
        data = brotli.compress(data)
        return data

    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        data = brotli.decompress(data)
        data = super().deserialize(data)
        return data


class LMDBCacheCompressed(LMDBCache, SerializeWithCompressionMixIn):
    pass
