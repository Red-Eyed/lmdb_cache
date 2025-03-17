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
    def __init__(self, db_path: Path) -> None:
        """
        db_path: valid path which points to a directory which contains "data.mdb" and "lock.mdb" files
        """
        super().__init__()

        if not lmdb_exists(db_path):
            raise RuntimeError(f"lmdb doesn't exists: {db_path}")

        self._db_path = Path(db_path).expanduser().resolve()
        self._env = self.get_env(self._db_path)

    def __getstate__(self):
        return self._db_path

    def __setstate__(self, path):
        self._db_path = path
        self._env = self.get_env(self._db_path)

    def __len__(self):
        env = self._env
        return env.stat()["entries"]

    @staticmethod
    def get_env(db_path: Path):
        env = lmdb.open(
            db_path.as_posix(),
            subdir=True,
            metasync=False,
            sync=False,
            writemap=True,
            map_async=True,
        )
        return env

    def _txn(self):
        return self._env.begin()

    def __getitem__(self, index):
        lmdb_data = self._txn().get(str(index).encode())
        if lmdb_data is None:
            raise KeyError(index)
        else:
            data = self.deserialize(lmdb_data)

        return data

    @classmethod
    def from_iterable(
        cls,
        db_path: Path,
        iterable: Iterable,
        size_multiplier=100,
        block_size=1024**2,
        batch_size=128,
    ):
        db_path.mkdir(parents=True)
        all_size = 0
        open_lmdb = partial(cls.get_env, db_path=db_path)
        try:
            env = open_lmdb()
            i = 0
            for batch in chunked(iterable, batch_size, False):
                batch_dict = {}
                for value in batch:
                    key, value = cls.get_data(i, value)
                    i += 1
                    batch_dict[key] = value
                    env, all_size = cls.write_batch(
                        env=env,
                        batch=batch_dict,
                        all_size=all_size,
                        block_size=block_size,
                        size_multiplier=size_multiplier,
                        open_lmdb=open_lmdb,
                    )

            env.close()
        except Exception:
            shutil.rmtree(db_path)
            raise

        assert lmdb_exists(db_path)

        return cls(db_path)

    @classmethod
    def write_batch(
        cls,
        env,
        batch: dict,
        all_size,
        block_size,
        size_multiplier,
        open_lmdb,
    ):
        for _ in range(2):
            try:
                with env.begin(write=True) as txn:
                    for key, value in batch.items():
                        size = cls.get_size(key, value)
                        all_size += size
                        txn.put(key, value)
            except lmdb.MapFullError:
                # when LMDB reaches max size: close it, then reopenen with increased size
                all_size += max(block_size, size) * size_multiplier
                env.close()

                env = open_lmdb(map_size=all_size)
                continue
            else:
                break

        return env, all_size


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
