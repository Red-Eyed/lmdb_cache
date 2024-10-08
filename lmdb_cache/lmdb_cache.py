__author__ = "Vadym Stupakov"
__maintainer__ = "Vadym Stupakov"
__email__ = "vadim.stupakov@gmail.com"

from functools import partial
import shutil
from typing import Any, Iterable
import lmdb
from pathlib import Path
import dill


def _serialize(obj) -> bytes:
    data = dill.dumps(obj, protocol=dill.HIGHEST_PROTOCOL)
    return data


def _deserialize(data: bytes) -> Any:
    obj = dill.loads(data)
    return obj


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


class LMDBReadDict:
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
        env = lmdb.open(db_path.as_posix(),
                        subdir=True,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        return env

    def _txn(self):
        return self._env.begin()

    def __getitem__(self, index):
        lmdb_data = self._txn().get(str(index).encode())
        data = _deserialize(lmdb_data)

        return data


def _get_data(key: int, value: Any):
    return str(key).encode(), _serialize(value)


def _get_size(key: bytes, value: bytes):
    return len(key) + len(value)


def dump2lmdb(db_path: Path, iterable: Iterable, size_multiplier=100, block_size=1024**2, overwrite: bool = False) -> Path:
    if overwrite:
        shutil.rmtree(db_path)

    if lmdb_exists(db_path):
        return db_path
        
    db_path.mkdir(parents=True)
    all_size = 0
    open_lmdb = partial(lmdb.open, path=db_path.as_posix(), subdir=True, metasync=False, sync=False, writemap=True, map_async=True)
    try:
        env = open_lmdb()
        for i, value in enumerate(iterable):
            key, value = _get_data(i, value)
            size = _get_size(key, value)
            all_size += size

            try:
                with env.begin(write=True) as txn:
                    txn.put(key, value)
            except lmdb.MapFullError:
                # when LMDB reaches max size: close it, then reopenen with increased size
                all_size += max(block_size, size) * size_multiplier
                env.close()

                env = open_lmdb(map_size=all_size)
                with env.begin(write=True) as txn:
                    txn.put(key, value)

        env.close()
    except Exception:
        shutil.rmtree(db_path)
        raise

    assert lmdb_exists(db_path)

    return db_path
