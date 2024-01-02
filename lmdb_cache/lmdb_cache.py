
from functools import cached_property
import shutil
from typing import Any, Iterable
import lmdb
from pathlib import Path
import dill
import tqdm
import blosc2


def _serialize(obj) -> bytes:
    data = dill.dumps(obj, protocol=dill.HIGHEST_PROTOCOL)
    data = blosc2.compress2(data, typesize=4,
                            codec=blosc2.Codec.ZSTD,
                            )
    return data


def _deserialize(data: bytes) -> Any:
    data = blosc2.decompress2(data)
    obj = dill.loads(data)
    return obj


class LMDBReadDict:
    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self._db_path = Path(db_path).expanduser().resolve()

    def __getstate__(self):
        state = self.__dict__.copy()

        # Do not pickle next attributes:
        state.pop("_env", None)
        state.pop("_txn", None)
        return state

    @cached_property
    def __len__(self):
        env = self._env
        return env.stat()["entries"]

    @cached_property
    def _env(self):
        env = lmdb.open(self._db_path.as_posix(),
                        subdir=True,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        return env

    @cached_property
    def _txn(self):
        return self._env.begin()

    def __getitem__(self, index):
        lmdb_data = self._txn.get(str(index).encode())
        data = _deserialize(lmdb_data)

        return data


def _get_data(key: int, value: Any):
    return str(key).encode(), _serialize(value)


def _get_size(key: bytes, value: bytes):
    return len(key) + len(value)


def dump2lmdb(db_path: Path, iterable: Iterable, size_multiplier=100):
    db_path.mkdir(parents=True)
    all_size = 0
    with tqdm.tqdm(desc=f"Dumping to {db_path}", unit="B", unit_scale=True) as pbar:
        try:
            env = lmdb.open(db_path.as_posix(), subdir=True)
            for key, value in enumerate(iterable):
                key, value = _get_data(key, value)
                size = _get_size(key, value)
                all_size += size
                pbar.update(size)

                try:
                    with env.begin(write=True) as txn:
                        txn.put(key, value)
                except lmdb.MapFullError:
                    # when LMDB reaches max size: close it, then reopenen with increased size
                    map_size = all_size + len(value) * size_multiplier
                    env.close()

                    env = lmdb.open(db_path.as_posix(), subdir=True, map_size=map_size)
                    with env.begin(write=True) as txn:
                        txn.put(key, value)

            env.close()
        except Exception:
            shutil.rmtree(db_path)
            raise


def cache2lmdb(db_path: Path, iterable: Iterable, overwrite=False) -> LMDBReadDict:
    if overwrite:
        shutil.rmtree(db_path, ignore_errors=True)

    if db_path.exists():
        if len(list(db_path.iterdir())) == 0:
            db_path.rmdir()

    if not db_path.exists():
        dump2lmdb(db_path, iterable)

    cache_dataset = LMDBReadDict(db_path)
    return cache_dataset
