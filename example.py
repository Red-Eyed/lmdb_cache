from pathlib import Path
import shutil
from lmdb_cache import LMDBCacheCompressed, LMDBCache, lmdb_exists
from tempfile import TemporaryDirectory

if __name__ in "__main__":
    with TemporaryDirectory() as db_path:
        data_iterable = [(i, f"data_compressed_{i}") for i in range(1000)]
        shutil.rmtree(db_path)
        compressed_lmdb = LMDBCacheCompressed.from_iterable(Path(db_path), data_iterable)
        for i in range(len(compressed_lmdb)):
            print(i, compressed_lmdb[i])

    with TemporaryDirectory() as db_path:
        data_iterable = [(i, f"data_{i}") for i in range(1000)]
        shutil.rmtree(db_path)
        compressed_lmdb = LMDBCache.from_iterable(Path(db_path), data_iterable)
        for i in range(len(compressed_lmdb)):
            print(i, compressed_lmdb[i])
