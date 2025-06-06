from pathlib import Path
import random
import pytest
from multiprocessing import Pool, Process, Value
from lmdb_cache import LMDBCache, LMDBCacheCompressed, lmdb_exists
import os


def generate_dataset(num_items: int, max_mb: int = 10) -> list[bytes]:
    return [
        os.urandom(random.randint(0, max_mb * 1024 * 1024)) for _ in range(num_items)
    ]


class_under_test_t = LMDBCache | LMDBCacheCompressed


@pytest.fixture(params=[LMDBCache, LMDBCacheCompressed])
def class_under_test(request):
    return request.param


@pytest.fixture
def setup_lmdb(request, class_under_test: class_under_test_t, tmp_path):
    db_path = tmp_path / f"{request.node.name}_lmdb_test"
    data = {i: f"data_{i}" for i in range(100)}
    class_under_test.from_iterable(db_path, data.values())
    return db_path, data


def test_serialize_deserialize(class_under_test):
    obj = {"a": 1, "b": 2}
    serialized_obj = class_under_test.serialize(obj)
    deserialized_obj = class_under_test.deserialize(serialized_obj)
    assert obj == deserialized_obj


def test_lmdb_exists(tmp_path):
    db_path = tmp_path / "lmdb_test"
    assert not lmdb_exists(db_path)
    db_path.mkdir()
    (db_path / "data.mdb").touch()
    (db_path / "lock.mdb").touch()
    assert lmdb_exists(db_path)


def test_lmdb_exists_not_a_directory(tmp_path):
    db_file = tmp_path / "not_a_dir.mdb"
    db_file.touch()
    with pytest.raises(NotADirectoryError):
        lmdb_exists(db_file)


def test_init(class_under_test: class_under_test_t, setup_lmdb):
    db_path, _ = setup_lmdb
    db = class_under_test(db_path)
    assert isinstance(db, class_under_test)


def test_len(class_under_test: class_under_test_t, setup_lmdb):
    db_path, data = setup_lmdb
    db = class_under_test(db_path)
    assert len(db) == len(data)


def test_getitem(class_under_test, setup_lmdb):
    db_path, data = setup_lmdb
    db = class_under_test(db_path)
    for key, value in data.items():
        assert db[key] == value


def test_from_iterable(class_under_test, tmp_path):
    db_path = tmp_path / "lmdb_dump"
    data = [f"data_{i}" for i in range(10)]
    class_under_test.from_iterable(db_path, data)
    assert lmdb_exists(db_path)

    db = class_under_test(db_path)
    for i, value in enumerate(data):
        assert db[i] == value


def test_from_iterable_cleanup_on_failure(tmp_path):
    db_path = tmp_path / "lmdb_fail"
    with pytest.raises(Exception):
        class_under_test.from_iterable(
            db_path, iter(Exception("Forced failure")), size_multiplier=1, block_size=1
        )
    assert not db_path.exists()


def test_concurrent_reading(class_under_test, setup_lmdb):
    db_path, data = setup_lmdb
    db = class_under_test(db_path)
    with Pool(5) as p:
        keys = list(data.keys())
        results = p.map(db.__getitem__, keys)

    # Check that all read results match the expected data
    for key, result in zip(keys, results):
        assert data[key] == result


def test_invalid_path(class_under_test):
    with pytest.raises(RuntimeError):  # Assuming RuntimeError for non-existent DB
        class_under_test(Path("/path/to/nonexistent/db"))


def test_nonexistent_key(class_under_test, setup_lmdb):
    db_path, _ = setup_lmdb
    db = class_under_test(db_path)
    with pytest.raises(KeyError):  # Assuming KeyError for non-existent key
        _ = db["nonexistent_key"]


def test_invalid_db_path_type(class_under_test):
    with pytest.raises(AttributeError):  # Adjust the exception type as needed
        class_under_test(12345)  # Passing an integer instead of a Path or string


def attempt_write(db_path, success_flag):
    try:
        db = class_under_test(db_path)  # Assuming this opens the DB in read-only mode
        with db._env.begin(write=True) as txn:  # Attempt to start a write transaction
            txn.put(b"key", b"value")
        success_flag.value = 1
    except Exception:
        success_flag.value = 0


def test_concurrent_write_attempt(setup_lmdb):
    db_path, _ = setup_lmdb
    success_flag = Value("i", 0)
    p = Process(target=attempt_write, args=(db_path, success_flag))
    p.start()
    p.join()
    assert success_flag.value == 0  # Expect the write attempt to fail


def test_from_iterable_large(class_under_test, tmp_path):
    db_path = tmp_path / "lmdb_dump"
    data = [f"data_{i}" for i in range(10_000)]
    cache = class_under_test.from_iterable(db_path, data)
    assert lmdb_exists(db_path)
    for i in range(1000):
        k = random.randint(0, len(data) - 1)
        cache[k]


def test_lazy_env_after_pickle(setup_lmdb, class_under_test):
    import pickle

    db_path, data = setup_lmdb
    db = class_under_test(db_path)

    # Simulate pickling in a parent process and unpickling in a child
    pickled = pickle.dumps(db)
    db_unpickled = pickle.loads(pickled)

    assert db_unpickled[42] == data[42]


def test_custom_keys_from_iterable(tmp_path, class_under_test):
    db_path = tmp_path / "lmdb_custom_keys"
    data = {f"key_{i}": f"value_{i}" for i in range(50)}
    class_under_test.from_iterable(db_path, data.values())
    db = class_under_test(db_path)
    assert db[10] == "value_10"


def test_read_fallback_does_not_suppress_missing_key(setup_lmdb, class_under_test):
    db_path, _ = setup_lmdb
    db = class_under_test(db_path)
    with pytest.raises(KeyError):
        _ = db[9999]  # Key not inserted


@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("size_multiplier", [1, 10])
def test_varied_batch_and_map_size(
    tmp_path, batch_size, size_multiplier, class_under_test
):
    db_path = tmp_path / "lmdb_varied"
    data = generate_dataset(batch_size * 2, 10)
    db = class_under_test.from_iterable(
        db_path,
        data,
        batch_size=batch_size,
        size_multiplier=size_multiplier,
    )

    assert len(data) == len(db)
