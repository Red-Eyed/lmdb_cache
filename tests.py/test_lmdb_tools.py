import pytest
from multiprocessing import Pool
from lmdb_cache.lmdb_cache import _serialize, _deserialize, lmdb_exists, LMDBReadDict, dump2lmdb

@pytest.fixture
def setup_lmdb(tmp_path):
    db_path = tmp_path / "lmdb_test"
    data = {i: f"data_{i}" for i in range(100)}
    dump2lmdb(db_path, data.values())
    return db_path, data

def test_serialize_deserialize():
    obj = {'a': 1, 'b': 2}
    serialized_obj = _serialize(obj)
    deserialized_obj = _deserialize(serialized_obj)
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

def test_LMDBReadDict_init(setup_lmdb):
    db_path, _ = setup_lmdb
    db = LMDBReadDict(db_path)
    assert isinstance(db, LMDBReadDict)

def test_LMDBReadDict_len(setup_lmdb):
    db_path, data = setup_lmdb
    db = LMDBReadDict(db_path)
    assert len(db) == len(data)

def test_LMDBReadDict_getitem(setup_lmdb):
    db_path, data = setup_lmdb
    db = LMDBReadDict(db_path)
    for key, value in data.items():
        assert db[key] == value

def test_dump2lmdb(tmp_path):
    db_path = tmp_path / "lmdb_dump"
    data = [f"data_{i}" for i in range(10)]
    dump2lmdb(db_path, data)
    assert lmdb_exists(db_path)

    db = LMDBReadDict(db_path)
    for i, value in enumerate(data):
        assert db[i] == value

def test_dump2lmdb_cleanup_on_failure(tmp_path):
    db_path = tmp_path / "lmdb_fail"
    with pytest.raises(Exception):
        dump2lmdb(db_path, iter(Exception('Forced failure')), size_multiplier=1, block_size=1)
    assert not db_path.exists()


def test_concurrent_reading(setup_lmdb):
    db_path, data = setup_lmdb
    db = LMDBReadDict(db_path)
    with Pool(5) as p:
        keys = list(data.keys())
        results = p.map(db.__getitem__, keys)

    # Check that all read results match the expected data
    for key, result in zip(keys, results):
        assert data[key] == result
