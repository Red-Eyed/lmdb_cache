# LMDB Cache

`lmdb_cache` is a Python library leveraging LMDB for efficient and fast data handling, ideal for machine learning workflows. It simplifies the process of storing and retrieving large datasets using LMDB.

## Key Features

- **Efficient Serialization**: Serialize anything: it utilizes `dill` for object serialization and deserialization.
- **Two-Stage Data Handling**:
  - **Stage 1**: Use once `dump2lmdb` to efficiently dump large datasets into an LMDB database.
  - **Stage 2**: Retrieve data using `LMDBReadDict`, supporting multiprocessing for high-throughput applications.
- **Supposed to be used whithin ML training pipelines**: Can be integrated with PyTorch `Dataset` and `DataLoader`, making it ideal for multi-process data loading in machine pipelines.

## Installation

```bash
python3 -m pip install https://github.com/Red-Eyed/lmdb_cache.git
```

## Usage

### Stage 1: Data Dumping with `dump2lmdb`

```python
from lmdb_cache import dump2lmdb
from pathlib import Path

db_path = Path("/path/to/lmdb/database")
data_iterable = [(i, f"data_{i}") for i in range(1000)]
dump2lmdb(db_path, data_iterable)
```

### Stage 2: Retrieving Data with `LMDBReadDict`

```python
from lmdb_cache import LMDBReadDict
from pathlib import Path

db_path = Path("/path/to/lmdb/database")
lmdb_dict = LMDBReadDict(db_path)

for i in range(1000):
    data = lmdb_dict[i]
    print(f"Key: {i}, Data: {data}")
```

#### Usage within `PyTorch`
```python
import torch
from lmdb_cache import LMDBReadDict
from torch.utils.data import Dataset, DataLoader

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_dict = LMDBReadDict(lmdb_path)

    def __len__(self):
        return len(self.lmdb_dict)

    def __getitem__(self, idx):
        return self.lmdb_dict[idx]

# Usage
# dump once
db_path = Path("/path/to/lmdb/database")
dump2lmdb(db_path, data_iterable)

# read multiple times
lmdb_dataset = LMDBDataset(db_path)
data_loader = DataLoader(lmdb_dataset, batch_size=32, shuffle=True)

for batch in data_loader:
    # Process your batch
```
