# LMDB Cache

`lmdb_cache` is a lightweight Python utility that wraps [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) for **fast, safe, and multiprocessing-friendly caching**.   
It is mostly intended for machine learning data pipelines (torch `Dataloader`).

---

## ✅ Features

- **Fast multi-process reads** with memory-mapped performance
- **Write once, read many** architecture
- **Clean serialization** with support for any Python object
- **Safe for PyTorch DataLoader** (supports `num_workers > 0`)
- **Batched LMDB writes** with auto-expanding map size

---

## 📦 Installation

```bash
python3 -m pip install https://github.com/Red-Eyed/lmdb_cache.git
```

## 🚀 Example usage

### Simple example
```python
from lmdb_cache import LMDBCache
from pathlib import Path
import tempfile

# Sample data
data = [("foo", i) for i in range(100)]

# Create a temporary LMDB directory
db_path = Path(tempfile.gettempdir()) / "example_lmdb"

# Write dataset (once)
lmdb_cache = LMDBCache.from_iterable(db_path, data)

# Random access by index
print(lmdb_cache[10])  # Output: ("foo", 10)
print(len(lmdb_cache))  # Output: 100

```

### PyTorch `Dataloader` example

```python
from torch.utils.data import Dataset, DataLoader
from lmdb_cache import LMDBCache
from pathlib import Path
import tempfile

class LMDBDataset(Dataset):
    def __init__(self, db_path):
        self.db = LMDBCache(db_path)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        return self.db[idx]

# Load LMDB
db_path = Path(tempfile.gettempdir()) / "example_lmdb"
dataset = LMDBDataset(db_path)
loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True)

for batch in loader:
    print(batch)  # Use your data here
```
