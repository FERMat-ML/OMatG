from omg.datamodule import DataModule

# 1. direct lmdb loading
ds = DataModule.from_lmdb(["example.lmdb"], dynamic_loading=True)
print(ds[0].coords)

ds.cleanup()