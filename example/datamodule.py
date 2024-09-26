from omg.datamodule import DataModule

# 1. direct lmdb loading
ds = DataModule(["example.lmdb"])
print(ds[0].coords)

ds.cleanup()