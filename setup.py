from setuptools import setup, find_packages

setup(
    name='OMG',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'lightning',
        'torch_geometric',
        'lmdb',
        'loguru',
        'monty',
        'ase',
        'lmdb',
        'loguru',
        'tqdm'
    ],
    author='FERMat',
)