# abinitioTools
AbinitioTools is a Python package for convenient tools of quantum chemical calculations.

## Features
- Application of static electric field for HF, DFT and CCSD calculations
- Correlation functions
  - Same-time and same-position current-current correlation function
  - Spin-spin correlation function
  - Charge-charge correlation function
  - Exciton correlation function
- Calculation of Green's function for a mean-field calculation
- Under consideration
  - Detection of a metal-insulator transition for a periodic system

## Usages

```python
from abtools.tools import AbinitioToolsClass
from abtools.utils import heatmap_ao
from pyscf import gto
from pyscf.fci import direct_spin1
import numpy as np

dist = 0.7
hydrogen = gto.M(
    atom=f"""
        H  0.000000  0.00000  0.000000
        H  0.000000  0.00000  {dist}
        H  0.000000  0.00000  {dist*2}
        H  0.000000  0.00000  {dist*3}
    """,
    basis="sto-3g",  # 基底関数系: STO-3Gを使用
    verbose=0,
)
mf_jj = AbinitioToolsClass(hydrogen)
mf_jj.run_rks()
mf_jj.calc_jj(0, 1)
```

```python
dist = 0.7
hydrogen = gto.M(
    atom = f'''
        H  0.000000  0.00000  0.000000
        H  0.000000  0.00000  {dist}
        H  0.000000  0.00000  {dist*2}
        H  0.000000  0.00000  {dist*3}
    ''',
    basis = 'sto-3g',
    verbose = 0,
)
mf_jj = AbinitioToolsClass(hydrogen)
mf_jj.run_uks()
mf_jj.calc_spin_corr(0, 1)
```

## Installation

```shell
conda create -n abtool python=3.9
conda activate abtool
git clone https://github.com/nkitamuraQC/abinitioTools.git
cd abinitioTools
pip install -e .
```
