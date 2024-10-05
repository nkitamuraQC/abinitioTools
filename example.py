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
mf_jj._init_dms("scf")
print(mf_jj.dm1)
mf_jj.calc_spin_corr(0, 1)