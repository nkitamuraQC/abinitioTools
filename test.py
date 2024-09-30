import pytest
from abtools.correlation import AbinitioToolsclass
from pyscf import gto
from pyscf.fci import direct_spin1
import numpy as np

def generate_ints():
    return

def test_spin_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints()
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_spin_corr(0, 0)
    s01 = myclass.calc_spin_corr(0, 1)
    s02 = myclass.calc_spin_corr(0, 2)
    assert(s00 > s01)
    assert(s01 > s02)
    return

def test_chg_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints()
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_chg_corr(0, 0)
    s01 = myclass.calc_chg_corr(0, 1)
    s02 = myclass.calc_chg_corr(0, 2)
    assert(s00 > s01)
    assert(s01 > s02)
    return


def test_cc_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints()
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_jj(0, 0)
    s01 = myclass.calc_jj(0, 1)
    s02 = myclass.calc_jj(0, 2)
    assert(s00 > s01)
    assert(s01 > s02)
    return
   

def test_example():
    dist = 0.7
    E = 10
    hydrogen = gto.M(
        atom = f'''
            H  0.000000  0.00000  0.000000
            H  0.000000  0.00000  {dist}
            H  0.000000  0.00000  {dist*2}
            H  0.000000  0.00000  {dist*3}
        ''',
        basis = 'sto-3g',  # 基底関数系: STO-3Gを使用
        verbose = 0,
    )
        
    Efield = np.array([0, 0, E])
    mf_jj = AbinitioToolsclass(hydrogen)
    mf_jj.run_dft(Efield)
    mf_jj.calc_jj(0, 1)
    return