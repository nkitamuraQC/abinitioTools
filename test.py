import pytest
from abtools.correlation import AbinitioToolsclass
from abtools.utils import heatmap_ao
from pyscf import gto
from pyscf.fci import direct_spin1
import numpy as np


def generate_ints(norb, U=3):
    int1e = np.zeros((norb, norb), dtype=float)
    int2e = np.zeros((norb, norb, norb, norb), dtype=float)

    for i in range(norb):
        int1e[i, (i + 1) % norb] = 1
        int1e[(i + 1) % norb, i] = 1
        int2e[i, i, i, i] = U
    return int1e, int2e


def test_spin_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints(norb)
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_spin_corr(0, 0)
    s01 = myclass.calc_spin_corr(0, 1)
    s02 = myclass.calc_spin_corr(0, 2)
    assert s00 > s01
    assert s01 > s02
    return


def test_chg_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints(norb)
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_chg_corr(0, 0)
    s01 = myclass.calc_chg_corr(0, 1)
    s02 = myclass.calc_chg_corr(0, 2)
    assert s00 > s01
    assert s01 > s02
    return


def test_cc_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints(norb)
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_jj(0, 0)
    s01 = myclass.calc_jj(0, 1)
    s02 = myclass.calc_jj(0, 2)
    assert s00 > s01
    assert s01 > s02
    return


def test_example():
    dist = 0.7
    E = 10
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

    Efield = np.array([0, 0, E])
    mf_jj = AbinitioToolsclass(hydrogen)
    mf_jj.run_dft(Efield)
    mf_jj.calc_jj(0, 1)
    return


def run_example(dist, E, to):
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

    Efield = np.array([0, 0, E])
    mf_jj = AbinitioToolsclass(hydrogen)
    mf_jj.run_dft(Efield)
    jj = mf_jj.calc_jj(0, to)
    return jj


def test_jj():
    dist = 1
    E = 0
    to = 1
    jj_101 = run_example(dist, E, to)

    dist = 2
    E = 0
    to = 1
    jj_201 = run_example(dist, E, to)

    dist = 3
    E = 0
    to = 1
    jj_301 = run_example(dist, E, to)

    assert jj_101 > jj_201
    assert jj_201 > jj_301

    dist = 1
    E = 1
    to = 1
    jj_111 = run_example(dist, E, to)

    dist = 1
    E = 2
    to = 1
    jj_121 = run_example(dist, E, to)

    assert jj_121 > jj_111
    assert jj_111 > jj_101

    dist = 1
    E = 0
    to = 2
    jj_102 = run_example(dist, E, to)

    dist = 1
    E = 0
    to = 3
    jj_103 = run_example(dist, E, to)

    assert jj_101 > jj_102
    assert jj_102 > jj_103

    return

def test_util():
    E = 0
    dist = 0.75
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

    Efield = np.array([0, 0, E])
    mf_jj = AbinitioToolsclass(hydrogen)
    mf_jj.run_dft(Efield)
    nao = len(hydrogen.ao_labels())
    jj_all = np.zeros((nao, nao))
    for i in range(nao):
        for j in range(nao):
            jj_all[i, j] = mf_jj.calc_jj(i, j)

    heatmap_ao(mf_jj.mf, jj_all)
    return


