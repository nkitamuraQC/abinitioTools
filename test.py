import pytest
from abtools.tools import AbinitioToolsClass
from abtools.utils import heatmap_ao
from matplotlib import pyplot as plt
from pyscf import gto
from pyscf.fci import direct_spin1
import numpy as np


def generate_ints(norb, U=30, V=10):
    int1e = np.zeros((norb, norb), dtype=float)
    int2e = np.zeros((norb, norb, norb, norb), dtype=float)

    for i in range(norb):
        int1e[i, (i + 1) % norb] = 1
        int1e[(i + 1) % norb, i] = 1
        int2e[i, i, i, i] = U
        int2e[i, i, (i+1)%norb, (i+1)%norb] = V
        int2e[(i+1)%norb, (i+1)%norb, i, i] = V
    return int1e, int2e


def test_spin_corr():
    norb = 12
    nelec = 18
    mol = gto.Mole()
    myclass = AbinitioToolsClass(mol)
    int1e, int2e = generate_ints(norb)
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_spin_corr(0, 0)
    s01 = myclass.calc_spin_corr(0, 1)
    s02 = myclass.calc_spin_corr(0, 2)
    s03 = myclass.calc_spin_corr(0, 3)
    print(s00, s01, s02, s03)
    assert s00 > s01
    assert s01 > s02
    return


def test_chg_corr():
    norb = 12
    nelec = 18
    mol = gto.Mole()
    myclass = AbinitioToolsClass(mol)
    int1e, int2e = generate_ints(norb)
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    c00 = myclass.calc_chg_corr(0, 0)
    c01 = myclass.calc_chg_corr(0, 1)
    c02 = myclass.calc_chg_corr(0, 2)
    c03 = myclass.calc_chg_corr(0, 3)
    print(c00, c01, c02, c03)
    assert abs(c00) > abs(c01)
    assert abs(c01) > abs(c02)
    return


def test_cc_corr():
    norb = 12
    nelec = 18
    mol = gto.Mole()
    myclass = AbinitioToolsClass(mol)
    int1e, int2e = generate_ints(norb)
    myclass.hcore = int1e
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    jj01 = myclass.calc_jj(0, 1)
    jj02 = myclass.calc_jj(0, 2)
    jj03 = myclass.calc_jj(0, 3)
    print(jj01, jj02, jj03)
    assert jj01 >= jj02
    assert jj02 >= jj03
    return


def test_example():
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
    mf_jj.calc_spin_corr(0, 1)
    return


def run_example(dist, frm, to):
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
    jj = mf_jj.calc_jj(frm, to)
    return jj


def test_jj():
    dist = 1
    frm = 1
    to = 2
    jj_1 = run_example(dist, frm, to)

    dist = 2
    frm = 1
    to = 2
    jj_2 = run_example(dist, frm, to)

    dist = 3
    frm = 1
    to = 2
    jj_3 = run_example(dist, frm, to)

    assert jj_1 > jj_2
    assert jj_2 > jj_3
    print("test_jj1:", jj_1, jj_2, jj_3)

    dist = 1
    frm = 0
    to = 3
    jj_4 = run_example(dist, frm, to)

    dist = 1
    frm = 1
    to = 2
    jj_5 = run_example(dist, frm, to)

    assert jj_5 > jj_4
    print("test_jj2:", jj_4, jj_5)

    #dist = 1
    #E = 1
    #frm = 0
    #to = 2
    #jj_6 = run_example(dist, E, frm, to)

    #dist = 1
    #E = 2
    #frm = 0
    #to = 2
    #jj_7 = run_example(dist, E, frm, to)

    #assert jj_7 > jj_6
    #print("test_jj3:", jj_6, jj_7)

    return

def test_util():
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

    mf_jj = AbinitioToolsClass(hydrogen)
    mf_jj.run_rks()
    nao = len(hydrogen.ao_labels())
    jj_all = np.zeros((nao, nao))
    for i in range(nao):
        for j in range(nao):
            jj_all[i, j] = mf_jj.calc_jj(i, j)

    heatmap_ao(mf_jj.mf, jj_all, path="heatmap_ao_jj.png")
    return

def test_green():
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

    mf_jj = AbinitioToolsClass(hydrogen)
    mf_jj.run_rks()
    moe = mf_jj.mf.mo_energy
    
    omega = np.linspace(-2, 2, 400)
    green = mf_jj.calc_green(omega_list=omega)
    dos = np.trace(green, axis1=0, axis2=1).imag
    plt.cla()
    plt.plot(omega, dos)
    for a in moe:
        plt.axvline(x=a, color='r')
    plt.savefig("./green.png")
    
    return


def test_exciton():
    dist = 0.75
    hydrogen = gto.M(
        atom=f"""
            H  0.000000  0.00000  0.000000
            Li  0.000000  0.00000  {dist}
        """,
        basis="sto-3g",  # 基底関数系: STO-3Gを使用
        verbose=0,
    )

    mf_jj = AbinitioToolsClass(hydrogen)
    mf_jj.run_rks()
    mf_jj.run_tddft()
    mf_jj.calc_exciton_corr()
    return


