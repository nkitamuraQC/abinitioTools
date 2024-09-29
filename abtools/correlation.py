from pyscf import dft, scf, cc
import numpy as np


class AbinitioToolsClass:
    def __init__(self, mol):
        self.mol = mol

    def run_dft(self, E):
        mol = self.mol
        mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
          + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
        self.mol = mol
        self.mf = dft.RKS(self.mol)
        self.mf.get_hcore = lambda *args: h
        self.mf.kernel()
        self.mf = self.mf.to_rhf()
        return
    
    def run_hf(self, E):
        mol = self.mol
        mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
          + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
        self.mol = mol
        self.mf = scf.RHF(self.mol)
        self.mf.get_hcore = lambda *args: h
        self.mf.kernel()
        return
    
    def run_ccsd(self, E):
        mol = self.mol
        mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
          + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
        self.mol = mol
        self.mf = scf.RHF(self.mol)
        self.mf.get_hcore = lambda *args: h
        self.mf.kernel()
        self.mycc = cc.CCSD(self.mf)
        self.mycc.kernel()
        return


    def calc_jj(self, site_i, site_j, calc_type="ccsd"):
        if calc_type == "scf":
            dm1 = self.mf.make_rdm1()
            dm2 = self.mf.make_rdm2()
            hcore = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
            hcore = hcore[site_i, site_j]
        if calc_type == "ccsd":
            dm1 = self.mycc.make_rdm1()
            dm2 = self.mycc.make_rdm2()
            mo = self.mf.mo_coeff
            hcore = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
            hcore = hcore[site_i, site_j]
            dm2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", mo, mo, mo, mo, dm2)

        ijji = dm1[site_i, site_i] + dm2[site_i, site_j, site_j, site_i] 
        jiij = dm1[site_j, site_j] + dm2[site_j, site_i, site_i, site_j] 
        ijij = dm2[site_i, site_j, site_i, site_j] 
        jiji = dm2[site_j, site_i, site_j, site_i]

        jj = - (- ijji + jiji + ijij - jiij) * (hcore ** 2)
        return jj
    
    
    def calc_green(self, omega_list, eta=0.01):
        omega = np.array(omega_list)
        e = self.mf.mo_energy
        mo = self.mf.mo_coeff
        denom = omega[:, None] - e[None, :] + eta*1.0j
        green = np.einsum("an,bn,on->abo", mo, mo, 1./denom)
        return green

　　　def calc_chg_corr(self, site_i, site_j):
        if self.dm1 is not None:
            dm1 = self.dm1
        if self.dm2 is not None:
            dm2 = self.dm2
        if calc_type == "scf":
            if self.dm1 is None:
                dm1 = self.mf.make_rdm1()
            if self.dm2 is None:
                dm2 = self.mf.make_rdm2()
            hcore = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
            hcore = hcore[site_i, site_j]
        if calc_type == "ccsd":
            if self.dm1 is None:
                dm1 = self.mycc.make_rdm1()
            if self.dm2 is not None:
                dm2 = self.mycc.make_rdm2()
            mo = self.mf.mo_coeff
            hcore = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
            hcore = hcore[site_i, site_j]
            dm2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", mo, mo, mo, mo, dm2)
        corr = 0
        norb = dm1.shape[0]
        identity = np.identity(norb)
        if site_i == site_j:
            corr += dm1[0][site_i, site_i] + dm1[1][site_i, site_i]
        i = site_i
        j = site_j
        corr += dm2[0][i,j,j,i] + dm2[1][i,i,j,j] + dm2[1][j,j,i,i] - dm2[2][i,j,j,i]
        corr -= (dm1[0][i,i] + dm1[1][i,i]) * (dm1[0][j,j] + dm1[1][j,j])
        return corr

def calc_spin_corr(fci_vec,norb,nelec):
    delta = numpy.identity(norb,dtype=float)
    rdm1_,rdm2_ = fci.direct_spin1.make_rdm12s(fci_vec,norb, nelec)
    rdm1,rdm2 = [],[]
    for i in range(len(rdm1_)):
        rdm1.append(rdm1_[i])
    for i in range(len(rdm2_)):
        rdm2.append(rdm2_[i])
    rdm2_aaaa = numpy.zeros((norb,norb))
    rdm2_bbbb = numpy.zeros((norb,norb))
    rdm2_abba = numpy.zeros((norb,norb))
    rdm2_baab = numpy.zeros((norb,norb))
    rdm2_diagonal = numpy.zeros((norb,norb))
    for i in range(norb):
        for j in range(norb):
            #rdm2_diagonal[i,j] = rdm2[0][i,j,j,i] + rdm2[2][i,j,j,i] + rdm2[1][i,j,j,i]
            #rdm_ = rdm2[1].T.conj()
            #rdm2_diagonal[i,j] += rdm_[i,j,j,i]
            rdm2_aaaa[i,j] = delta[i,j]*rdm1[0][j,i] - rdm2[0][i,j,j,i]
            rdm2_bbbb[i,j] = delta[i,j]*rdm1[1][j,i] - rdm2[2][i,j,j,i]
            rdm2_abba[i,j] = -rdm2[1][i,i,j,j]
            rdm_ = numpy.einsum("pqrs -> rspq",rdm2[1])
            rdm_ = rdm_.conj()
            rdm2_baab[i,j] = -rdm_[i,i,j,j]
    spin_corr = rdm2_aaaa + rdm2_bbbb +  rdm2_abba + rdm2_baab
    #spin_corr = rdm2_diagonal
    return spin_corr,rdm1,rdm2

