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

