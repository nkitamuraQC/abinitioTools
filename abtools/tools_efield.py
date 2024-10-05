from pyscf import dft, scf, cc, tddft, tdscf
import numpy as np


class AbinitioToolsEfield:
    def __init__(self, mol):
        """
        Main class (apply electric field)
        
        Args:
            mol (pyscf.gto.Mole): PySCF mol object.
        """
        
        self.mol = mol
        self.dm1 = None
        self.dm2 = None
        self.mf = None
        self.mytd = None
        self.mycc = None

        self.hcore = None
        

    def run_rks(self, E, xc="b3lyp"):
        """
        Run restricted Kohn-Sham DFT calculation

        Args:
            E (np.ndarray): electric field
            xc (str): XC functional
        """
        mol = self.mol
        mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h = (
            mol.intor("cint1e_kin_sph")
            + mol.intor("cint1e_nuc_sph")
            + np.einsum("x,xij->ij", E, mol.intor("cint1e_r_sph", comp=3))
        )
        self.mol = mol
        self.mfks = dft.RKS(self.mol)
        self.mfks.xc = xc
        self.mfks.get_hcore = lambda *args: h
        self.mfks.kernel()
        self.mf = self.mfks.to_rhf()
        return

    
    def run_rhf(self, E):
        """
        Run restricted Hartree-Fock calculation

        """
        mol = self.mol
        mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h = (
            mol.intor("cint1e_kin_sph")
            + mol.intor("cint1e_nuc_sph")
            + np.einsum("x,xij->ij", E, mol.intor("cint1e_r_sph", comp=3))
        )
        self.mol = mol
        self.mf = scf.RHF(self.mol)
        self.mf.get_hcore = lambda *args: h
        self.mf.kernel()
        return


    def run_ccsd(self, E):
        """
        Run CCSD calculation

        Args:
            E (np.ndarray): electric field.
        """
        mol = self.mol
        mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h = (
            mol.intor("cint1e_kin_sph")
            + mol.intor("cint1e_nuc_sph")
            + np.einsum("x,xij->ij", E, mol.intor("cint1e_r_sph", comp=3))
        )
        self.mol = mol
        self.mf = scf.RHF(self.mol)
        self.mf.get_hcore = lambda *args: h
        self.mf.kernel()
        self.mycc = cc.CCSD(self.mf)
        self.mycc.kernel()
        return
