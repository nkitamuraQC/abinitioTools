from pyscf import dft, scf, cc
import numpy as np


class AbinitioToolsClass:
    def __init__(self, mol):
        """
        Main class
        
        Args:
            mol (pyscf.gto.Mole): PySCF mol object.
        """
        
        self.mol = mol

    def run_rks(self, E, xc=None):
        """
        Run restricted Kohn-Sham DFT calculation

        Args:
            E (np.ndarray): electric field.

        Kwargs:
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
        self.mf = dft.RKS(self.mol)
        self.mf.xc = xc
        self.mf.get_hcore = lambda *args: h
        self.mf.kernel()
        self.mf = self.mf.to_rhf()
        return

    def run_rhf(self, E):
        """
        Run restricted Hartree-Fock calculation

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

    def run_tddft(self, nstates=10):
        """
        Run TDDFT calculation

        Kwargs:
            nstates (int): number of states
        """
        self.mytd = tddft.TDDFT(self.mf)
        self.mytd.nstates = nstates
        self.td_e, self.td_xy = self.mytd.kernel()
        self.mytd.analyze()
        return

    def calc_exciton_corr(self, target=1):
        """
        Calculates the exciton correlation function based on atomic orbitals.

        Kwargs:
            target (int): target state
            
        Returns:
            np.ndarray: The computed correlation function.
        """
        X = self.td_xy[target][0]
        Y = self.td_xy[target][1]
        mo_occ = self.mf.mo_coeff[:, :nocc]
        mo_vir = self.mf.mo_coeff[:, nocc:]
        X_ao = np.einsum("ia,mi,na->mn", X, mo_occ, mo_vir)
        return X_ao

    def calc_jj(self, site_i, site_j, calc_type="ccsd"):
        """
        Calculates the current-current correlation function between two atomic orbitals.

        Args:
            site_i (int): The index of the first atomic orbital.
            site_j (int): The index of the second atomic orbital.

        Kwargs:
            calc_type (str): calculation types ("scf" or "ccsd")
            
        Returns:
            float: The computed correlation function between site_i and site_j.
        """
        if calc_type == "scf":
            dm1 = self.mf.make_rdm1()
            dm2 = self.mf.make_rdm2()
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
        if calc_type == "ccsd":
            dm1 = self.mycc.make_rdm1()
            dm2 = self.mycc.make_rdm2()
            mo = self.mf.mo_coeff
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
            dm2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", mo, mo, mo, mo, dm2)

        ijji = dm1[site_i, site_i] + dm2[site_i, site_j, site_j, site_i]
        jiij = dm1[site_j, site_j] + dm2[site_j, site_i, site_i, site_j]
        ijij = dm2[site_i, site_j, site_i, site_j]
        jiji = dm2[site_j, site_i, site_j, site_i]

        jj = -(-ijji + jiji + ijij - jiij) * (hcore**2)
        return jj

    def calc_green(self, omega_list, eta=0.01):
        """
        Compute a Green's function
        
        Args:
            omega_list (list[float]): frequency axis.

        Returns:
            np.ndarray: The computed Green's function.
        """
        omega = np.array(omega_list)
        e = self.mf.mo_energy
        mo = self.mf.mo_coeff
        denom = omega[:, None] - e[None, :] + eta * 1.0j
        green = np.einsum("an,bn,on->abo", mo, mo, 1.0 / denom)
        return green

    def calc_chg_corr(self, site_i, site_j, calc_type="scf"):
        """
        Calculates the charge-charge correlation function between two atomic orbitals.

        Args:
            site_i (int): The index of the first atomic orbital.
            site_j (int): The index of the second atomic orbital.

        Kwargs:
            calc_type (str): calculation types ("scf" or "ccsd")
            
        Returns:
            float: The computed correlation function between site_i and site_j.
        """
        if self.dm1 is not None:
            dm1 = self.dm1
        if self.dm2 is not None:
            dm2 = self.dm2
        if calc_type == "scf":
            if self.dm1 is None:
                dm1 = self.mf.make_rdm1()
            if self.dm2 is None:
                dm2 = self.mf.make_rdm2()
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
        if calc_type == "ccsd":
            if self.dm1 is None:
                dm1 = self.mycc.make_rdm1()
            if self.dm2 is not None:
                dm2 = self.mycc.make_rdm2()
            mo = self.mf.mo_coeff
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
            dm1 = np.einsum("ai,bj,ij->ab", mo, mo, dm1)
            dm2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", mo, mo, mo, mo, dm2)
        corr = 0
        norb = dm1.shape[0]
        identity = np.identity(norb)
        if site_i == site_j:
            corr += dm1[0][site_i, site_i] + dm1[1][site_i, site_i]
        i = site_i
        j = site_j
        corr += (
            dm2[0][i, j, j, i]
            + dm2[1][i, i, j, j]
            + dm2[1][j, j, i, i]
            - dm2[2][i, j, j, i]
        )
        corr -= (dm1[0][i, i] + dm1[1][i, i]) * (dm1[0][j, j] + dm1[1][j, j])
        return corr

    def calc_spin_corr(self, site_i, site_j, calc_type="scf"):
        """
        Calculates the spin-spin correlation function between two atomic orbitals.

        Args:
            site_i (int): The index of the first atomic orbital.
            site_j (int): The index of the second atomic orbital.

        Kwargs:
            calc_type (str): calculation types ("scf" or "ccsd")
            
        Returns:
            float: The computed correlation function between site_i and site_j.
        """
        if self.dm1 is not None:
            dm1 = self.dm1
        if self.dm2 is not None:
            dm2 = self.dm2
        if calc_type == "scf":
            if self.dm1 is None:
                dm1 = self.mf.make_rdm1()
            if self.dm2 is None:
                dm2 = self.mf.make_rdm2()
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
        if calc_type == "ccsd":
            if self.dm1 is None:
                dm1 = self.mycc.make_rdm1()
            if self.dm2 is not None:
                dm2 = self.mycc.make_rdm2()
            mo = self.mf.mo_coeff
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
            dm2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", mo, mo, mo, mo, dm2)
        corr = 0
        norb = dm1.shape[0]
        delta = numpy.identity(norb, dtype=float)

        i = site_i
        j = site_j

        rdm2_aaaa[i, j] = delta[i, j] * dm1[0][j, i] - dm2[0][i, j, j, i]
        rdm2_bbbb[i, j] = delta[i, j] * dm1[1][j, i] - dm2[2][i, j, j, i]
        rdm2_abba[i, j] = -dm2[1][i, i, j, j]
        rdm_ = numpy.einsum("pqrs -> rspq", dm2[1])
        rdm_ = rdm_.conj()
        rdm2_baab[i, j] = -dm_[i, i, j, j]
        spin_corr = rdm2_aaaa + rdm2_bbbb + rdm2_abba + rdm2_baab
        return spin_corr
