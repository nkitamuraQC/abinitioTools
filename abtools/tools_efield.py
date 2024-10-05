from pyscf import dft, scf, cc, tddft, tdscf
import numpy as np


class AbinitioToolsClass:
    def __init__(self, mol):
        """
        Main class
        
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

    def _init_dms(self, calc_type):
        """
        DMs are cached when those are calculated.
        """
        if self.dm1 is not None:
            dm1 = self.dm1
        if self.dm2 is not None:
            dm2 = self.dm2
        if calc_type == "scf":
            if self.dm1 is None:
                dm1 = self.mf.make_rdm1()
                self.dm1 = dm1
            if self.dm2 is None:
                dm2 = self.mf.make_rdm2()
                self.dm2 = dm2
        if calc_type == "ccsd":
            if self.dm1 is None:
                mo = self.mf.mo_coeff
                dm1 = self.mycc.make_rdm1()
                dm1 = np.einsum("ai,bj,ij->ab", mo, mo, dm1)
                self.dm1 = dm1
            if self.dm2 is not None:
                mo = self.mf.mo_coeff
                dm2 = self.mycc.make_rdm2()
                dm2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", mo, mo, mo, mo, dm2)
                self.dm2 = dm2
        return dm1, dm2
        

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

    def run_uks(self, xc="b3lyp"):
        """
        Run unrestricted Kohn-Sham DFT calculation

        Args:
            E (np.ndarray): electric field
            xc (str): XC functional
        """
        self.mfks = dft.UKS(self.mol)
        self.mfks.xc = xc
        self.mfks.kernel()
        self.mf = self.mfks.to_uhf()
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
    
    def run_uhf(self):
        """
        Run unrestricted Hartree-Fock calculation

        """
        self.mf = scf.UHF(self.mol)
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

        Args:
            nstates (int): number of states
        """
        if isinstance(self.mf, dft.UKS):
            raise NotImplementedError
        self.mytd = tddft.TDDFT(self.mfks)
        self.mytd.nstates = nstates
        self.td_e, self.td_xy = self.mytd.kernel()
        self.mytd.analyze()
        return
    
    def run_tdscf(self, nstates=10):
        """
        Run TDDFT calculation

        Args:
            nstates (int): number of states
        """
        if isinstance(self.mf, scf.UHF):
            raise NotImplementedError
        self.mytd = tdscf.TDHF(self.mf)
        self.mytd.nstates = nstates
        self.td_e, self.td_xy = self.mytd.kernel()
        self.mytd.analyze()
        return


    def calc_exciton_corr(self, target=1):
        """
        Calculates the exciton correlation function based on atomic orbitals.

        Args:
            target (int): target state
            
        Returns:
            np.ndarray: The computed correlation function.
        """
        nocc = self.mol.nelectron // 2
        X = self.td_xy[target][0]
        Y = self.td_xy[target][1]
        mo_occ = self.mf.mo_coeff[:, :nocc]
        mo_vir = self.mf.mo_coeff[:, nocc:]
        X_ao = np.einsum("ia,mi,na->mn", X, mo_occ, mo_vir)
        return X_ao

    def calc_jj(self, site_i, site_j, calc_type="scf"):
        """
        Calculates the current-current correlation function between two atomic orbitals.

        Args:
            site_i (int): The index of the first atomic orbital.
            site_j (int): The index of the second atomic orbital.
            calc_type (str): calculation types ("scf" or "ccsd")
            
        Returns:
            float: The computed correlation function between site_i and site_j.
        """
        if self.hcore is None:
            hcore = self.mol.intor("cint1e_kin_sph") + self.mol.intor("cint1e_nuc_sph")
            hcore = hcore[site_i, site_j]
        else:
            hcore = self.hcore[site_i, site_j]
        dm1, dm2 = self._init_dms(calc_type)

        if isinstance(dm1, list) or isinstance(dm2, list):
            raise NotImplementedError

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
            calc_type (str): calculation types ("scf" or "ccsd")
            
        Returns:
            float: The computed correlation function between site_i and site_j.
        """
        dm1, dm2 = self._init_dms(calc_type)
        corr = 0
        if site_i == site_j:
            corr += dm1[0][site_i, site_i] + dm1[1][site_i, site_i]
        i = site_i
        j = site_j
        corr += (
            - dm2[0][i, j, j, i]
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
            calc_type (str): calculation types ("scf" or "ccsd")
            
        Returns:
            float: The computed correlation function between site_i and site_j.
        """
        dm1, dm2 = self._init_dms(calc_type)
        norb = dm1[0].shape[0]
        delta = np.identity(norb, dtype=float)

        i = site_i
        j = site_j

        rdm2_aaaa = delta[i, j] * dm1[0][j, i] - dm2[0][i, j, j, i]
        rdm2_bbbb = delta[i, j] * dm1[1][j, i] - dm2[2][i, j, j, i]
        rdm2_abba = -dm2[1][i, i, j, j]
        rdm_ = np.einsum("pqrs -> rspq", dm2[1])
        rdm_ = rdm_.conj()
        rdm2_baab = -rdm_[i, i, j, j]
        spin_corr = rdm2_aaaa + rdm2_bbbb + rdm2_abba + rdm2_baab
        return spin_corr
