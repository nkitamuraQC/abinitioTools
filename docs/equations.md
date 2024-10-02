#### Current-current correlation function
- Assumptions
  - A transfer integral between $m$th atomic orbital(AO) and $n$th AO is denoted by $t_{mn}$
  - current operator:
 $\hat{J}=i\sum_{mn\sigma}t_{mn}(\hat{a}^{\dagger}_{m\sigma}\hat{a}_{n\sigma}-\mathrm{H.c.}
- current-current correlation function $JJ$ is defined as below
  - $JJ = \langle \Psi | \hat{J}\hat{J} | \Psi \rangle$ 
    - ($|\Psi\rangle$ is a wavefunction of system)
  - In this program, $JJ$ is limited to a case with same time and same position (i.e. $JJ$ has only two AO indices)

 #### Green's function for a mean-field calculation
 - see Eq.7 of https://arxiv.org/abs/2002.05875
   - In this program, real space not reciprocal space is treated.
