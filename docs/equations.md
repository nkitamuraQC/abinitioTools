<!-- <script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script> -->
#### Current-current correlation function
- Assumptions
  - A transfer integral between m th atomic orbital(AO) and n th AO is denoted by $t_{mn}$
  - current operator $\hat{J}$: see below of Eq.3 of https://arxiv.org/abs/1807.01625
- current-current correlation function $JJ$ is defined as below
  - $JJ = \langle \Psi | \hat{J}\hat{J} | \Psi \rangle$ 
    - ($|\Psi\rangle$ is a wavefunction of system)
  - In this program, $JJ$ is limited to a case with same time and same position (i.e. $JJ$ has only two AO indices)
 
 #### Exciton correlation
 \[
\begin{pmatrix}
A & B \\
-B^* & -A^*
\end{pmatrix}
\begin{pmatrix}
X \\
Y
\end{pmatrix}
= \omega
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
\begin{pmatrix}
X \\
Y
\end{pmatrix}
\]
- $ |\Phi_X \rangle = \sum_{mi} X_{mi}|mi\rangle$ で表される。
- $X_{mi}$を分子軌道係数でAO基底に変換しexciton相関を得る。

 #### Green's function for a mean-field calculation
 - see Eq.7 of https://arxiv.org/abs/2002.05875
   - In this program, real space not reciprocal space is treated.
