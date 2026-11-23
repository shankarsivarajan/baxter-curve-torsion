# Baxter's Curve Torsion

Companion code for 

> Shankar Sivarajan, *Rational points on Baxter's curve and roots of unity in the eight-vertex model*, [hexagon:2611.00791](https://hexagonmath.org/2611.00791)  (2026).

Computes the torsion loci of the crossing point $c$ over $\mathbb{Q}(\Delta, \Gamma)$ via the chord group law on $E_{\Delta,\Gamma} \subset \mathbb{P}^3$ and verifies every computation in the paper.

## Usage

```bash
pip install -e .
python reproduce.py
```

Runs 104 checks (exact over $\mathbb{Q}$, $\mathbb{Q}(\Delta, \Gamma)$, and $\mathbb{F}_p$; 40–60 digit `mpmath` for Jacobi functions and complex loci of orders 12 and 20) in under a minute. A saved run is in `reproduce.log`.

## Files

- `baxter_curve/curve.py` — quadrics $Q_1, Q_2$, diagonal pencil, base points, and involutions $\iota, \kappa$
- `baxter_curve/group.py` — plane-section group law over $\mathbb{Q}$, $\mathbb{F}_p$, $\mathbb{Q}(\Delta,\Gamma)$, and $\mathbb{C}$
- `baxter_curve/weierstrass.py` — Weierstrass model, Legendre parameter, $j$-invariant, and $\mathbb{F}_p$ point counts
- `baxter_curve/jacobi.py` — Baxter's Jacobi elliptic parametrization
- `baxter_curve/loci.py` — exact multiples $nc$, torsion loci, and rational test points
- `baxter_curve/boukraa_maillard.py` — polynomials from Boukraa–Maillard (2001) and Maillard–Boukraa (2001)
- `baxter_curve/data/order7.txt` — order-7 polynomial in $(\Delta, \Gamma)$
- `reproduce.py` — verification suite keyed to `main.tex`
