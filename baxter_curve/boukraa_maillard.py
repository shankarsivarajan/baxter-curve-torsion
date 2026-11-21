"""Polynomials transcribed from S. Boukraa and J.-M. Maillard, "Let's Baxterise", J. Stat. Phys. 102 (2001) 641-700,
arXiv:hep-th/0003212 (v1, TeX source BBplus.tex; equation numbers as in the arXiv PDF).

  J2       eq. (49), Sec. V.C: the polynomial representation of the shift doubling (J_x, J_y, J_z) -> J^(2)
           (Sec. VI.A: J^(2)_z = 0 corresponds to K^8(R) = zeta R)
  V3, V3HAT_Z
           App. B, display before eqs. (134)-(135): V^(3) (listed with K^6) and hat V_z^(3) (listed with K^3);
           V3HAT_X, V3HAT_Y: hat V_x^(3)(J_x,J_y,J_z) = hat V_z^(3)(J_y,J_z,J_x), hat V_y^(3) = hat V_z^(3)(J_z,J_x,J_y)
           (text after eq. (135))
  P3_Z     eq. (136): P_z^(3) = hat V_z^(3)(J^(2)) (listed with K^12)
  P3_X, P3_Y
           P_x^(3), P_y^(3) (listed with K^12 after eq. (136)) are not printed; App. B describes P_x P_y P_z = 0 as the
           image of hat V_x^(3) hat V_y^(3) hat V_z^(3) = 0 under the doubling, so by analogy with eq. (136) we take
           P_x^(3) = hat V_x^(3)(J^(2)) and P_y^(3) = hat V_y^(3)(J^(2))
           (confirmed: up to a constant these equal P_x^(3) of Maillard-Boukraa eq. (48) below and its cyclic
           permutations; checked in reproduce.py)
  VY5      eq. (137): V_y^(5) (listed with K^5)
  VY5HAT   eq. (138): hat V_y^(5)(J_x, J_y, J_z) = V_y^(5)(J_y, J_x, J_z) (listed with K^10)
  P5       eq. (55), Sec. VI (printed as P_5; J^(5)_x = J_x P^(5)_x by eq. (54); P^(5) listed with K^20 in App. B)
  P5_X, P5_Y, P5_Z
           P^(5)_x = P5, and by eq. (54) P^(5)_y(J_x,J_y,J_z) = P^(5)_x(J_y,J_z,J_x), P^(5)_z = P^(5)_x(J_z,J_x,J_y)
           (the three order-20 varieties of App. B; consistent with eq. (139), checked in reproduce.py)

In the normalization of the paper (O = (1:0:1:0), P_c = (0:1:1:0)), Delta = J_z/J_x and Gamma = J_y/J_x.
"""
import sympy as sp

from .loci import Jx as X, Jy as Y, Jz as Z

J2 = (-Z**2*X**2 + Z**2*Y**2 - X**2*Y**2,
      Z**2*X**2 - Z**2*Y**2 - X**2*Y**2,
      -Z**2*X**2 - Z**2*Y**2 + X**2*Y**2)

V3 = X*Y + Z*X + Z*Y
V3HAT_Z = -X*Y + Z*X + Z*Y
V3HAT_X = V3HAT_Z.subs({X: Y, Y: Z, Z: X}, simultaneous=True)
V3HAT_Y = V3HAT_Z.subs({X: Z, Y: X, Z: Y}, simultaneous=True)
P3_Z = sp.expand(V3HAT_Z.subs(dict(zip((X, Y, Z), J2)), simultaneous=True))
P3_X = sp.expand(V3HAT_X.subs(dict(zip((X, Y, Z), J2)), simultaneous=True))
P3_Y = sp.expand(V3HAT_Y.subs(dict(zip((X, Y, Z), J2)), simultaneous=True))

VY5 = (Z**2*X*Y**3 - 2*Z**2*X**2*Y**2 + Z**2*X**3*Y + Z*X**2*Y**3 - Z*X**3*Y**2 - Z**3*Y**3 - Z**3*X*Y**2
       + Z**3*X**2*Y + Z**3*X**3 - X**3*Y**3)
VY5HAT = (Z**2*X**3*Y - 2*Z**2*X**2*Y**2 + Z**2*X*Y**3 + Z*X**3*Y**2 - Z*X**2*Y**3 - Z**3*X**3 - Z**3*X**2*Y
          + Z**3*X*Y**2 + Z**3*Y**3 - X**3*Y**3)

P5 = (Z**12*X**12 + 5*Z**12*Y**12 + 6*Y**2*Z**12*X**10 + 14*Y**4*Z**10*X**10 - 29*Y**4*Z**12*X**8
      - 6*Y**2*Z**10*X**12 - 20*Y**8*Z**6*X**10 + 15*Y**8*Z**4*X**12 - 36*Y**8*Z**10*X**6 + 50*Y**8*Z**8*X**8
      - 9*Y**8*Z**12*X**4 + 15*Y**4*Z**8*X**12 - 20*Y**6*Z**8*X**10 + 4*Y**6*Z**10*X**8 + 36*Y**6*Z**12*X**6
      - 20*Y**6*Z**6*X**12 - 10*Y**10*Z**12*X**2 + 34*Y**10*Z**10*X**4 + 14*Y**10*Z**4*X**10 + 4*Y**10*Z**6*X**8
      - 6*Y**10*Z**2*X**12 - 36*Y**10*Z**8*X**6 - 9*Y**12*Z**8*X**4 - 29*Y**12*Z**4*X**8 - 10*Y**12*Z**10*X**2
      + Y**12*X**12 + 36*Y**12*Z**6*X**6 + 6*Y**12*Z**2*X**10)

P5_X = P5
P5_Y = sp.expand(P5.subs({X: Y, Y: Z, Z: X}, simultaneous=True))
P5_Z = sp.expand(P5.subs({X: Z, Y: X, Z: Y}, simultaneous=True))

SOURCE = "Boukraa-Maillard, arXiv:hep-th/0003212v1"


def dehomogenize(f):
    """f(J_x, J_y, J_z) -> f(1, Gamma, Delta)."""
    from .curve import D, G
    return sp.expand(f.subs({X: 1, Y: G, Z: D}, simultaneous=True))


# ---------------------------------------------------------------------------------------------------------------
# J.-M. Maillard and S. Boukraa, "Modular invariance in lattice statistical mechanics", Ann. Fond. Louis de Broglie
# 26 (2001), no. special, 287-328 (PDF https://www.lptmc.jussieu.fr/files/maillard/Modular_aflb26jp287.pdf),
# Secs. 5.1-5.2.2; transcribed from the printed pages 305-314.
#
#   AFLB_G2, AFLB_G3   eq. (38): g2, g3 of the elliptic curve, homogeneous in J_x, J_y, J_z
#   AFLB_J2            eq. (44): shift doubling (identical to J2 = eq. (49) of "Let's Baxterise")
#   AFLB_J3            eq. (46): multiplication of the shift by three
#   AFLB_P3_X          eq. (48): P_x^(3), with J_x^(3) = J_x P_x^(3) (form (45)); AFLB_P3_Y, AFLB_P3_Z its cyclic
#                      permutations by eq. (45); by eq. (53) (N = 3) P^(3) = 0 are conditions of order 12
#   AFLB_P5            unnumbered display after eq. (50): P_5 = P_x^(5) of the multiplication by five; by eq. (53)
#                      (N = 5) conditions of order 20
#   AFLB_T54           eq. (54): t = 2 J_x J_y J_z, "conditions of order four"
#   AFLB_T55           eq. (55): (t0, tx, ty, tz), t = t0 tx ty tz, "conditions of order six"
#   AFLB_T56           eq. (56): (t1, t2, t3, t4), t = t1 t2 t3 t4, "conditions of order ten" (multiplication by five)
# Under the multiplication by M (M = 2, 3, 5) the pair (g2, g3) becomes (t^4 g2, t^6 g3) (checked in reproduce.py;
# for M = 5 with J^(5) = (J_x P_x^(5), J_y P_y^(5), J_z P_z^(5)) from "Let's Baxterise", eqs. (54)-(55)).

AFLB_SOURCE = "Maillard-Boukraa, Ann. Fond. L. de Broglie 26 (2001) 287-328, Sec. 5.2.2"

AFLB_G2 = 12*(X**4 + Y**4 + Z**4 - (X**2*Y**2 + X**2*Z**2 + Y**2*Z**2))
AFLB_G3 = 4*(X**2 - 2*Y**2 + Z**2)*(X**2 + Y**2 - 2*Z**2)*(-2*X**2 + Y**2 + Z**2)

AFLB_J2 = (-Z**2*X**2 + Z**2*Y**2 - X**2*Y**2,
           Z**2*X**2 - Z**2*Y**2 - X**2*Y**2,
           -Z**2*X**2 - Z**2*Y**2 + X**2*Y**2)

AFLB_J3 = (X*(-2*Z**2*Y**2*X**4 - 3*Y**4*Z**4 + 2*Y**2*Z**4*X**2 + Y**4*X**4 + 2*Y**4*Z**2*X**2 + Z**4*X**4),
           Y*(2*Z**2*Y**2*X**4 - 3*Z**4*X**4 + Y**4*X**4 - 2*Y**4*Z**2*X**2 + Y**4*Z**4 + 2*Y**2*Z**4*X**2),
           Z*(Y**4*Z**4 + 2*Y**4*Z**2*X**2 - 3*Y**4*X**4 - 2*Y**2*Z**4*X**2 + 2*Z**2*Y**2*X**4 + Z**4*X**4))

AFLB_T54 = 2*X*Y*Z

AFLB_T55 = (X*Z + Y*Z + X*Y,          # t0
            X*Z + X*Y - Y*Z,          # tx
            Y*Z + X*Y - X*Z,          # ty
            X*Z + Y*Z - X*Y)          # tz

AFLB_T56 = ((Z + Y)*(-Y + Z)**2*X**3 - Y*Z*(Z + Y)**2*X**2 - Y**2*Z**2*(Z + Y)*X + Z**3*Y**3,       # t1
            (-Y + Z)*(Z + Y)**2*X**3 - Y*Z*(-Y + Z)**2*X**2 - Z**2*Y**2*(-Y + Z)*X + Z**3*Y**3,     # t2
            -(Z + Y)*(-Y + Z)**2*X**3 - Y*Z*(Z + Y)**2*X**2 + Y**2*Z**2*(Z + Y)*X + Z**3*Y**3,      # t3
            -(-Y + Z)*(Z + Y)**2*X**3 - Y*Z*(-Y + Z)**2*X**2 + Z**2*Y**2*(-Y + Z)*X + Z**3*Y**3)    # t4

AFLB_P3_X = (-2*Z**2*Y**2*X**4 - 3*Y**4*Z**4 + 2*Y**2*Z**4*X**2 + Y**4*X**4 + 2*Y**4*Z**2*X**2 + Z**4*X**4)
AFLB_P3_Y = AFLB_P3_X.subs({X: Y, Y: Z, Z: X}, simultaneous=True)
AFLB_P3_Z = AFLB_P3_X.subs({X: Z, Y: X, Z: Y}, simultaneous=True)

AFLB_P5 = (5*Z**12*Y**12 + (Z**2 - Y**2)**6*X**12
           - 10*Y**10*Z**10*(Z**2 + Y**2)*X**2 + 36*Y**6*Z**6*(Z**2 + Y**2)*(Z**2 - Y**2)**2*X**6
           - Z**4*Y**4*(29*Z**4 + 54*Z**2*Y**2 + 29*Y**4)*(Z**2 - Y**2)**2*X**8
           + Y**8*Z**8*(4*Z*Y + 3*Z**2 - 3*Y**2)*(3*Y**2 + 4*Z*Y - 3*Z**2)*X**4
           + 2*Z**2*Y**2*(Y**2 + 3*Z**2)*(3*Y**2 + Z**2)*(Z**2 + Y**2)*(Z**2 - Y**2)**2*X**10)
