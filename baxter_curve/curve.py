"""Baxter's curve E_{Delta,Gamma} = {Q1 = Q2 = 0} in P^3 (Section 2 of the paper).

Q1 = a^2 + b^2 - c^2 - d^2 - 2 Delta (ab + cd),   Q2 = (ab - cd) - Gamma (ab + cd).
Origin O = (1:0:1:0), P_c = (0:1:1:0), crossing point c = [P_c] - [O].
"""
import sympy as sp

D, G = sp.symbols("Delta Gamma")
a, b, c, d = sp.symbols("a b c d")
p, m, P, M = sp.symbols("p m P M")

Q1 = a**2 + b**2 - c**2 - d**2 - 2*D*(a*b + c*d)
Q2 = (a*b - c*d) - G*(a*b + c*d)

# the discriminant locus: E is smooth of genus one iff DEG != 0 (Lemma 2.1)
DEG = (D**2 - 1)*(G**2 - 1)*(D**2 - G**2)
DEG_FACTORS = [D - 1, D + 1, G - 1, G + 1, D - G, D + G]

# diagonal coordinates p = a+b, m = a-b, P = c+d, M = c-d; 2Q1 = sum alpha_i x_i^2, 4Q2 = sum beta_i x_i^2
ALPHA = [1 - D, 1 + D, -(1 + D), -(1 - D)]
BETA = [1 - G, -(1 - G), -(1 + G), 1 + G]
TO_ABCD = {a: (p + m)/2, b: (p - m)/2, c: (P + M)/2, d: (P - M)/2}

O = (1, 0, 1, 0)
PC = (0, 1, 1, 0)
EIGHT = [(1, 0, 1, 0), (1, 0, -1, 0), (0, 1, 1, 0), (0, 1, -1, 0),
         (1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 0, 1), (0, 1, 0, -1)]
E2_STATED = [(1, 0, 1, 0), (1, 0, -1, 0), (0, 1, 0, 1), (0, 1, 0, -1)]   # E[2] as listed in Lemma 2.2


def iota(v):
    a_, b_, c_, d_ = v
    return (-a_, -b_, c_, d_)


def kappa(v):
    a_, b_, c_, d_ = v
    return (b_, a_, d_, c_)


def q1(Dv, v):
    a_, b_, c_, d_ = v
    return a_*a_ + b_*b_ - c_*c_ - d_*d_ - 2*Dv*(a_*b_ + c_*d_)


def q2(Gv, v):
    a_, b_, c_, d_ = v
    return (a_*b_ - c_*d_) - Gv*(a_*b_ + c_*d_)


def invariants(w):
    """(Delta, Gamma) of a weight vector (a, b, c, d)."""
    a_, b_, c_, d_ = w
    return (a_*a_ + b_*b_ - c_*c_ - d_*d_) / (2*(a_*b_ + c_*d_)), (a_*b_ - c_*d_) / (a_*b_ + c_*d_)


def couplings(w):
    """(J_x, J_y, J_z) = (ab + cd, ab - cd, (a^2 + b^2 - c^2 - d^2)/2)."""
    a_, b_, c_, d_ = w
    return a_*b_ + c_*d_, a_*b_ - c_*d_, (a_*a_ + b_*b_ - c_*c_ - d_*d_) / 2


def nondegenerate(Dv, Gv):
    return (Dv*Dv - 1)*(Gv*Gv - 1)*(Dv*Dv - Gv*Gv) != 0


def singular_points_exist(Dv, Gv):
    """Jacobian criterion at a numerical parameter value: does E have a singular point in some affine chart?"""
    sub = {D: sp.nsimplify(Dv), G: sp.nsimplify(Gv)}
    f1, f2 = Q1.subs(sub), Q2.subs(sub)
    X = [a, b, c, d]
    Jm = sp.Matrix([[sp.diff(f, v) for v in X] for f in (f1, f2)])
    minors = [Jm[:, [i, j]].det() for i in range(4) for j in range(i + 1, 4)]
    for v in X:
        eqs = [e.subs(v, 1) for e in [f1, f2] + minors]
        others = [u for u in X if u != v]
        if sp.groebner(eqs, *others, order="grevlex").exprs != [1]:
            return True
    return False
