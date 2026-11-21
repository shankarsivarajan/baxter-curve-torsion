"""The group law on E by the chord (plane) construction, over any ring with field-like operators.

E is a curve of degree 4 in P^3, so four of its points are coplanar exactly when their classes sum to the hyperplane
class H. The plane through three points P1, P2, P3 of E therefore meets E in a fourth point
F(P1, P2, P3) = H - P1 - P2 - P3, and translation by c = [A] - [C] is x -> F(F(x, A, B), C, B) for any auxiliary B.

Points are 4-tuples of ring elements, kept in a canonical projective normalization supplied by a `Field` object:
  QQ        Fraction entries, first nonzero coordinate 1
  GF(p)     python-flint nmod entries, first nonzero coordinate 1
  QQ(D,G)   sympy QQ[Delta, Gamma] polynomial entries with no common factor, first nonzero coordinate monic
  numeric   mpmath entries, scaled by the coordinate of largest modulus (compared up to a tolerance)
"""
from fractions import Fraction

import flint
import mpmath as mp
import sympy as sp

from .curve import D as DSYM, G as GSYM, O, PC, iota, kappa, EIGHT


class Field:
    def __init__(self, name, Dv, Gv, conv, normalize, eq=None):
        self.name, self.D, self.G, self.conv, self._norm = name, Dv, Gv, conv, normalize
        self.eq = eq or (lambda x, y: x == y)

    def pt(self, v):
        return self._norm(tuple(self.conv(x) for x in v))

    def norm(self, v):
        return self._norm(tuple(v))

    def same(self, x, y):
        return self.eq(x, y)


def _first_nonzero_one(v):
    for x in v:
        if x != 0:
            inv = 1 / x
            return tuple(y * inv for y in v)
    raise ZeroDivisionError("zero vector")


def QQ(Dv, Gv):
    return Field("QQ", Fraction(Dv), Fraction(Gv), Fraction, _first_nonzero_one)


def GF(p, Dv, Gv):
    conv = lambda x: flint.nmod(int(Fraction(x).numerator), p) / flint.nmod(int(Fraction(x).denominator), p)
    return Field(f"GF({p})", conv(Dv), conv(Gv), conv, _first_nonzero_one)


_R = sp.QQ[DSYM, GSYM]


def _poly_norm(v):
    g = _R.zero
    for x in v:
        g = _R.gcd(g, x)
    v = [_R.exquo(x, g) for x in v]
    for x in v:
        if x != 0:
            lc = _R.domain.quo(_R.domain.one, x.LC)
            return tuple(y * lc for y in v)
    raise ZeroDivisionError("zero vector")


def QQDG():
    """The function field Q(Delta, Gamma), points represented by coprime polynomial coordinates."""
    return Field("QQ(Delta,Gamma)", _R.convert(DSYM), _R.convert(GSYM), _R.convert, _poly_norm)


def NUM(Dv, Gv, tol=None):
    tol = tol or mp.mpf(10) ** (-(mp.mp.dps * 2 // 3))

    def norm(v):
        k = max(range(4), key=lambda i: abs(v[i]))
        return tuple(x / v[k] for x in v)

    def eq(x, y):
        return max(abs(x[i]*y[j] - x[j]*y[i]) for i in range(4) for j in range(4)) < tol

    return Field("numeric", mp.mpmathify(Dv), mp.mpmathify(Gv), mp.mpmathify, norm, eq)


def bil1(F, x, y):
    """Polar form of Q1 (up to a constant factor 1)."""
    return (x[0]*y[0] + x[1]*y[1] - x[2]*y[2] - x[3]*y[3]
            - F.D*(x[0]*y[1] + x[1]*y[0] + x[2]*y[3] + x[3]*y[2]))


def bil2(F, x, y):
    """Polar form of Q2 (up to the constant factor 2)."""
    return (1 - F.G)*(x[0]*y[1] + x[1]*y[0]) - (1 + F.G)*(x[2]*y[3] + x[3]*y[2])


def on_curve(F, v):
    return bil1(F, v, v) == 0 and bil2(F, v, v) == 0


def residual(F, P1, P2, P3):
    """Fourth point of E on the plane through P1, P2, P3 (distinct points of E), or None when the plane is tangent
    at one of them or the construction degenerates.

    A point s1 P1 + s2 P2 + s3 P3 lies on Q = 0 iff s1 s2 B(P1,P2) + s1 s3 B(P1,P3) + s2 s3 B(P2,P3) = 0. For the two
    quadrics this is a linear system in (1/s1, 1/s2, 1/s3), solved by a cross product."""
    r1, q1, p1 = bil1(F, P1, P2), bil1(F, P1, P3), bil1(F, P2, P3)
    r2, q2, p2 = bil2(F, P1, P2), bil2(F, P1, P3), bil2(F, P2, P3)
    X, Y, Z = q1*r2 - r1*q2, r1*p2 - p1*r2, p1*q2 - q1*p2
    if F.name == "numeric":
        scale = max(abs(t) for t in (r1, q1, p1, r2, q2, p2)) ** 2
        if min(abs(X), abs(Y), abs(Z)) < mp.mpf(10) ** (-mp.mp.dps // 2) * scale:
            return None
    elif X == 0 or Y == 0 or Z == 0:
        return None
    v = [Y*Z*P1[i] + X*Z*P2[i] + X*Y*P3[i] for i in range(4)]
    if all(x == 0 for x in v):
        return None
    return F.norm(v)


def crossing_pairs(F):
    """Pairs (A, C) with [A] - [C] = c: the images of (P_c, O) under 1, iota, kappa, iota kappa (translations)."""
    out = []
    for f in (lambda v: v, iota, kappa, lambda v: iota(kappa(v))):
        out.append((F.pt(f(PC)), F.pt(f(O))))
    return out


def _det3(r0, r1, r2):
    return (r0[0]*(r1[1]*r2[2] - r1[2]*r2[1]) - r0[1]*(r1[0]*r2[2] - r1[2]*r2[0])
            + r0[2]*(r1[0]*r2[1] - r1[1]*r2[0]))


def tangent_vector(F, x):
    """A vector t, not proportional to x, spanning with x the tangent line of E at x (kernel of the Jacobian)."""
    e = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    g1 = [bil1(F, x, e[i]) for i in range(4)]
    g2 = [bil2(F, x, e[i]) for i in range(4)]
    for k in range(4):
        rows = [g1, g2, e[k]]
        # generalized cross product: t_i = (-1)^i det of the 3x3 minor without column i
        t = [(-1)**i * _det3(*[[r[j] for j in range(4) if j != i] for r in rows]) for i in range(4)]
        t = [F.conv(0) + ti for ti in t]
        if any(ti != 0 for ti in t) and not F.same(F.norm(t), x):
            return t
    return None


def tangent_residual(F, x, B):
    """Fourth point H - 2x - B on the plane spanned by the tangent line at x and the point B (B != x), or None.
    On v = alpha x + beta t + gamma B, with Q(x) = Q(B) = B(x, t) = 0:
    beta^2 Q(t) + 2 alpha gamma B(x, B) + 2 beta gamma B(t, B) = 0 for both quadrics."""
    t = tangent_vector(F, x)
    if t is None:
        return None
    q1t, q2t = bil1(F, t, t), bil2(F, t, t)
    x1, x2 = bil1(F, x, B), bil2(F, x, B)
    t1, t2 = bil1(F, t, B), bil2(F, t, B)
    den = q1t*x2 - q2t*x1
    num = -2*(t1*x2 - t2*x1)
    if den == 0 or num == 0:
        return None
    beta = num/den
    if x1 != 0:
        alpha = -(beta*beta*q1t + 2*beta*t1)/(2*x1)
    elif x2 != 0:
        alpha = -(beta*beta*q2t + 2*beta*t2)/(2*x2)
    else:
        return None
    v = [alpha*x[i] + beta*t[i] + B[i] for i in range(4)]
    if all(c == 0 for c in v):
        return None
    return F.norm(v)


def add_diff(F, x, A, C, aux=None):
    """x + [A] - [C] by two plane sections, trying the auxiliary points B in turn; None if all degenerate.
    If x = A the first plane is the plane through the tangent line at x."""
    aux = aux if aux is not None else [F.pt(v) for v in EIGHT]
    for B in aux:
        if any(F.same(B, y) for y in (x, A, C)):
            continue
        y = tangent_residual(F, x, B) if F.same(x, A) else residual(F, x, A, B)   # H - x - A - B
        if y is None or F.same(y, C) or F.same(y, B):
            continue
        z = residual(F, y, C, B)                 # x + A - C
        if z is not None:
            return z
    return None


def _translate(F, x, pairs, aux):
    for A, C in sorted(pairs, key=lambda pc: F.same(x, pc[0])):     # tangent planes only as a last resort
        if F.same(x, C):
            continue
        z = add_diff(F, x, A, C, aux)
        if z is not None:
            return z
    return None


def translate(F, x, pairs=None, aux=None):
    """x + c, trying the four representations [A] - [C] of c; if all are degenerate, use that iota, kappa and
    iota kappa are translations by points of order two: x + c = g(g(x) + c) for g in {iota, kappa, iota kappa}."""
    pairs = pairs or crossing_pairs(F)
    z = _translate(F, x, pairs, aux)
    if z is not None:
        return z
    for g in (iota, kappa, lambda v: iota(kappa(v))):
        z = _translate(F, F.norm(g(x)), pairs, aux)
        if z is not None:
            return F.norm(g(z))
    return None


def add(F, x, y, aux=None):
    """x + y with origin O: F(F(x, y, B), O, B)."""
    return add_diff(F, x, y, F.pt(O), aux)


def multiples(F, kmax, start=None):
    """[start + k c for k = 1..kmax] (start defaults to O, giving the multiples k c)."""
    x = F.pt(start or O)
    out = []
    pairs = crossing_pairs(F)
    for _ in range(kmax):
        x = translate(F, x, pairs)
        if x is None:
            raise RuntimeError(f"chord construction degenerate over {F.name}")
        out.append(x)
    return out


def order_of_c(F, kmax):
    """Smallest k <= kmax with k c = O, else None."""
    o = F.pt(O)
    x, pairs = o, crossing_pairs(F)
    for k in range(1, kmax + 1):
        x = translate(F, x, pairs)
        if x is None:
            raise RuntimeError(f"chord construction degenerate over {F.name}")
        if F.same(x, o):
            return k
    return None
