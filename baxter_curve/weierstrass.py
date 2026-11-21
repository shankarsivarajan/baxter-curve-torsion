"""Weierstrass model of E (Remark 2.3) and point counts over finite fields."""
import itertools

import sympy as sp

from .curve import D, G, ALPHA, BETA

lam, x, y = sp.symbols("lambda x y")

T = [(D - 1)/(G - 1), (D + 1)/(G - 1), (D + 1)/(G + 1), (D - 1)/(G + 1)]
E_STATED = [-4*(D + G)/((G - 1)*(G + 1)**2),
            -4*(D - 1)/((G - 1)*(G + 1)**2),
            -4*(D - 1)*(D + G)/((G - 1)**2*(G + 1)**2)]
LEGENDRE_STATED = (D**2 - G**2)/(1 - G**2)
J_STATED = 256*(D**4 - D**2*G**2 - D**2 + G**4 - G**2 + 1)**3 / ((D**2 - 1)**2*(G**2 - 1)**2*(D**2 - G**2)**2)
W = (G - 1)*(G + 1)          # the substitution x -> x / W^2 of the remark


def e_from_t():
    return [sp.factor(-sp.prod([T[3] - T[j] for j in range(3) if j != i])) for i in range(3)]


def quartic():
    """det(lambda A + B), A, B the (diagonal) Gram matrices of 2Q1 and 4Q2 in (p, m, P, M)."""
    A, B = sp.diag(*ALPHA), sp.diag(*BETA)
    return sp.factor((lam*A + B).det())


def quartic_roots():
    return [sp.factor(-bt/al) for al, bt in zip(ALPHA, BETA)]


def cubic_from_root(r):
    """Send the root lambda = r of the quartic to infinity: lambda = r + 1/x, y -> y / x^2.
    Returns the coefficients (c3, c2, c1, c0) of y^2 = c3 x^3 + c2 x^2 + c1 x + c0."""
    q = quartic()
    cub = sp.expand(sp.cancel(x**4 * q.subs(lam, r + 1/x)))
    P = sp.Poly(cub, x)
    assert P.degree() == 3
    return [sp.factor(cf) for cf in P.all_coeffs()]


def monic(coeffs):
    """(X, Y) = (c3 x, c3 y): Y^2 = X^3 + c2 X^2 + c1 c3 X + c0 c3^2."""
    c3, c2, c1, c0 = coeffs
    return [sp.Integer(1), c2, sp.factor(c1*c3), sp.factor(c0*c3**2)]


def matches_stated(coeffs):
    """Is the monic cubic Y^2 = f(X) carried to y^2 = prod (x - e_i) by X = W^2 (x - s), Y = W^3 y, for a
    translation s in Q(Delta, Gamma)? Returns (ok, s)."""
    one, c2, c1, c0 = monic(coeffs)
    f = lambda X: X**3 + c2*X**2 + c1*X + c0
    target = sp.expand(sp.prod([x - e for e in E_STATED]))
    # match the x^2 coefficient: W^-6 f(W^2 (x - s)) has x^2 coefficient -3 s + c2 / W^2
    s = sp.factor((sp.Poly(target, x).all_coeffs()[1] - c2/W**2) / -3)
    lhs = sp.together(f(W**2*(x - s)) / W**6)
    ok = sp.simplify(sp.numer(sp.together(lhs - target))) == 0
    return ok, s


def legendre(es):
    return sp.factor((es[2] - es[0])/(es[1] - es[0]))


def j_from_cubic(a2, a4, a6):
    """j-invariant of y^2 = x^3 + a2 x^2 + a4 x + a6 from the standard b- and c-invariants."""
    b2, b4, b6, b8 = 4*a2, 2*a4, 4*a6, 4*a2*a6 - a4**2
    c4 = sp.cancel(b2**2 - 24*b4)
    disc = sp.cancel(-b2**2*b8 - 8*b4**3 - 27*b6**2 + 9*b2*b4*b6)
    return sp.cancel(c4**3/disc)


def _mod(r, p):
    r = sp.Rational(r)
    return int(r.p) * pow(int(r.q), -1, p) % p


def count_weierstrass(es, p):
    """#{y^2 = (x - e1)(x - e2)(x - e3)}(F_p) including the point at infinity, es rational."""
    ev = [_mod(e, p) for e in es]
    sq = [0]*p
    for t in range(p):
        sq[t*t % p] += 1
    return 1 + sum(sq[(X - ev[0])*(X - ev[1])*(X - ev[2]) % p] for X in range(p))


def count_quadrics(Dv, Gv, p):
    """#E(F_p) for the intersection of quadrics in P^3, by enumeration of normalized points."""
    Dm, Gm = _mod(Dv, p), _mod(Gv, p)
    n = 0
    for lead in range(4):
        for rest in itertools.product(range(p), repeat=3 - lead):
            v = (0,)*lead + (1,) + rest
            A, B, C, Dd = v
            if (A*A + B*B - C*C - Dd*Dd - 2*Dm*(A*B + C*Dd)) % p == 0 and ((A*B - C*Dd) - Gm*(A*B + C*Dd)) % p == 0:
                n += 1
    return n
