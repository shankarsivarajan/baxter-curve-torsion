"""Baxter's elliptic parametrization (Lemma 2.4).

w(u) = (sn(lambda - u) : sn u : sn lambda : k sn lambda sn u sn(lambda - u)),  modulus k (mpmath parameter m = k^2),
Delta = -cn(lambda) dn(lambda) / (1 + k sn^2 lambda),  Gamma = (1 - k sn^2 lambda) / (1 + k sn^2 lambda).
"""
import mpmath as mp
import sympy as sp


def sn(z, k):
    return mp.ellipfun("sn", z, m=k**2)


def cn(z, k):
    return mp.ellipfun("cn", z, m=k**2)


def dn(z, k):
    return mp.ellipfun("dn", z, m=k**2)


def periods(k):
    """(K, K') for modulus k."""
    return mp.ellipk(k**2), mp.ellipk(1 - k**2)


def w(u, lamv, k):
    s1, s2, s3 = sn(lamv - u, k), sn(u, k), sn(lamv, k)
    return [s1, s2, s3, k*s3*s2*s1]


def invariants_stated(lamv, k):
    S = sn(lamv, k)
    return -cn(lamv, k)*dn(lamv, k)/(1 + k*S**2), (1 - k*S**2)/(1 + k*S**2)


def symbolic_reduction():
    """Reduce the numerators of Q1, Q2 at the parametrized weights modulo the Jacobi relations
    cn^2 = 1 - sn^2, dn^2 = 1 - k^2 sn^2 (for u and for lambda), using the addition formula for sn(lambda - u).
    Returns (remainder of Q2, remainder of Q1, (ab - cd) - stated, (ab + cd) - stated)."""
    s, cu, du, S, C, Dn, k = sp.symbols("s cu du S C Dn k")
    a = (S*cu*du - s*C*Dn)/(1 - k**2*S**2*s**2)
    b, c = s, S
    d = k*S*s*a
    Gam = (1 - k*S**2)/(1 + k*S**2)
    Del = -C*Dn/(1 + k*S**2)
    GB = sp.groebner([cu**2 - (1 - s**2), du**2 - (1 - k**2*s**2), C**2 - (1 - S**2), Dn**2 - (1 - k**2*S**2)],
                     cu, du, C, Dn, s, S, k, order="lex")

    def red(expr):
        return GB.reduce(sp.expand(sp.numer(sp.together(expr))))[1]

    Q2 = (a*b - c*d) - Gam*(a*b + c*d)
    Q1 = a**2 + b**2 - c**2 - d**2 - 2*Del*(a*b + c*d)
    proof_line = (sp.simplify(a*b - c*d - s*a*(1 - k*S**2)), sp.simplify(a*b + c*d - s*a*(1 + k*S**2)))
    return red(Q2), red(Q1), proof_line
