"""Torsion loci of the crossing point (Theorem 3.1, Remarks 3.2 and 3.3).

The multiples k c are computed over the function field Q(Delta, Gamma) by the chord construction; k c = O along a
curve exactly where gcd(b_k, d_k, a_k - c_k) vanishes, O = (1:0:1:0).
"""
from fractions import Fraction

import sympy as sp

from .curve import D, G, DEG_FACTORS
from .group import QQDG, multiples, _R

# Theorem 3.1(3), as stated (polynomials whose vanishing gives order N)
STATED = {
    3: [D - G + D*G],
    4: [D, G],                                     # Delta Gamma = 0
    6: [D + G + D*G, D - G - D*G, D + G - D*G],
    8: [D**2 + G**2 - D**2*G**2, G**2 - D**2 - D**2*G**2, G**2 - D**2 + D**2*G**2],
}

# the factor lists of the proof of Theorem 3.1(3), as stated (k = 1..8)
PROOF_FACTORS = {
    1: [], 2: [],
    3: [D*G + D - G],
    4: [D, G],
    6: [D*G + D + G, D*G + D - G, D*G - D + G, D*G - D - G],
    8: [D, G, D**2*G**2 - D**2 - G**2, D**2*G**2 - (D**2 - G**2), D**2*G**2 + (D**2 - G**2)],
}

ORDER5_STATED = (D**3*G**3 - D**3*G**2 - D**3*G + D**3 + D**2*G**3 + 2*D**2*G**2 + D**2*G - D*G**3 - D*G**2 - G**3)

# Remark 3.2, the order-7 locus as printed
ORDER7_STATED = ((G - 1)**2*(G + 1)**4*D**6 - 2*G*(G - 1)**2*(G + 1)*(G**2 + 1)*D**5 - G**2*(G - 1)**2*(G + 1)**2*D**4
                 + 4*G**3*(G + 1)*(G**2 - G + 1)*D**3 - G**4*(G + 1)**2*D**2 - 2*G**5*(G + 1)*D + G**6)

# Theorem 3.1 coupling form (l.165): Delta = Jz/Jx, Gamma = Jy/Jx
Jx, Jy, Jz = sp.symbols("J_x J_y J_z")
STATED_J = {
    3: [-Jx*Jy + Jx*Jz + Jy*Jz],
    4: [Jy*Jz],
    6: [Jx*Jy + Jx*Jz + Jy*Jz, Jx*Jy - Jx*Jz + Jy*Jz, -Jx*Jy - Jx*Jz + Jy*Jz],
    8: [Jx**2*(Jy**2 + Jz**2) - Jy**2*Jz**2, Jx**2*(Jy**2 - Jz**2) - Jy**2*Jz**2, Jx**2*(Jy**2 - Jz**2) + Jy**2*Jz**2],
}


def same_up_to_unit(f, g):
    """f = +- g (or rational multiple)."""
    q = sp.cancel(sp.sympify(f)/sp.sympify(g))
    return q.is_number and q != 0


def to_J(f):
    """Homogenize f(Delta, Gamma) with Delta = Jz/Jx, Gamma = Jy/Jx; returns the numerator."""
    return sp.factor(sp.numer(sp.together(f.subs({D: Jz/Jx, G: Jy/Jx}))))


def symbolic_multiples(kmax=8, log=None):
    """[(k, coordinates of k c over QQ[Delta, Gamma], loci factors in U, all factors of the gcd, common factor)]."""
    F = QQDG()
    out = []
    pts = multiples(F, kmax)
    for k, xk in enumerate(pts, start=1):
        ak, bk, ck, dk = xk
        g = _R.gcd(_R.gcd(bk, dk), ak - ck)
        gall = _R.gcd(_R.gcd(ak, bk), _R.gcd(ck, dk))
        allf = [f for f, e in sp.factor_list(_R.to_sympy(g))[1]]
        inU = [f for f in allf if not any(same_up_to_unit(f, h) for h in DEG_FACTORS)]
        deg = max(sp.Poly(_R.to_sympy(t), D, G).total_degree() for t in xk)
        out.append(dict(k=k, point=xk, loci=inU, gcd_factors=allf, common=_R.to_sympy(gall), degree=deg))
        if log:
            log(f"  n = {k}: coordinates of total degree {deg}; {len(inU)} locus factor(s) in U")
    return out


def factor_sets_equal(found, stated):
    if len(found) != len(stated):
        return False
    used = set()
    for f in found:
        hit = [i for i, s in enumerate(stated) if i not in used and same_up_to_unit(f, s)]
        if not hit:
            return False
        used.add(hit[0])
    return True


# rational points on each locus (parametrized), used for spot checks of the exact order
def locus_points(N, which, ts):
    out = []
    for t in ts:
        t = Fraction(t)
        try:
            if N == 3:
                pt = (t, t/(1 - t))                            # Delta - Gamma + Delta Gamma = 0
            elif N == 4:
                pt = (Fraction(0), t) if which == 0 else (t, Fraction(0))
            elif N == 6:
                pt = [(t, -t/(1 + t)), (t, t/(1 + t)), (t, t/(t - 1))][which]
            elif N == 8:
                pt = [((t*t + 1)/(2*t), (t*t + 1)/(t*t - 1)),        # Delta^2 + Gamma^2 = Delta^2 Gamma^2
                      ((1 - t*t)/(1 + t*t), (1 - t*t)/(2*t)),        # Gamma^2 - Delta^2 = Delta^2 Gamma^2
                      ((t*t - 1)/(2*t), (t*t - 1)/(t*t + 1))][which]  # Gamma^2 - Delta^2 = -Delta^2 Gamma^2
        except ZeroDivisionError:
            continue
        out.append(pt)
    return out


def on_any_locus(Dv, Gv):
    return any(sp.sympify(f).subs({D: sp.Rational(Dv), G: sp.Rational(Gv)}) == 0
               for fs in STATED.values() for f in fs)


def meets_only_outside_U(g, h):
    """True if every common zero (Delta, Gamma) in C^2 of g and of the locus h (linear in Gamma, or h = Delta) lies on
    (Delta^2-1)(Gamma^2-1)(Delta^2-Gamma^2) = 0. Exact: h is solved for one variable, var = r(other) (this is the
    resultant of g and h in var up to a power of the leading coefficient of h); every irreducible factor of the
    numerator of g(var = r) must divide the numerator of DEG(var = r), and h has no zero where the denominator of r
    vanishes. Returns False if h is a component of g."""
    from .curve import DEG
    g, h = sp.expand(g), sp.expand(h)
    var = G if sp.degree(h, G) == 1 else D
    other = D if var == G else G
    assert sp.degree(h, var) == 1
    sol = sp.solve(h, var)[0]
    den = sp.denom(sp.together(sol))
    for r in sp.Poly(den, other).all_roots() if sp.degree(den, other) > 0 else []:
        hr = sp.Poly(h.subs(other, r), var)
        if hr.degree() > 0 or hr.is_zero:      # h has a point with other = r: not covered by var = r(other)
            return False
    N = sp.numer(sp.together(g.subs(var, sol)))
    if sp.expand(N) == 0:
        return False
    dg = sp.numer(sp.together(DEG.subs(var, sol)))
    return all(sp.rem(dg, q, other) == 0 for q, _ in sp.factor_list(N)[1] if sp.degree(q, other) > 0)


DATA = __import__("pathlib").Path(__file__).resolve().parent / "data"


def save_order7(poly):
    """Write the order-7 polynomial (recomputed by symbolic_multiples) to data/order7.txt; if the file exists, it must
    agree with the recomputation. Returns the path."""
    DATA.mkdir(exist_ok=True)
    path = DATA / "order7.txt"
    text = str(sp.expand(poly)) + "\n"
    if path.exists():
        old = sp.sympify(path.read_text(), locals={"Delta": D, "Gamma": G})
        if not same_up_to_unit(old, poly):
            raise ValueError("data/order7.txt disagrees with the recomputed order-7 locus")
    else:
        path.write_text(text)
    return path.relative_to(DATA.parent.parent)


def numeric_orders(fJ, deltas, kmax=24, roots_per_delta=2, dps=60):
    """Order of c at complex points of the curve fJ(1, Gamma, Delta) = 0 (numerically, chord construction in
    `dps` digits): for each Delta in `deltas`, solve for Gamma and iterate the translation by c up to kmax times.
    Returns the list of orders found (None if > kmax, 'degenerate' if the construction failed)."""
    import mpmath as mp
    from .group import NUM, crossing_pairs, translate
    from .curve import O, nondegenerate
    with mp.workdps(dps):
        f = sp.expand(fJ.subs({Jx: 1, Jy: G, Jz: D}, simultaneous=True))
        res = []
        for Dv in deltas:
            Dv = mp.mpmathify(Dv)
            P = sp.Poly(f, G)
            cf = [sp.lambdify(D, cc, "mpmath")(Dv) for cc in P.all_coeffs()]
            roots = mp.polyroots(cf, maxsteps=300, extraprec=300)
            roots = [g for g in roots if abs(nondegenerate(Dv, g)) > mp.mpf(10)**-10][:roots_per_delta]
            for Gv in roots:
                F = NUM(Dv, Gv)
                o = F.pt(O)
                x, k, pairs = o, None, crossing_pairs(F)
                for kk in range(1, kmax + 1):
                    x = translate(F, x, pairs)
                    if x is None:
                        k = "degenerate"
                        break
                    if F.same(x, o):
                        k = kk
                        break
                res.append(k)
        return res
