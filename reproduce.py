"""Reproduce every computed claim in "Rational points on Baxter's curve and roots of unity in the eight-vertex model"
(Shankar Sivarajan).

Each check recomputes a quantity and compares it with the value stated in the paper, printing PASS or FAIL; labels
follow the paper's numbering (Lemma 2.1, Lemma 2.2, Remark 2.3, Lemma 2.4, Theorem 3.1, Remarks 3.2 and 3.3,
Corollary 4.1, Example 4.2) and end with the line(s) of main.tex they refer to. Claims that are citations or that
are not recomputed are printed as NOTE lines and listed at the end.

Run:  python reproduce.py            (about a minute)
"""
import random
import sys
import time
from fractions import Fraction as Fr
from itertools import combinations
from math import gcd, isqrt
from pathlib import Path

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baxter_curve import boukraa_maillard as BM  # noqa: E402
from baxter_curve import curve as C  # noqa: E402
from baxter_curve import group as GR  # noqa: E402
from baxter_curve import jacobi as JA  # noqa: E402
from baxter_curve import loci as LO  # noqa: E402
from baxter_curve import weierstrass as WE  # noqa: E402

D, G = C.D, C.G
T0 = time.time()
RESULTS, NOTES = [], []


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def check(label, ok, detail=""):
    ok = bool(ok)
    RESULTS.append((label, ok))
    log(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail else ""))


def note(label):
    NOTES.append(label)
    log(f"[NOTE] {label}")


def section(name):
    log(f"===== {name} =====")


def rat(x):
    x = Fr(x)
    return sp.Rational(x.numerator, x.denominator)


def zero(expr):
    return sp.cancel(sp.together(expr)) == 0


SAMPLES_U = [(Fr(2, 7), Fr(3, 11)), (Fr(-5, 28), Fr(-5, 21)), (Fr(3, 2), Fr(-1, 5))]

# ---------------------------------------------------------------------------------------------------------------
section("Section 2, Lemma 2.1 (l.68-80)")
p, m, P, M = C.p, C.m, C.P, C.M
Q1d = sp.expand(C.Q1.subs(C.TO_ABCD, simultaneous=True))
Q2d = sp.expand(C.Q2.subs(C.TO_ABCD, simultaneous=True))
check("Lemma 2.1 proof: 2Q1 = (1-D)p^2 + (1+D)m^2 - (1+D)P^2 - (1-D)M^2 and "
      "4Q2 = (1-G)p^2 - (1-G)m^2 - (1+G)P^2 + (1+G)M^2 in p=a+b, m=a-b, P=c+d, M=c-d (l.76-78)",
      zero(2*Q1d - sum(al*v**2 for al, v in zip(C.ALPHA, (p, m, P, M))))
      and zero(4*Q2d - sum(bt*v**2 for bt, v in zip(C.BETA, (p, m, P, M)))))
pencil = list(zip(C.ALPHA, C.BETA))
stated_pencil = [(1 - D, 1 - G), (1 + D, G - 1), (1 + D, 1 + G), (1 - D, -1 - G)]
check("Lemma 2.1 proof: the four points (alpha_i:beta_i) are (1-D:1-G), (1+D:G-1), (1+D:1+G), (1-D:-1-G) (l.81)",
      all(zero(a1*b2 - a2*b1) for (a1, b1), (a2, b2) in zip(pencil, stated_pencil)))
dets = [sp.factor(pencil[i][0]*pencil[j][1] - pencil[j][0]*pencil[i][1]) for i, j in combinations(range(4), 2)]
stated_coinc = [G - 1, D - G, D - 1, D + 1, D + G, G + 1]     # Gamma=1, Delta=Gamma, Delta=1, Delta=-1, Delta=-Gamma, Gamma=-1
check("Lemma 2.1 proof: the six coincidences (pairs 12,13,14,23,24,34) occur exactly at Gamma=1, Delta=Gamma, "
      "Delta=1, Delta=-1, Delta=-Gamma, Gamma=-1 (l.81)",
      all(LO.same_up_to_unit(dt, s) for dt, s in zip(dets, stated_coinc)), str(dets))
disc = sp.factor(sp.discriminant(WE.quartic(), WE.lam))
ratio = sp.cancel(disc / C.DEG**2)
check("Lemma 2.1: discriminant of the pencil det(mu A + B) = const * ((D^2-1)(G^2-1)(D^2-G^2))^2 (l.71)",
      ratio.is_number and ratio != 0, f"disc = {disc}")
smooth_ok = all(not C.singular_points_exist(*pt) for pt in [(Fr(2, 7), Fr(3, 11)), (Fr(-5, 6), Fr(-25, 27)),
                                                              (Fr(0), Fr(1, 3)), (Fr(3, 4), Fr(3, 5))])
sing_pts = [(Fr(1), Fr(1, 3)), (Fr(-1), Fr(1, 3)), (Fr(2, 5), Fr(1)), (Fr(2, 5), Fr(-1)), (Fr(2, 5), Fr(2, 5)),
            (Fr(2, 5), Fr(-2, 5))]
sing_ok = all(C.singular_points_exist(*pt) for pt in sing_pts)
check("Lemma 2.1: Jacobian criterion: E smooth at 4 sample points of U, singular at a sample point of each of the six "
      "lines Delta=+-1, Gamma=+-1, Delta=+-Gamma (l.69)", smooth_ok and sing_ok)
note("Lemma 2.1: 'two diagonal quadrics meet in a smooth genus-one curve iff the four pencil points are "
     "distinct' is a standard fact; only checked through the Jacobian criterion at sample parameters above (l.81)")

# ---------------------------------------------------------------------------------------------------------------
section("Section 2, Lemma 2.2 and the eight points (l.84-102)")
subs_pt = lambda v: {C.a: v[0], C.b: v[1], C.c: v[2], C.d: v[3]}
check("Sec 2: the eight points (1:0:+-1:0), (0:1:+-1:0), (1:0:0:+-1), (0:1:0:+-1) lie on E for all Delta, Gamma "
      "and are distinct (l.84-86)",
      all(zero(C.Q1.subs(subs_pt(v))) and zero(C.Q2.subs(subs_pt(v))) for v in C.EIGHT)
      and len({GR.QQ(0, 0).pt(v) for v in C.EIGHT}) == 8)
abcd = (C.a, C.b, C.c, C.d)
pres = all(zero(Q.subs(dict(zip(abcd, f(abcd))), simultaneous=True) - Q) for Q in (C.Q1, C.Q2) for f in (C.iota, C.kappa))
check("Lemma 2.2 proof: iota(a,b,c,d) = (-a,-b,c,d) and kappa(a,b,c,d) = (b,a,d,c) preserve Q1 and Q2 (l.103)", pres)
Kfield = sp.QQ.frac_field(D, G)
cc, dd, aa, bb = C.c, C.d, C.a, C.b
I1 = sp.groebner([C.Q1.subs({aa: 0, bb: 0}), C.Q2.subs({aa: 0, bb: 0})], cc, dd, domain=Kfield)
I2 = sp.groebner([C.Q1.subs({cc: 0, dd: 0}), C.Q2.subs({cc: 0, dd: 0})], aa, bb, domain=Kfield)
check("Lemma 2.2 proof: iota has no fixed point on E: a=b=0 forces c=d=0 and c=d=0 forces a=b=0 "
      "(c^3, d^3, resp. a^3, b^3 lie in the ideal over Q(Delta,Gamma)) (l.103)",
      all(I1.reduce(v**3)[1] == 0 for v in (cc, dd)) and all(I2.reduce(v**3)[1] == 0 for v in (aa, bb)))
t = sp.symbols("t")
kappa_conds = []
for mu in (1, -1):
    v = {aa: 1, bb: mu, cc: t, dd: mu*t}
    r = sp.factor(sp.resultant(sp.expand(C.Q1.subs(v)), sp.expand(C.Q2.subs(v)), t))
    kappa_conds.append(r)
stated_kappa = [(1 - D)/(1 + D) - (1 - G)/(1 + G), (1 + D)/(1 - D) - (1 - G)/(1 + G)]
kap_ok = all(any(LO.same_up_to_unit(f, h) for h in (D - G, D + G)) or sp.Poly(f, D, G).is_ground
             or any(LO.same_up_to_unit(f, h) for h in C.DEG_FACTORS)
             for r in kappa_conds for f, e in sp.factor_list(r)[1])
kap_ok &= LO.same_up_to_unit(sp.numer(sp.together(stated_kappa[0])), D - G)
kap_ok &= LO.same_up_to_unit(sp.numer(sp.together(stated_kappa[1])), D + G)
check("Lemma 2.2 proof: fixed points (1:mu:t:mu t) of kappa on E need (1-D)/(1+D)=(1-G)/(1+G) (mu=1) or "
      "(1+D)/(1-D)=(1-G)/(1+G) (mu=-1), i.e. Delta=+-Gamma (resultants in t factor into D-+G and "
      "degenerate factors) (l.103)", kap_ok, str(kappa_conds))
check("Lemma 2.2 proof: the fixed points (0:0:1:+-1) of kappa are not on E since Q2 = -+(1+Gamma) != 0 (l.103)",
      all(zero(C.Q2.subs({aa: 0, bb: 0, cc: 1, dd: s}) + s*(1 + G)) for s in (1, -1)))
check("Lemma 2.2: iota(O) = (1:0:-1:0), kappa(O) = (0:1:0:1), iota kappa(O) = (0:1:0:-1); images of P_c under iota, "
      "kappa, iota kappa are (0:1:-1:0), (1:0:0:1), (1:0:0:-1) (l.95-101)",
      [GR.QQ(0, 0).pt(f(C.O)) for f in (C.iota, C.kappa, lambda v: C.iota(C.kappa(v)))]
      == [GR.QQ(0, 0).pt(v) for v in ((1, 0, -1, 0), (0, 1, 0, 1), (0, 1, 0, -1))]
      and [GR.QQ(0, 0).pt(f(C.PC)) for f in (C.iota, C.kappa, lambda v: C.iota(C.kappa(v)))]
      == [GR.QQ(0, 0).pt(v) for v in ((0, 1, -1, 0), (1, 0, 0, 1), (1, 0, 0, -1))])
# translation property, exactly over Q at sample curves
trans_ok, n_tested = True, 0
for Dv, Gv in SAMPLES_U:
    F = GR.QQ(Dv, Gv)
    tau, taup, o = F.pt((1, 0, -1, 0)), F.pt((0, 1, 0, 1)), F.pt(C.O)
    pts = GR.multiples(F, 5)
    for R in pts:
        for f, s in ((C.iota, tau), (C.kappa, taup)):
            z = GR.add(F, R, s)
            trans_ok &= z is not None and z == F.pt(f(R))
            n_tested += 1
    s2 = GR.add(F, tau, taup)
    trans_ok &= s2 == F.pt((0, 1, 0, -1))
    trans_ok &= GR.add(F, F.pt((0, 1, 0, -1)), tau) == taup and GR.add(F, F.pt((0, 1, 0, -1)), taup) == tau
check("Lemma 2.2: iota and kappa act as translation by tau = (1:0:-1:0) and tau' = (0:1:0:1) in the chord group law, "
      "and tau + tau' = (0:1:0:-1), so E[2] = {O, tau, tau', tau+tau'} (exact, 3 curves, R = c..5c) (l.95-96)",
      trans_ok, f"{n_tested} translations")
FQ = GR.QQDG()
pair_ok = True
for A, Cc in GR.crossing_pairs(FQ):
    for Y in [FQ.pt(v) for v in C.EIGHT]:
        if any(Y == z for z in (A, Cc, FQ.pt(C.O), FQ.pt(C.PC))):
            continue
        u1, u2 = GR.residual(FQ, A, FQ.pt(C.O), Y), GR.residual(FQ, Cc, FQ.pt(C.PC), Y)
        if u1 is not None and u2 is not None:
            pair_ok &= u1 == u2
            break
    else:
        pair_ok = False
check("Thm 3.1 proof: each pair (A,C) in {(P_c,O), (iota P_c, iota O), (kappa P_c, kappa O), (iota kappa P_c, "
      "iota kappa O)} satisfies [A]-[C] = c (A+O and C+P_c cut the same residual point, over Q(Delta,Gamma)) (l.160)",
      pair_ok)
dbl_ok = True
for Dv, Gv in SAMPLES_U:
    F = GR.QQ(Dv, Gv)
    ms = GR.multiples(F, 8)
    for i in (0, 1, 2, 3):
        dbl_ok &= GR.add_diff(F, ms[i], ms[i], F.pt(C.O)) == ms[2*i + 1]
check("Thm 3.1 proof, group law (consistency of the implementation): doubling n c through the tangent plane equals "
      "2n c from the chord iteration (n = 1..4, 3 curves, exact) (l.160)", dbl_ok)
check("Lemma 2.2: the eight points are E[2] and P_c + E[2] (P_c + tau etc. by the chord law, exact at 3 curves) (l.99)",
      all(GR.add(GR.QQ(*s), GR.QQ(*s).pt(C.PC), GR.QQ(*s).pt(t2)) == GR.QQ(*s).pt(img)
          for s in SAMPLES_U for t2, img in (((1, 0, -1, 0), (0, 1, -1, 0)), ((0, 1, 0, 1), (1, 0, 0, 1)),
                                              ((0, 1, 0, -1), (1, 0, 0, -1)))))

# ---------------------------------------------------------------------------------------------------------------
section("Section 2, Remark 2.3 (l.106-113)")
check("Rem 2.3: e_i = -prod_{j != i}(t_4 - t_j) gives e_1 = -4(D+G)/((G-1)(G+1)^2), e_2 = -4(D-1)/((G-1)(G+1)^2), "
      "e_3 = -4(D-1)(D+G)/((G-1)^2(G+1)^2) (l.107-108)",
      all(zero(x - y) for x, y in zip(WE.e_from_t(), WE.E_STATED)))
check("Rem 2.3: the four roots of det(mu A + B) (A, B Gram matrices of 2Q1, 4Q2) are -1/t_i (l.111)",
      all(zero(WE.quartic().subs(WE.lam, -1/ti)) for ti in WE.T)
      and all(zero(r + 1/ti) for r, ti in zip(WE.quartic_roots(), WE.T)), str(WE.quartic_roots()))
iso = []
for r in WE.quartic_roots():
    cf = WE.cubic_from_root(r)
    ok, s = WE.matches_stated(cf)
    iso.append((r, cf[0], ok, s))
check("Rem 2.3: for every root sent to infinity, the cubic model (made monic) becomes y^2 = (x-e_1)(x-e_2)(x-e_3) "
      "by x -> x/((G-1)(G+1))^2 (y -> y/((G-1)(G+1))^3) and a translation in Q(Delta,Gamma) (l.111)",
      all(o for _, _, o, _ in iso), "; ".join(f"root {r}: lead coeff {c3}, shift {s}" for r, c3, _, s in iso))
note("Rem 2.3: the cubic obtained by sending a root to infinity has leading coefficient q'(root) != 1 "
     "(e.g. 8(Delta+Gamma)(Gamma-1)/(Delta+1)), so the scaling (x, y) -> (c_3 x, c_3 y), stated in the text, "
     "is needed before the coordinate change (x, y) -> (x/((Gamma-1)(Gamma+1))^2, y/((Gamma-1)(Gamma+1))^3) and "
     "the translation; with it the claim holds for all 4 roots, as checked above (l.111)")
note("Rem 2.3: that E is isomorphic to the Jacobian of y^2 = det(mu A + B) is [AKMMMP, Sec. 3.3 and Thm. 3.1], cited; "
     "it is cross-checked by the point counts below (l.111)")
mu = WE.legendre(WE.E_STATED)
check("Rem 2.3: Legendre parameter (e_3-e_1)/(e_2-e_1) = (Delta^2-Gamma^2)/(1-Gamma^2) (l.111)",
      zero(mu - WE.LEGENDRE_STATED), str(mu))
a2, a4, a6 = sp.Poly(sp.expand(sp.prod([WE.x - e for e in WE.E_STATED])), WE.x).all_coeffs()[1:]
jW = WE.j_from_cubic(a2, a4, a6)
jL = 256*(mu**2 - mu + 1)**3/(mu**2*(mu - 1)**2)
check("Rem 2.3: j(E) = 256 (D^4 - D^2G^2 - D^2 + G^4 - G^2 + 1)^3 / ((D^2-1)^2 (G^2-1)^2 (D^2-G^2)^2), from c_4^3/disc "
      "of the Weierstrass model and from the Legendre parameter (l.111-112)",
      zero(jW - WE.J_STATED) and zero(jL - WE.J_STATED))
cnt = []
for Dv, Gv in [(Fr(2, 7), Fr(3, 11)), (Fr(-11, 46), Fr(-11, 23)), (Fr(3, 4), Fr(3, 5)), (Fr(1, 3), Fr(1, 4))]:
    es = [e.subs({D: rat(Dv), G: rat(Gv)}) for e in WE.E_STATED]
    for pr in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        bad = any(sp.Rational(x).q % pr == 0 for x in es + [rat(Dv), rat(Gv)])
        bad = bad or rat(C.DEG.subs({D: rat(Dv), G: rat(Gv)})).p % pr == 0 or (rat(Gv)**2 - 1).p % pr == 0
        if bad:
            continue
        cnt.append((Dv, Gv, pr, C_ := WE.count_quadrics(Dv, Gv, pr), WE.count_weierstrass(es, pr)))
check("Rem 2.3 (sanity): #E(F_p) for the quadric intersection = #(y^2 = prod(x - e_i))(F_p) at every good prime "
      "5 <= p <= 37 for 4 sample curves (l.107)", all(x == y for *_, x, y in cnt), f"{len(cnt)} (curve, prime) pairs")
check("Lemma 2.2: E[2] subset E(Q): the 2-division values e_i of the Weierstrass model are rational, so full "
      "rational 2-torsion also on the cubic model (l.38, l.99)", all(e.is_rational_function(D, G) for e in WE.E_STATED))

# ---------------------------------------------------------------------------------------------------------------
section("Section 2, Lemma 2.4 (l.117-135)")
r2, r1, (l1, l2) = JA.symbolic_reduction()
check("Lemma 2.4 proof: ab - cd = sn u sn(lambda-u)(1 - k sn^2 lambda), ab + cd = sn u sn(lambda-u)(1 + k sn^2 lambda) "
      "(l.130)", l1 == 0 and l2 == 0)
check("Lemma 2.4: with the addition formula, the numerators of Q2 and Q1 reduce to 0 modulo cn^2 = 1 - sn^2, "
      "dn^2 = 1 - k^2 sn^2 (for u and lambda), with the stated Delta, Gamma (l.122-123, l.130-132)",
      r2 == 0 and r1 == 0, f"remainders {r2}, {r1}")
mp.mp.dps = 40
num_ok, grp_ok, per_ok, n_num = True, True, True, 0
for kk, lamv in [(mp.mpf("0.6"), mp.mpf("0.37")), (mp.mpf("0.3"), mp.mpc("0.5", "0.3")),
                 (mp.mpc("0.4", "0.2"), mp.mpc("0.9", "-0.4"))]:
    K, Kp = JA.periods(kk)
    Ds, Gs = JA.invariants_stated(lamv, kk)
    num_ok &= abs(1 + kk*JA.sn(lamv, kk)**2) > 1e-6
    F = GR.NUM(Ds, Gs)
    wf = lambda u: F.norm(JA.w(u, lamv, kk))
    for u in [mp.mpf("0.11"), mp.mpc("0.29", "0.17"), mp.mpf("-0.83")]:
        Dw, Gw = C.invariants(JA.w(u, lamv, kk))
        num_ok &= abs(Dw - Ds) < 1e-30 and abs(Gw - Gs) < 1e-30
        per_ok &= F.same(wf(u + 2*K), F.norm(C.iota(JA.w(u, lamv, kk))))
        per_ok &= F.same(wf(u + 1j*Kp), F.norm(C.kappa(JA.w(u, lamv, kk))))
        per_ok &= F.same(wf(u + 4*K), wf(u)) and F.same(wf(u + 2j*Kp), wf(u))
        for v in [mp.mpf("0.23"), mp.mpc("-0.41", "0.07")]:
            s = GR.add(F, wf(u), wf(v))
            grp_ok &= s is not None and F.same(s, wf(u + v))
            n_num += 1
        tr = GR.translate(F, wf(u))
        grp_ok &= tr is not None and F.same(tr, wf(u + lamv))
    num_ok &= F.same(F.norm(JA.w(mp.mpf(10)**-35, lamv, kk)), F.pt(C.O))
    num_ok &= F.same(F.norm(JA.w(lamv + mp.mpf(10)**-35, lamv, kk)), F.pt(C.PC))
check("Lemma 2.4 (numerical, 40 digits, 3 moduli incl. complex, 1 + k sn^2 lambda != 0): Delta, Gamma of w(u) are "
      "independent of u and equal the stated formulas; w(0) = O, w(lambda) = P_c (l.118-124)", num_ok)
check("Lemma 2.4 proof: w(u+2K) = iota w(u), w(u+iK') = kappa w(u), w(u+4K) = w(u+2iK') = w(u) (l.134)", per_ok)
check("Lemma 2.4: w is a group homomorphism: w(u+v) = w(u) + w(v) by the chord construction, and w(u+lambda) = "
      "w(u) + c (l.126)", grp_ok, f"{n_num} sums")
# every class lambda of exact order N in C/Lambda lands on an order-N locus of Theorem 3.1 and on no other
lam_ok, lam_n, lam_inf, inf_classes = True, 0, 0, []
hom = {N: [LO.to_J(f) for f in fs] for N, fs in LO.STATED.items()}
for kk in [mp.mpf("0.6"), mp.mpf("0.3"), mp.mpc("0.4", "0.2")]:
    K, Kp = JA.periods(kk)
    for N in (3, 4, 6, 8):
        for pq in [(i, j) for i in range(N) for j in range(N) if gcd(gcd(i, j), N) == 1]:
            lamv = 4*K*pq[0]/N + 2j*Kp*pq[1]/N
            Jv = C.couplings(JA.w(mp.mpf("0.1234"), lamv, kk))
            sc = max(abs(z) for z in Jv)
            Jv = [z/sc for z in Jv]
            if abs(Jv[0]) < 1e-25:          # J_x = 0: (Delta, Gamma) at infinity, outside the affine (Delta, Gamma)-plane
                lam_inf += 1
                inf_classes.append((str(kk), N, pq))
                continue
            vals = {N2: min(abs(sp.lambdify((LO.Jx, LO.Jy, LO.Jz), f, "mpmath")(*Jv)) for f in fs)
                    for N2, fs in hom.items()}
            lam_ok &= vals[N] < 1e-25 and all(vals[N2] > 1e-10 for N2 in vals if N2 != N)
            lam_n += 1
check("Lemma 2.4 + Thm 3.1 (numerical): every lambda of exact order N = 3, 4, 6, 8 in C/Lambda (3 moduli) gives "
      "(J_x:J_y:J_z) on an order-N locus of Thm 3.1(3) and on no locus of another order (l.126, l.148-149)",
      lam_ok, f"{lam_n} classes; {lam_inf} classes of order 4 with J_x = 0 (Delta, Gamma infinite) skipped")

# J_x = 0 (Delta, Gamma infinite): order 4 projectively, lambda = +-iK'/2 mod 2K
expect_inf = {(0, 1), (0, 3), (2, 1), (2, 3)}        # (i, j): lambda = i K + j iK'/2, i.e. +-iK'/2 mod 2K
by_k = {}
for kname, N, pq in inf_classes:
    by_k.setdefault(kname, set()).add((N, pq))
check("Thm 3.1 (couplings), J_x = 0 (numerical): among the classes lambda = (4K i + 2iK' j)/N of exact order N = 3, 4, "
      "6, 8 (3 moduli), J_x(w(u)) = 0 exactly for N = 4 and lambda = iK'/2, 3iK'/2 = -iK'/2 mod 2K and their "
      "translates by 2K, i.e. lambda = +-iK'/2 mod 2K (l.165)",
      len(by_k) == 3 and all(v == {(4, x) for x in expect_inf} for v in by_k.values()), str(sorted(by_k.items())))
jx_ok, jx_n = True, 0
for kk in [mp.mpf("0.6"), mp.mpf("0.3"), mp.mpc("0.4", "0.2")]:
    K, Kp = JA.periods(kk)
    for lamv in (1j*Kp/2, -1j*Kp/2, 1j*Kp/2 + 2*K, -1j*Kp/2 + 2*K):
        jx_ok &= abs(kk*JA.sn(lamv, kk)**2 + 1) < 1e-30
        # coordinates of lambda in the basis (4K, 2iK') of Lambda; exact order 4 means 4x, 4y integers, 2x or 2y not
        x, y = mp.lu_solve(mp.matrix([[mp.re(4*K), mp.re(2j*Kp)], [mp.im(4*K), mp.im(2j*Kp)]]),
                           mp.matrix([mp.re(lamv), mp.im(lamv)]))
        def near(z):
            return abs(z - mp.nint(z)) < 1e-30
        jx_ok &= near(4*x) and near(4*y) and not (near(2*x) and near(2*y))
        for u in [mp.mpf("0.1234"), mp.mpc("0.29", "0.17")]:
            Jv = C.couplings(JA.w(u, lamv, kk))
            jx_ok &= abs(Jv[0]) < 1e-30*max(abs(z) for z in Jv)
        jx_n += 1
check("Thm 3.1 (couplings): lambda = +-iK'/2 mod 2K gives k sn^2 lambda = -1, J_x(w(u)) = 0 for all u (Delta, Gamma "
      "infinite), and has exact order 4 in C/Lambda (numerical, 40 digits, 3 moduli incl. complex) (l.165)",
      jx_ok, f"{jx_n} values of lambda")

# Lemma 2.4, converse (l.126, l.136)
kq, gq = sp.symbols("k g")
quad = gq*kq**2 - (1 + gq**2 - (1 + gq)**2*D**2)*kq + gq
gG = (1 - G)/(1 + G)
check("Lemma 2.4 proof (converse): with k sn^2 lambda = g, cn^2 = 1 - g/k, dn^2 = 1 - k g, the conditions "
      "Gamma = (1-g)/(1+g) and Delta^2 (1+g)^2 = cn^2 dn^2 are g = (1-Gamma)/(1+Gamma) and g k^2 - (1 + g^2 - "
      "(1+g)^2 Delta^2) k + g = 0 (symbolic) (l.136)",
      zero((1 - gG)/(1 + gG) - G) and zero(kq*(D**2*(1 + gq)**2 - (1 - gq/kq)*(1 - kq*gq)) - quad))
q1v, qm1v = sp.factor(quad.subs({kq: 1, gq: gG})), sp.factor(quad.subs({kq: -1, gq: gG}))
check("Lemma 2.4 proof (converse): the roots of the quadratic are k and 1/k; at k = 1 it is 4(Delta^2-Gamma^2)/(1+Gamma)^2, "
      "zero iff Delta = +-Gamma; at k = -1 it is 4(1-Delta^2)/(1+Gamma)^2, zero iff Delta^2 = 1; 1 + g = 2/(1+Gamma), "
      "and g = 0, infinity iff Gamma = 1, -1 (symbolic) (l.136)",
      zero(sp.Poly(quad, kq).all_coeffs()[0] - sp.Poly(quad, kq).all_coeffs()[2])
      and zero(q1v - 4*(D**2 - G**2)/(1 + G)**2) and zero(qm1v - 4*(1 - D**2)/(1 + G)**2)
      and zero(1 + gG - 2/(1 + G)) and LO.same_up_to_unit(sp.numer(sp.together(gG)), G - 1)
      and LO.same_up_to_unit(sp.denom(sp.together(gG)), G + 1), f"k=1: {q1v}; k=-1: {qm1v}")
conv_ok, conv_n, conv_flip = True, 0, 0
for Dv, Gv in SAMPLES_U + [(Fr(3, 4), Fr(3, 5)), (Fr(5, 2), Fr(7, 3)), (Fr(-1, 3), Fr(1, 9)), (Fr(-5, 6), Fr(-25, 27))]:
    dv, gv = mp.mpf(Dv.numerator)/Dv.denominator, mp.mpf(Gv.numerator)/Gv.denominator
    g_ = (1 - gv)/(1 + gv)
    for kk in mp.polyroots([g_, -(1 + g_**2 - (1 + g_)**2*dv**2), g_]):
        kk = mp.mpc(kk)
        conv_ok &= abs(kk**2) > 1e-20 and abs(kk**2 - 1) > 1e-20
        lamv = mp.ellipf(mp.asin(mp.sqrt(g_/kk)), kk**2)          # sn(lamv) = sqrt(g/k)
        K, Kp = JA.periods(kk)
        conv_ok &= abs(JA.sn(lamv, kk)**2 - g_/kk) < 1e-30
        conv_ok &= abs(1 + kk*JA.sn(lamv, kk)**2 - 2/(1 + gv)) < 1e-30
        (D0, G0), (D1, G1) = JA.invariants_stated(lamv, kk), JA.invariants_stated(lamv + 2*K, kk)
        conv_ok &= abs(D0 + D1) < 1e-30 and abs(G0 - G1) < 1e-30          # lambda -> lambda + 2K flips Delta
        lam_good = lamv if abs(D0 - dv) < 1e-30 else lamv + 2*K
        conv_flip += lam_good is not lamv
        Ds, Gs = JA.invariants_stated(lam_good, kk)
        conv_ok &= abs(Ds - dv) < 1e-30 and abs(Gs - gv) < 1e-30
        for u in [mp.mpf("0.123"), mp.mpc("0.31", "-0.22")]:
            Dw, Gw = C.invariants(JA.w(u, lam_good, kk))
            conv_ok &= abs(Dw - dv) < 1e-30 and abs(Gw - gv) < 1e-30
        conv_n += 1
check("Lemma 2.4 (converse, numerical, 40 digits): for 7 rational (Delta, Gamma) in U and both roots k of the "
      "quadratic (real, complex, |k| > 1 and < 1), sn^2 lambda = g/k gives 1 + k sn^2 lambda = 2/(1+Gamma), "
      "lambda -> lambda + 2K changes the sign of Delta, and lambda or lambda + 2K returns (Delta, Gamma) by the "
      "Jacobi formulas and as the invariants of w(u) (l.126, l.136)",
      conv_ok, f"{conv_n} (Delta, Gamma, k); lambda + 2K needed in {conv_flip}")

# ---------------------------------------------------------------------------------------------------------------
section("Section 3, Theorem 3.1 (l.141-161)")
log("computing n c over Q(Delta, Gamma) for n = 1..10 by the chord construction")
sym = LO.symbolic_multiples(10, log=log)
F = GR.QQDG()
check("Thm 3.1 proof: each n c (n <= 10) computed over Q(Delta,Gamma) lies on E identically (l.160)",
      all(GR.on_curve(F, r["point"]) for r in sym))
check("Thm 3.1 proof: the coordinates of n c (n <= 8) are polynomials with no common factor (l.160)",
      all(r["common"] == 1 for r in sym[:8]), str([r["degree"] for r in sym[:8]]) + " = total degrees")
for k in range(1, 9):
    r = sym[k - 1]
    if k in LO.PROOF_FACTORS:
        check(f"Thm 3.1 proof: irreducible factors in U of gcd(x1, x3, x0 - x2) for n = {k} are exactly "
              f"{LO.PROOF_FACTORS[k] or 'none'} (l.160)",
              LO.factor_sets_equal(r["loci"], LO.PROOF_FACTORS[k]), str(r["loci"]))
degenerate_seen = {k: [f for f in r["gcd_factors"] if f not in r["loci"]] for k, r in ((r["k"], r) for r in sym)}
log(f"  (factors of the gcd outside U, discarded: {degenerate_seen})")
Z = {r["k"]: r["loci"] for r in sym}
new = {}
for N in (3, 4, 6, 8):
    lower = [f for dvs in range(1, N) if N % dvs == 0 for f in Z[dvs]]
    new[N] = [f for f in Z[N] if not any(LO.same_up_to_unit(f, h) for h in lower)]
check("Thm 3.1(3): exact-order loci (factors of Z_N not in Z_d, d | N, d < N) are as stated: N=3 D-G+DG; N=4 D, G; "
      "N=6 D+G+DG, D-G-DG, D+G-DG; N=8 D^2+G^2-D^2G^2, G^2-D^2-+D^2G^2 (l.148-149)",
      all(LO.factor_sets_equal(new[N], LO.STATED[N]) for N in (3, 4, 6, 8)), str(new))
flat = [(N, f) for N, fs in LO.STATED.items() for f in fs]
bad = []
for (N1, f1), (N2, f2) in combinations(flat, 2):
    for solu in sp.solve([f1, f2], [D, G], dict=True):
        if len(solu) < 2:
            bad.append((f1, f2, solu))       # common component: would be a curve
        elif sp.simplify(C.DEG.subs(solu)) != 0:
            bad.append((f1, f2, solu))
check("Thm 3.1 proof: distinct loci (all 9 curves of (3)) meet only outside U (l.160)", not bad, str(bad))
spot, spot_n = True, 0
ts = [Fr(1, 3), Fr(2, 5), Fr(-3, 7), Fr(5, 2), Fr(4, 9), Fr(-7, 3)]
for N, fs in LO.STATED.items():
    for which in range(len(fs)):
        for Dv, Gv in LO.locus_points(N, which, ts):
            if not C.nondegenerate(Dv, Gv):
                continue
            assert sp.sympify(fs[which]).subs({D: rat(Dv), G: rat(Gv)}) == 0
            o = GR.order_of_c(GR.QQ(Dv, Gv), 12)
            spot &= o == N
            spot_n += 1
check("Thm 3.1(3): at rational points in U on each of the 9 curves, c has exact order 3, 4, 6, 8 as stated "
      "(exact chord arithmetic over Q) (l.148-149)", spot, f"{spot_n} points")
rng = random.Random(20261120)
off_n, off_ok = 0, True
while off_n < 150:
    Dv = Fr(rng.randint(-30, 30), rng.randint(1, 30))
    Gv = Fr(rng.randint(-30, 30), rng.randint(1, 30))
    if not C.nondegenerate(Dv, Gv) or LO.on_any_locus(Dv, Gv):
        continue
    off_ok &= GR.order_of_c(GR.QQ(Dv, Gv), 12) is None
    off_n += 1
check("Thm 3.1(4): at 150 random rational points of U off the loci, n c != O for n = 1..12 (seed 20261120) (l.153)",
      off_ok)
note("Thm 3.1(1)-(2),(4): the torsion structure Z/2 x Z/2nu, nu <= 4, and positive rank rest on Mazur's theorem "
     "[Mazur, Thm. 8], cited; checking n c != O for n <= 12 suffices to exclude every rational torsion order "
     "(l.144-151, l.158, l.162)")
check("Intro: the free-fermion line Delta = 0 is a component of the order-4 locus; the supersymmetric line "
      "J_xJ_y + J_xJ_z + J_yJ_z = 0 is an order-6 locus (l.57, l.167)",
      any(LO.same_up_to_unit(f, D) for f in LO.STATED[4])
      and any(LO.same_up_to_unit(f, LO.Jx*LO.Jy + LO.Jx*LO.Jz + LO.Jy*LO.Jz) for f in LO.STATED_J[6]))

section("Section 3, the loci in the couplings (l.165)")
check("Thm 3.1 (couplings): Delta = J_z/J_x and Gamma = J_y/J_x with J_x = ab+cd, J_y = ab-cd, J_z = (a^2+b^2-c^2-d^2)/2 "
      "(l.47-53)", all(zero(x - y) for x, y in zip(C.invariants((aa, bb, cc, dd)),
                                                    (C.couplings((aa, bb, cc, dd))[2] / C.couplings((aa, bb, cc, dd))[0],
                                                     C.couplings((aa, bb, cc, dd))[1] / C.couplings((aa, bb, cc, dd))[0]))))
for N in (3, 4, 6, 8):
    got = [LO.to_J(f) for f in LO.STATED[N]]
    # strip powers of J_x that come from clearing denominators
    got = [sp.factor(g / LO.Jx**sp.Poly(g, LO.Jx).monoms()[-1][0]) for g in got]
    if N == 4:                      # stated as the single equation Delta Gamma = 0, i.e. J_y J_z = 0
        got = [sp.factor(sp.prod(got))]
    check(f"Thm 3.1 (couplings): order-{N} loci in (J_x:J_y:J_z) are {LO.STATED_J[N]} (l.165)",
          LO.factor_sets_equal(got, LO.STATED_J[N]), str(got))
section("Comparison with Boukraa-Maillard (l.38, l.55, l.165, l.179)")

log(f"  polynomials transcribed from {BM.SOURCE}; equation numbers of the arXiv PDF (see baxter_curve/boukraa_maillard.py)")
check("Thm 3.1 (couplings): the order-3 locus is BM's hat V_z^(3) = -J_xJ_y + J_zJ_x + J_zJ_y (App. B, display "
      "before eq. (134)) and the order-6 locus J_xJ_y + J_xJ_z + J_yJ_z is BM's V^(3) (same display); J_z = 0 is a "
      "component of the order-4 locus (l.165, l.167)", LO.same_up_to_unit(BM.V3HAT_Z, LO.STATED_J[3][0])
      and any(LO.same_up_to_unit(BM.V3, f) for f in LO.STATED_J[6]) and LO.same_up_to_unit(LO.to_J(D), LO.Jz) and any(LO.same_up_to_unit(f, D) for f in LO.STATED[4]))
check("Thm 3.1 (couplings): the three order-8 loci are exactly the zero sets of BM's doubling polynomials J_x^(2), "
      "J_y^(2), J_z^(2) (eq. (49); J_z^(2) = 0 in Sec. VI.A of arXiv:hep-th/0003212) (l.167)", LO.factor_sets_equal(list(BM.J2), LO.STATED_J[8]), str(BM.J2))
mp.mp.dps = 40
dbl_ok = True
for kk, lamv in [(mp.mpf("0.6"), mp.mpf("0.37")), (mp.mpc("0.4", "0.2"), mp.mpc("0.9", "-0.4"))]:
    J1 = C.couplings(JA.w(mp.mpf("0.123"), lamv, kk))
    Jd = C.couplings(JA.w(mp.mpf("0.123"), 2*lamv, kk))
    v = [sp.lambdify((LO.Jx, LO.Jy, LO.Jz), e, "mpmath")(*J1) for e in BM.J2]
    v = [v[0], v[1], -v[2]]
    dbl_ok &= max(abs(v[i]*Jd[j] - v[j]*Jd[i]) for i in range(3) for j in range(3)) < 1e-25 * max(abs(t) for t in v) * max(abs(t) for t in Jd)
check("Rem 3.2, supporting the order-10 claim and the comparison with BM (a relation between the normalizations, "
      "not a statement of the paper): BM eq. (49) in the paper's normalization (numerical, 2 moduli): "
      "(J_x^(2) : J_y^(2) : -J_z^(2)) at lambda equals (J_x : J_y : J_z) at 2 lambda, i.e. BM's doubling is "
      "lambda -> 2 lambda followed by J_z -> -J_z (Delta -> -Delta, lambda -> lambda + 2K) (l.167, l.179)", dbl_ok)
pre_ok, pre_n = True, 0
fJ2 = [sp.lambdify((LO.Jx, LO.Jy, LO.Jz), e, "mpmath") for e in BM.J2]
for kk in [mp.mpf("0.6"), mp.mpf("0.3"), mp.mpc("0.4", "0.2")]:
    K, Kp = JA.periods(kk)
    for N in (3, 6, 8):
        for i, j in [(i, j) for i in range(N) for j in range(N) if gcd(gcd(i, j), N) == 1]:
            lamv = 4*K*i/N + 2j*Kp*j/N
            Jv = C.couplings(JA.w(mp.mpf("0.1234"), lamv, kk))
            Jv = [z/max(abs(t) for t in Jv) for z in Jv]
            v2 = [f(*Jv) for f in fJ2]
            v2 = [z/max(abs(t) for t in v2) for z in v2]
            Jd = C.couplings(JA.w(mp.mpf("0.1234"), 2*lamv, kk))
            Jd = [z/max(abs(t) for t in Jd) for z in Jd]
            nz2, nzd = sum(abs(z) < 1e-25 for z in v2), sum(abs(z) < 1e-25 for z in Jd)
            pre_ok &= (nz2 == 1 and nzd == 1) if N == 8 else (min(abs(z) for z in v2) > 1e-10 and nzd == 0)
            pre_n += 1
check("Thm 3.1 (couplings): the order-8 loci, the zero sets of J_x^(2), J_y^(2), J_z^(2), are the preimages of the "
      "order-4 components under the doubling (numerical, 3 moduli): for every lambda of exact order 8 in C/Lambda "
      "exactly one of J^(2)_x, J^(2)_y, J^(2)_z vanishes and (J_x:J_y:J_z) at 2 lambda lies on J_xJ_yJ_z = 0; for "
      "lambda of exact order 3 or 6 none does (l.167)", pre_ok, f"{pre_n} classes")
Z10 = Z[10]
for name, f, eq in (("V_y^(5)", BM.VY5, "(137)"), ("hat V_y^(5)", BM.VY5HAT, "(138)")):
    g = BM.dehomogenize(f)
    in10 = any(LO.same_up_to_unit(g, h) for h in Z10)
    inlow = any(LO.same_up_to_unit(g, h) for kk in (1, 2, 5) for h in Z[kk])
    check(f"Rem 3.2: BM's {name} (eq. {eq}) is a component of Z_10 and of no Z_n, n = 1, 2, 5, by the exact "
          f"computation of 10c over Q(Delta,Gamma): c has exact order 10 on it (l.179)", in10 and not inlow)
check("Rem 3.2, BM eq. (138): hat V_y^(5)(J_x,J_y,J_z) as printed equals V_y^(5)(J_y,J_x,J_z) (transcription "
      "consistency) (l.179)",
      sp.expand(BM.VY5HAT - BM.VY5.subs({LO.Jx: LO.Jy, LO.Jy: LO.Jx}, simultaneous=True)) == 0)
img = sp.expand(BM.VY5.subs(LO.Jx, -LO.Jx))
check("Rem 3.2: the image of BM's V_y^(5) under J_x -> -J_x is the order-5 curve of Rem 3.2 (l.179)",
      LO.same_up_to_unit(img, LO.to_J(LO.ORDER5_STATED)))
exact10 = [h for h in Z10 if not any(LO.same_up_to_unit(h, g) for kk in (1, 2, 5) for g in Z[kk])]
log(f"  (Z_10 has {len(Z10)} components in U: the order-5 curve and {len(exact10)} of exact order 10, among them BM's "
    f"V_y^(5) and hat V_y^(5) dehomogenized; all order-10 components: {exact10})")
deltas = [mp.mpc("0.31", "0.17"), mp.mpc("-0.7", "0.4"), mp.mpc("1.3", "-0.25")]
o12 = LO.numeric_orders(BM.P3_Z, deltas)
check("Intro and Rem 3.2: BM's P_z^(3) = hat V_z^(3)(J^(2)) (eq. (136), listed with K^12) has order 12 in the "
      "paper's normalization (numerical, 60 digits, complex points) (l.55, l.179)", set(o12) == {12}, str(o12))
o20 = LO.numeric_orders(BM.P5, deltas[:2])
check("Intro and Rem 3.2: BM's P_x^(5) (eq. (55), listed with K^20) has order 20 in the paper's normalization "
      "(numerical, 60 digits, complex points) (l.55, l.179)", set(o20) == {20}, str(o20))
# the other order-12 and order-20 varieties of BM App. B (see baxter_curve/boukraa_maillard.py for their definitions)
J2sub = dict(zip((LO.Jx, LO.Jy, LO.Jz), BM.J2))
check("BM eq. (134)-(135): V^(3)(J^(2)) = V^(3) hat V_x^(3) hat V_y^(3) hat V_z^(3), with hat V_x^(3), hat V_y^(3) the "
      "cyclic permutations of hat V_z^(3) (transcription consistency) (l.167, l.179)",
      sp.expand(BM.V3.subs(J2sub, simultaneous=True) - BM.V3*BM.V3HAT_X*BM.V3HAT_Y*BM.V3HAT_Z) == 0)
check("BM eq. (139): V_y^(5)(J^(2)) = -P_y^(5) and hat V_y^(5)(J^(2)) = -P_x^(5), with P_x^(5) = P_5 of eq. (55) and "
      "P_y^(5), P_z^(5) its cyclic permutations by eq. (54) (transcription consistency) (l.179)",
      sp.expand(BM.VY5.subs(J2sub, simultaneous=True) + BM.P5_Y) == 0
      and sp.expand(BM.VY5HAT.subs(J2sub, simultaneous=True) + BM.P5_X) == 0)
check("BM, text after eq. (55): P_x^(5) is invariant under J_y <-> J_z, P_y^(5) under J_x <-> J_z, P_z^(5) under "
      "J_x <-> J_y (transcription consistency) (l.179)",
      all(sp.expand(f - f.subs({u: v, v: u}, simultaneous=True)) == 0
          for f, u, v in ((BM.P5_X, LO.Jy, LO.Jz), (BM.P5_Y, LO.Jx, LO.Jz), (BM.P5_Z, LO.Jx, LO.Jy))))
oth = {"P_x^(3)": (BM.P3_X, 12, deltas), "P_y^(3)": (BM.P3_Y, 12, deltas),
       "P_y^(5)": (BM.P5_Y, 20, deltas[:2]), "P_z^(5)": (BM.P5_Z, 20, deltas[:2])}
for name, (f, N, dl) in oth.items():
    o = LO.numeric_orders(f, dl)
    check(f"Rem 3.2: BM's {name} (listed with K^{N}) has order {N} in the paper's normalization (numerical, 60 digits, "
          f"complex points) (l.55, l.179)", set(o) == {N}, str(o))
low = [(N, h) for N in (3, 4, 6) for h in LO.STATED[N]]
check("Rem 3.2 (control for the next check): the test reports the line Delta = 1/3, which meets the order-3 locus at "
      "(1/3, 1/2) in U, as meeting it inside U (l.179)", not LO.meets_only_outside_U(D - sp.Rational(1, 3), LO.STATED[3][0]))
bm_meet = {}
for name, f in (("P_x^(3)", BM.P3_X), ("P_y^(3)", BM.P3_Y), ("P_z^(3)", BM.P3_Z),
                ("P_x^(5)", BM.P5_X), ("P_y^(5)", BM.P5_Y), ("P_z^(5)", BM.P5_Z)):
    g = BM.dehomogenize(f)
    bm_meet[name] = [(N, h) for N, h in low if not LO.meets_only_outside_U(g, h)]
check("Rem 3.2: BM's varieties of orders 12 and 20 (P_x^(3), P_y^(3), P_z^(3), P_x^(5), P_y^(5), P_z^(5), dehomogenized "
      "at J_x = 1) meet the loci of orders 3, 4, 6 of Thm 3.1(3) (all 6 curves) only outside U: substituting each locus, "
      "solved for one variable, every irreducible factor of the resulting resultant divides "
      "(Delta^2-1)(Gamma^2-1)(Delta^2-Gamma^2) (exact) (l.179)",
      not any(bm_meet.values()), str({k: v for k, v in bm_meet.items() if v}))

# Maillard-Boukraa, Ann. Fond. L. de Broglie 26 (2001), Sec. 5.2.2, eqs. (54)-(56)
log(f"  polynomials transcribed from {BM.AFLB_SOURCE} (printed pages; see baxter_curve/boukraa_maillard.py)")
check("MB eq. (44) (shift doubling) is identical to BM eq. (49) (transcription consistency) (l.167)",
      all(sp.expand(x - y) == 0 for x, y in zip(BM.AFLB_J2, BM.J2)))


def sub3(f, J):
    return sp.expand(f.subs(dict(zip((LO.Jx, LO.Jy, LO.Jz), J)), simultaneous=True))


T55 = sp.prod(BM.AFLB_T55)
cov_ok = (sp.expand(sub3(BM.AFLB_G2, BM.AFLB_J2) - BM.AFLB_T54**4*BM.AFLB_G2) == 0
          and sp.expand(sub3(BM.AFLB_G3, BM.AFLB_J2) - BM.AFLB_T54**6*BM.AFLB_G3) == 0
          and sp.expand(sub3(BM.AFLB_G2, BM.AFLB_J3) - T55**4*BM.AFLB_G2) == 0
          and sp.expand(sub3(BM.AFLB_G3, BM.AFLB_J3) - T55**6*BM.AFLB_G3) == 0)
J5 = (LO.Jx*BM.P5_X, LO.Jy*BM.P5_Y, LO.Jz*BM.P5_Z)
T56 = sp.prod(BM.AFLB_T56)
rng5 = random.Random(5)
for _ in range(8):           # J^(5) has degree 13; exact evaluation at random integer points instead of expansion
    v = {LO.Jx: rng5.randint(-9, 9), LO.Jy: rng5.randint(-9, 9), LO.Jz: rng5.randint(-9, 9)}
    J5v = dict(zip((LO.Jx, LO.Jy, LO.Jz), [j.subs(v) for j in J5]))
    cov_ok &= BM.AFLB_G2.subs(J5v) == T56.subs(v)**4*BM.AFLB_G2.subs(v)
    cov_ok &= BM.AFLB_G3.subs(J5v) == T56.subs(v)**6*BM.AFLB_G3.subs(v)
check("MB eqs. (38), (44), (46), (54)-(56): (g2, g3) -> (t^4 g2, t^6 g3) under the multiplication of the shift by 2, 3 "
      "(exact, symbolic) and 5 (J^(5) from BM eqs. (54)-(55); exact at 8 random integer points), with t = 2 J_xJ_yJ_z, "
      "t0 tx ty tz, t1 t2 t3 t4 (transcription consistency) (l.167)", cov_ok)
check("MB eq. (54): t = 2 J_xJ_yJ_z, their condition of order dividing 4, is J_x times the order-4 locus J_yJ_z = 0 of "
      "Thm 3.1, i.e. the projective order-4 locus J_xJ_yJ_z = 0 (all its components have order 4: J_y, J_z by Thm 3.1, "
      "J_x by the numerical check above) (l.165, l.167)",
      LO.same_up_to_unit(BM.AFLB_T54, LO.Jx*LO.STATED_J[4][0]))
check("MB eq. (55): the factors t0, tx, ty, tz of their condition of order dividing 6 are the four sign patterns "
      "+-J_xJ_y +-J_xJ_z + J_yJ_z, i.e. the union of the order-3 locus (tz) and the three order-6 loci (t0 = the "
      "supersymmetric line, tx, ty) of Thm 3.1 (l.167)",
      LO.factor_sets_equal(list(BM.AFLB_T55), LO.STATED_J[3] + LO.STATED_J[6])
      and LO.same_up_to_unit(BM.AFLB_T55[3], LO.STATED_J[3][0]) and LO.same_up_to_unit(BM.AFLB_T55[0], BM.V3))
t56d = [BM.dehomogenize(f) for f in BM.AFLB_T56]
check("MB eq. (56): the factor t2 = (-Jy+Jz)(Jz+Jy)^2 Jx^3 - Jy Jz (-Jy+Jz)^2 Jx^2 - Jz^2 Jy^2 (-Jy+Jz) Jx + Jz^3 Jy^3 "
      "equals the order-5 curve of Rem 3.2 homogenized (Delta = J_z/J_x, Gamma = J_y/J_x) (l.167)",
      LO.same_up_to_unit(BM.AFLB_T56[1], LO.to_J(LO.ORDER5_STATED)))
z5 = [i for i, f in enumerate(t56d) if any(LO.same_up_to_unit(f, h) for h in Z[5])]
low10 = [i for i, f in enumerate(t56d) if any(LO.same_up_to_unit(f, h) for kk in (1, 2, 5) for h in Z[kk])]
check("MB eq. (56): t1, t2, t3, t4 (dehomogenized) are exactly the four components of Z_10 in U (exact computation of "
      "10 c over Q(Delta,Gamma)), so t = 0 is the condition of order dividing 10; t2 is the component of Z_5 and t1, "
      "t3, t4 have exact order 10 (t3, t4 are BM's hat V_y^(5), V_y^(5)) (l.167, l.179)",
      LO.factor_sets_equal(t56d, Z[10]) and z5 == [1] and low10 == [1]
      and LO.same_up_to_unit(BM.AFLB_T56[2], BM.VY5HAT) and LO.same_up_to_unit(BM.AFLB_T56[3], BM.VY5),
      f"components of Z_5 among t1..t4: {[f't{i + 1}' for i in z5]}")
o56 = {f"t{i + 1}": LO.numeric_orders(f, deltas[:2]) for i, f in enumerate(BM.AFLB_T56)}
check("MB eq. (56) (numerical, 60 digits, complex points): c has order 5 on t2 and order 10 on t1, t3, t4 (l.167, "
      "l.179)", set(o56["t2"]) == {5} and all(set(o56[n]) == {10} for n in ("t1", "t3", "t4")), str(o56))
o3 = LO.numeric_orders(BM.V3HAT_Z, deltas)
check("BM App. B: hat V_z^(3) = -J_xJ_y + J_xJ_z + J_yJ_z, to which BM assign order 3, has order 3 in the paper's "
      "normalization (numerical, 60 digits, complex points; exactly by Thm 3.1) (l.167)", set(o3) == {3}, str(o3))
check("MB eqs. (45), (46), (48): J^(3) = (J_x P_x^(3), J_y P_y^(3), J_z P_z^(3)) with P_x^(3) of eq. (48) and P_y^(3), "
      "P_z^(3) its cyclic permutations (transcription consistency) (l.167)",
      all(sp.expand(j - v*f) == 0 for j, v, f in zip(BM.AFLB_J3, (LO.Jx, LO.Jy, LO.Jz),
                                                    (BM.AFLB_P3_X, BM.AFLB_P3_Y, BM.AFLB_P3_Z))))
check("MB eqs. (48), (53): their conditions of order 12, P_x^(3), P_y^(3), P_z^(3) = 0, are BM's P_x^(3), P_y^(3), "
      "P_z^(3) up to a constant (so BM's unprinted P_x^(3), P_y^(3), taken as hat V_{x,y}^(3)(J^(2)), are confirmed), "
      "and their P_5 = P_x^(5) (display after eq. (50); order 20 by eq. (53)) equals BM's P_x^(5) of eq. (55); these "
      "have orders 12 and 20 in the paper's normalization by the numerical checks above (l.55, l.167)",
      all(LO.same_up_to_unit(a, b) for a, b in zip((BM.AFLB_P3_X, BM.AFLB_P3_Y, BM.AFLB_P3_Z),
                                                   (BM.P3_X, BM.P3_Y, BM.P3_Z)))
      and sp.expand(BM.AFLB_P5 - BM.P5) == 0)
r4 = sp.cancel(BM.AFLB_T56[3]/BM.VY5)
check("Rem 3.2: BM's variety listed with order 5, V_y^(5) of their eq. (137), is up to sign the factor t4 of MB's "
      "order-10 condition (56) (exact), and has order 10 (checks above) (l.179)", r4 in (1, -1), f"t4 / V_y^(5) = {r4}")
log("  summary (l.38, l.55, l.165, l.179): BM's varieties have orders 3 (hat V_z^(3)), 4 (J_z = 0), 6 (V^(3)), "
    "8 (J^(2) = 0), 10 (V_y^(5), hat V_y^(5)), 12 (P_x^(3), P_y^(3), P_z^(3)), 20 (P_x^(5), P_y^(5), P_z^(5)) in the "
    "paper's normalization, if the checks above pass")
note("Intro: the statement on [MaillardBoukraa] (rational points and the Mordell-Weil group, no torsion bound) is a "
     "citation, not checked (l.55)")

# ---------------------------------------------------------------------------------------------------------------
section("Section 3, Remark 3.2 (l.169-180)")
o5 = Z[5]
check("Rem 3.2: the order-5 locus is the single printed polynomial (l.170-171)",
      len(o5) == 1 and LO.same_up_to_unit(o5[0], LO.ORDER5_STATED), str(o5))
pd5 = sp.Poly(LO.ORDER5_STATED, D, G)
check("Rem 3.2: the order-5 polynomial is irreducible over Q, of degree 3 in each of Delta and Gamma (l.170)",
      len(sp.factor_list(LO.ORDER5_STATED)[1]) == 1 and sp.degree(LO.ORDER5_STATED, D) == 3
      and sp.degree(LO.ORDER5_STATED, G) == 3, f"total degree {pd5.total_degree()}")
o7 = Z[7]
check("Rem 3.2: the order-7 locus is one irreducible polynomial of degree 6 in each variable (l.174-177)",
      len(o7) == 1 and sp.degree(o7[0], D) == 6 and sp.degree(o7[0], G) == 6,
      f"total degree {sp.Poly(o7[0], D, G).total_degree()}")
check("Rem 3.2: the order-7 locus Z_7 (exact computation of 7 c) is exactly the printed polynomial of degree 6 in "
      "each variable (l.174-177)", len(o7) == 1 and LO.same_up_to_unit(o7[0], LO.ORDER7_STATED))
known = [f for fs in LO.STATED.values() for f in fs] + [LO.ORDER5_STATED, LO.ORDER7_STATED]
comp_ok = all(any(LO.same_up_to_unit(f, h) for h in known) for n in range(1, 9) for f in Z[n])
comp_ok &= all(any(LO.same_up_to_unit(f, h) for n in range(1, 9) for h in Z[n]) for f in known)
check("Thm 3.1 and Rem 3.2 (\"New here\"): in U, the components of Z_n = {n c = O} for n <= 8 (exact) are exactly the 9 "
      "curves of Thm 3.1(3) and the curves of orders 5 and 7 of Rem 3.2, and each of these occurs (l.167, l.160, "
      "l.170-177)", comp_ok, str({n: len(Z[n]) for n in range(1, 9)}) + " components")
log(f"  order-7 polynomial (saved to {LO.save_order7(o7[0])}): {o7[0]} = 0")
check("Rem 3.2: (0,0) lies on the order-5 curve and outside U (l.179)",
      LO.ORDER5_STATED.subs({D: 0, G: 0}) == 0 and not C.nondegenerate(0, 0))
hits = []
for f in (LO.ORDER5_STATED, o7[0]):
    for q in range(1, 21):
        for pn in range(-40, 41):
            if gcd(pn, q) != 1:
                continue
            Dv = sp.Rational(pn, q)
            for rt in sp.Poly(f.subs(D, Dv), G).ground_roots():
                if C.DEG.subs({D: Dv, G: rt}) != 0:
                    hits.append((f, Dv, rt))
check("Rem 3.2 (sanity, consistent with Mazur): no rational point in U on the order-5 and order-7 curves with "
      "Delta = p/q, |p| <= 40, q <= 20 (l.179)", not hits, str(hits[:3]))
note("Rem 3.2: that the curves of orders 5, 7 and BM's varieties of orders 10, 12, 20 have no rational point in U "
     "follows from Thm 3.1 (Mazur), cited; the orders of BM's varieties in the paper's normalization and the claim "
     "that those of orders 12 and 20 meet the loci of orders 3, 4, 6 only outside U are checked in the "
     "Boukraa-Maillard section (l.179)")

# ---------------------------------------------------------------------------------------------------------------
section("Section 3, Remark 3.3 (l.182)")
meet = {N: sorted({r for f in LO.STATED[N] for r in sp.solve(f.subs(G, 1), D)}, key=lambda z: float(z))
        for N in (3, 4, 6, 8)}
stated_meet = {3: [sp.Rational(1, 2)], 4: [sp.Integer(0)], 6: [sp.Rational(-1, 2)], 8: [-1/sp.sqrt(2), 1/sp.sqrt(2)]}
check("Rem 3.3: on Gamma = 1 the loci of orders 3, 4, 6 meet at Delta = 1/2, 0, -1/2 and the order-8 loci only at "
      "Delta = +-1/sqrt 2 (l.183)", all(len(meet[N]) == len(stated_meet[N])
                                         and all(sp.simplify(x - y) == 0 for x, y in zip(meet[N], stated_meet[N]))
                                         for N in meet), str(meet))
check("Rem 3.3: 1/2, 0, -1/2 = -cos(2 pi/N) for N = 3, 4, 6 (and -cos(2 pi/8) = -1/sqrt 2) (l.183)",
      all(sp.simplify(-sp.cos(2*sp.pi/N) - v) == 0 for N, v in
          ((3, sp.Rational(1, 2)), (4, 0), (6, sp.Rational(-1, 2)), (8, -1/sp.sqrt(2)))))
Dv, Gv = Fr(3, 4), Fr(3, 5)
check("Rem 3.3: (3/4, 3/5) lies in U on Gamma^2 - Delta^2 = -Delta^2 Gamma^2 and c has exact order 8 there (l.183)",
      C.nondegenerate(Dv, Gv) and Gv**2 - Dv**2 == -Dv**2*Gv**2 and GR.order_of_c(GR.QQ(Dv, Gv), 12) == 8)
w2412 = (2, 4, 1, 2)
check("Rem 3.3: the weights (2,4,1,2) have (Delta, Gamma) = (3/4, 3/5), on Gamma^2 - Delta^2 = -Delta^2 Gamma^2 and "
      "in U, and are not among the eight points (l.183)",
      C.invariants(tuple(Fr(x) for x in w2412)) == (Dv, Gv) and C.nondegenerate(Dv, Gv)
      and Gv**2 - Dv**2 == -Dv**2*Gv**2 and GR.QQ(0, 0).pt(w2412) not in {GR.QQ(0, 0).pt(v) for v in C.EIGHT})
es = [e.subs({D: rat(Dv), G: rat(Gv)}) for e in WE.E_STATED]
c16 = [(pr, WE.count_weierstrass(es, pr)) for pr in sp.primerange(7, 200)
       if all(sp.Rational(x).q % pr for x in es) and rat(C.DEG.subs({D: rat(Dv), G: rat(Gv)})).p % pr]
check("Rem 3.3 (sanity): at (3/4, 3/5), 16 | #E(F_p) for every good prime 7 <= p < 200 (Z/2 x Z/8 injects) "
      "(l.183)",
      all(n % 16 == 0 for _, n in c16), f"{len(c16)} primes")

# ---------------------------------------------------------------------------------------------------------------
section("Section 4, Corollary 4.1 and Example 4.2 (l.186-213)")
EX = {(Fr(-5, 28), Fr(-5, 21)): [(8, 14, 13, 14), (50600, 6678, 10350, 53053)],
      (Fr(-5, 6), Fr(-25, 27)): [(3360, 451, 5824, 6765)],
      (Fr(-11, 46), Fr(-11, 23)): [(24276960, 132832441, 64664880, 141295279)]}
eight = {GR.QQ(0, 0).pt(v) for v in C.EIGHT}
for (Dv, Gv), ws in EX.items():
    for wv in ws:
        inv = C.invariants(tuple(Fr(x) for x in wv))
        check(f"Ex 4.2: {wv} are positive integers with (Delta, Gamma) = ({Dv}, {Gv}), not among the eight points "
              f"(l.199-209)", all(isinstance(x, int) and x > 0 for x in wv) and inv == (Dv, Gv)
              and GR.QQ(0, 0).pt(wv) not in eight)
for (Dv, Gv) in EX:
    F = GR.QQ(Dv, Gv)
    check(f"Ex 4.2: ({Dv}, {Gv}) is in U and on none of the loci of Thm 3.1(3), and n c != O for n = 1..12, so c has "
          f"infinite order (with Mazur) (l.199)", C.nondegenerate(Dv, Gv) and not LO.on_any_locus(Dv, Gv)
          and GR.order_of_c(F, 12) is None)
    # Mazur-free: a torsion point of order N reduces to a point of order exactly N at every good prime p >= 3
    ords = []
    for pr in sp.primerange(5, 400):
        if Dv.denominator % pr == 0 or Gv.denominator % pr == 0:
            continue
        if (rat(C.DEG.subs({D: rat(Dv), G: rat(Gv)})).p % pr) == 0:
            continue
        o = GR.order_of_c(GR.GF(pr, Dv, Gv), pr + 2 + 2*isqrt(pr))
        if o is not None:
            ords.append((pr, o))
        if len({x for _, x in ords}) >= 2:
            break
    check(f"Ex 4.2 (independent of Mazur): at ({Dv}, {Gv}) the reductions of c at two good primes have different "
          f"orders, so c is not torsion (l.199)", len({x for _, x in ords}) >= 2, str(ords[-2:]))
cor_ok = True
for (Dv, Gv), ws in EX.items():
    F = GR.QQ(Dv, Gv)
    for wv in ws:
        orbit = [F.pt(wv)] + GR.multiples(F, 6, start=wv)
        cor_ok &= len(set(orbit)) == len(orbit) and all(GR.on_curve(F, x) for x in orbit)
check("Cor 4.1: for each example weight R, the column weights R + n c (n = 0..6) are rational points of E, pairwise "
      "distinct for distinct n (l.191-193)", cor_ok)
note("Cor 4.1 and Sec 4: commutation of inhomogeneous transfer matrices follows from the star-triangle relation "
     "[Baxter72], cited; not recomputed (l.188)")

# ---------------------------------------------------------------------------------------------------------------
n_pass = sum(ok for _, ok in RESULTS)
log(f"{n_pass}/{len(RESULTS)} checks passed")
for label, ok in RESULTS:
    if not ok:
        log(f"  FAIL {label}")
log(f"{len(NOTES)} notes (claims not checked, cited, or remarks on the text):")
for n in NOTES:
    log(f"  NOTE {n}")
log(f"total runtime {time.time() - T0:.1f} s")
sys.exit(0 if n_pass == len(RESULTS) else 1)
