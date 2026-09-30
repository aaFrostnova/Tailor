"""Top-k candidate configurations for one request, for the live loop to validate in order.

The certified optimum sits on the offline floors (the per-image acceptance floor and the capacity line
are what bind on most requests), so the offline margin decides how often the optimum still holds on the
user's images. A larger margin buys reliability with fidelity; a smaller one returns configurations that
fail live more often. Instead of a single answer, the solver can return the k best DISTINCT SKELETONS,
where a skeleton is (fragment set, embed order, front-end stages), each at its own certified optimum
and in non-increasing fidelity. A live driver walks the list and deploys the first candidate that
passes on the user's images, so the offline margin can be set for fidelity and the live check for
reliability.

Enumeration is by blocking clauses over the discrete part of the model: a skeleton is excluded by one
clause saying that at least one of its fragment, order or stage literals differs. Strengths inside a
skeleton stay continuous and are left to the certified optimum (solve_exact's certify-by-probe scheme,
repeated here on instances that carry the blocking clauses). Nothing in watermark_smt_v2 is changed;
this module only adds clauses to the instances it builds.
"""
import math
import z3
import watermark_smt_v2 as W
from live_calibration import read_config


class Skeleton(tuple):
    """(frags, order, fe): the discrete part of a configuration, hashable, with named access."""
    __slots__ = ()
    FIELDS = ("frags", "order", "fe")

    def __new__(cls, frags, order, fe):
        return super().__new__(cls, (tuple(frags), tuple(order), tuple(fe)))

    def __getitem__(self, k):
        if isinstance(k, str): k = self.FIELDS.index(k)
        return tuple.__getitem__(self, k)

    frags = property(lambda s: tuple.__getitem__(s, 0))
    order = property(lambda s: tuple.__getitem__(s, 1))
    fe = property(lambda s: tuple.__getitem__(s, 2))

    def as_dict(self):
        return {"frags": list(self.frags), "order": list(self.order), "fe": list(self.fe)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["frags"], d["order"], d["fe"])

    def __repr__(self):
        return "+".join(self.order) + ("|" + ",".join(self.fe) if self.fe else "")


def db(D):
    """The surrogate objective is a distortion (MSE); report it in dB."""
    return 10.0 * math.log10(255.0 ** 2 / D) if D > 0 else float("inf")


def skeleton_of(built, m):
    cfg = read_config(built, m)
    return Skeleton(sorted(cfg["S"]), cfg["order"], sorted(cfg["fe_on"]))


def skeleton_clause(built, skel, block=True):
    """z3 formula: the model differs from `skel` (block=True) or equals it (block=False).

    Equality is the conjunction of the fragment literals, the stage literals and, when the instance
    optimises the embed order, the precedence literal of every pair of the skeleton's fragments. With
    the order fixed canonical the precedence variables are pinned false and carry no information, so
    they are left out; there the skeleton is the fragment set plus the stages."""
    o, u = built[0], built[1]
    fev = getattr(o, "_fevars", None) or {}
    pv = getattr(o, "_pvars", None) or {}
    lits = [u[f] == z3.BoolVal(f in skel.frags) for f in W.FR]
    # A stage responsible for none of the requested columns changes nothing the request measures
    # (the solver prunes such a stage when it has a measured gain elsewhere, and leaves it free when it
    # has none). Toggling it would make a "new" skeleton out of the same embed and the same decode on
    # every requested column, so such stages are left out of the skeleton's identity; the optimum is
    # pinned with the fewest stages, which keeps them off.
    resp = getattr(o, "_fe_responsible", None)
    lits += [v == z3.BoolVal(c in skel.fe) for c, v in fev.items() if resp is None or resp.get(c)]
    if getattr(o, "_order_enabled", True):
        order = skel.order
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                key = (order[i], order[j])
                if key in pv: lits.append(pv[key] == z3.BoolVal(True))
    same = z3.And(*lits)
    return z3.Not(same) if block else same


def no_passenger_clause(built, attacks):
    """Every selected fragment clears at least one requested column.

    Coverage is best-path: a request is met when, for every column, some selected fragment clears it.
    A fragment that clears nothing is admissible under that rule, it only costs fidelity, so the
    certified optimum never carries one. Under a blocking clause it would: the cheapest skeleton
    different from the optimum is usually the optimum plus a passenger fragment at its clean-floor
    strength, which is the same deployment with more distortion and a higher presence threshold, not
    an alternative worth a live measurement. The enumeration therefore asks that each selected
    fragment clear a column of its own."""
    o, u = built[0], built[1]
    cov = getattr(o, "_cover", None)
    if not cov: return z3.BoolVal(True)
    return z3.And(*[z3.Implies(u[f], z3.Or(*[cov[(f, a)] for a in attacks if (f, a) in cov]))
                    for f in W.FR if any((f, a) in cov for a in attacks)])


def _instance(scen, blocks, pins, build_kw, no_passengers=True):
    built = W.build(**scen, **build_kw)
    o = built[0]
    o._order_enabled = bool(build_kw.get("enable_order", False))
    if no_passengers: o.add(no_passenger_clause(built, scen["attacks"]))
    for b in blocks: o.add(skeleton_clause(built, b, block=True))
    for p in pins: o.add(skeleton_clause(built, p, block=False))
    return built


def solve_exact_blocked(scen, blocks=(), pins=(), eps=1e-9, max_rounds=32, no_passengers=True, **build_kw):
    """The certified optimum of the request restricted by blocking (`blocks`) and pinning (`pins`)
    skeleton clauses, and (default) by the no-passenger rule. Returns (built, model, value, certified);
    (None, None, None, True) when nothing satisfies the restricted request. Mirrors solve_exact_model:
    the optimizer proposes, a fresh instance is asked for anything strictly better until that is
    unsat, and the optimum is then pinned on an instance the caller can read, with the fewest
    front-ends among ties."""
    _inst = lambda: _instance(scen, blocks, pins, build_kw, no_passengers)
    built = _inst()
    o, ps = built[0], built[6]
    o.maximize(ps)
    if o.check() != z3.sat:
        return None, None, None, True
    best = float(o.model().eval(ps).as_fraction())
    certified = False
    for _ in range(max_rounds):
        probe = _inst()
        o2, ps2 = probe[0], probe[6]
        o2.add(ps2 > best + eps)
        if o2.check() != z3.sat:
            certified = True; break
        o2.maximize(ps2); o2.check()
        best = float(o2.model().eval(ps2).as_fraction())
    pinned = _inst()
    o, ps = pinned[0], pinned[6]
    o.add(ps >= best - eps)
    assert o.check() == z3.sat, "the certified optimum became unsatisfiable when pinned"
    fev = getattr(o, "_fevars", None)
    if fev:
        o.minimize(z3.Sum([z3.If(v, 1, 0) for v in fev.values()]))
        assert o.check() == z3.sat, "pinning the optimum and minimising front-ends became unsat"
    return pinned, o.model(), best, certified


def candidate_of(scen, built, m, value, certified, rank):
    cfg = read_config(built, m)
    k = len(cfg["order"])
    cfg["threshold"] = min(1.0, W.presence_threshold(scen["min_ba"], k) + float(scen.get("margin", W.DEFAULT_MARGIN)))
    cfg["threshold_bits"] = W.bits_to_ba(scen["min_bits"]) if scen.get("min_bits", 0) > 0 else 0.0
    return {"rank": rank, "skeleton": skeleton_of(built, m), "cfg": cfg, "psnr_proxy": value,
            "psnr_db": db(-value), "certified": certified}


def enumerate_topk(scen, k, eps=1e-9, max_rounds=32, no_passengers=True, **build_kw):
    """The k best distinct skeletons of the request, each at its certified optimum, best first.

    Fewer than k come back when the request admits fewer feasible skeletons. Candidate 1 is exactly
    the solver's answer (solve_exact_model; the no-passenger rule never binds at the optimum); every
    later one is the certified optimum among the skeletons not yet returned, so fidelity never
    increases along the list."""
    out, blocks = [], []
    for rank in range(1, int(k) + 1):
        built, m, value, certified = solve_exact_blocked(scen, blocks=blocks, eps=eps, max_rounds=max_rounds,
                                                         no_passengers=no_passengers, **build_kw)
        if built is None: break
        c = candidate_of(scen, built, m, value, certified, rank)
        out.append(c); blocks.append(c["skeleton"])
    return out
