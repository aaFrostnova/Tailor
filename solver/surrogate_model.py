import numpy as np
import z3

RANGES = {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6), "VideoSeal": (0.5, 1.5)}
PHASE1_ATTACKS = ["jpeg25","blur","noise","bright","contrast","crop75","crop50","rot9","vaeB","vaeC"]

class PWL:
    """Piecewise-linear curve through knots (xs, ys), clamped outside [xs[0], xs[-1]]."""
    def __init__(self, xs, ys):
        assert len(xs) == len(ys) >= 2
        assert all(xs[i] < xs[i+1] for i in range(len(xs)-1)), "xs must be strictly increasing"
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]

    def eval(self, x):
        return float(np.interp(x, self.xs, self.ys))  # np.interp clamps to end values

    def add_to_z3(self, x_var, name):
        """Return (y_var, constraints) encoding y_var == PWL(x_var), CLAMPED outside
        [xs[0], xs[-1]] to the endpoint y-values -- matching np.interp's behavior in .eval()
        (an out-of-range x_var is no longer unsat). One boolean per interior segment plus a
        `low` and `high` boundary case; exactly one of {low, seg_0 .. seg_{n-2}, high} is
        active. Boundary overlap at a knot (e.g. x_var == xs[0]) is fine -- more than one case
        yields the same y there, z3 just picks one."""
        y = z3.Real(f"y_{name}")
        segs = [z3.Bool(f"seg_{name}_{i}") for i in range(len(self.xs) - 1)]
        low = z3.Bool(f"low_{name}")
        high = z3.Bool(f"high_{name}")
        cons = [z3.PbEq([(b, 1) for b in ([low] + segs + [high])], 1)]  # exactly one case active
        cons.append(z3.Implies(low, z3.And(x_var <= self.xs[0], y == self.ys[0])))
        cons.append(z3.Implies(high, z3.And(x_var >= self.xs[-1], y == self.ys[-1])))
        for i, b in enumerate(segs):
            x0, x1, y0, y1 = self.xs[i], self.xs[i+1], self.ys[i], self.ys[i+1]
            slope = (y1 - y0) / (x1 - x0)
            cons.append(z3.Implies(b, z3.And(x_var >= x0, x_var <= x1,
                                             y == y0 + slope * (x_var - x0))))
        return y, cons

class Surrogate:
    def __init__(self, fragments, attacks, ranges, base, delta, d, e, cap=None, frontend=None,
                 latency=None, perimage=None):
        self.fragments=list(fragments); self.attacks=list(attacks); self._ranges=dict(ranges)
        self._base=base; self._delta=delta; self._d=d; self._e=e   # dict-keyed PWLs
        # OPTIONAL measured extras (absent in older tables, so every accessor returns None instead
        # of raising -- callers must treat None as "not modelled" rather than "zero"):
        #   cap[(f,a)]      reliable-bit capacity (bits) as a PWL in f's native strength
        #   frontend[key]   front-end curves, e.g. "nested_penalty" (MSE vs s_VINE),
        #                   "base_nested_VINE|crop75", "base_resync_TrustMark|rot9"
        #   latency[key]    per-stage wall time in ms, e.g. "latency_ms_angle|rot9" -- a scalar,
        #                   not a function of strength, so it is kept out of the curve blocks
        #                   (a float placed in one is silently dropped when the table is loaded)
        self._cap=dict(cap or {}); self._frontend=dict(frontend or {})
        self._latency=dict(latency or {})
        # PER-IMAGE values behind a curve: {"F|a" or "stage:F|a": {"xs": knots, "ba": [[per image] per
        # knot], "ver": same shape of booleans or None, "n": images}}. The curve blocks store E[ba]; the
        # deployment accepts per image against a threshold that depends on the request, so what a
        # request needs is P(ba >= tau), which the mean cannot give when the per-image distribution
        # is bimodal (UnMarker under the ring at s=0.3: mean 0.694, half the images at 0.99 and half at
        # chance). Kept as raw values rather than as a rate curve because the rate depends on tau, and
        # tau on the request's false-positive budget and fragment count: one stored curve per (cell,
        # budget, count) would be 9,720 curves, while the raw values give any of them exactly.
        self._perimage=dict(perimage or {})

    # ---- per-image rates ------------------------------------------------------------------------
    @staticmethod
    def perimage_key(f, a, stage=None):
        return f"{stage}:{f}|{a}" if stage else f"{f}|{a}"

    def has_perimage(self, f, a, stage=None):
        return self.perimage_key(f, a, stage) in self._perimage

    def rate_curve(self, f, a, tau, stage=None):
        """P(accepted) at each measured knot for threshold `tau`, as a PWL in f's strength, or None.

        An image is accepted when its keyed verification passed (the identity test, budget-free) or
        its bit accuracy reaches tau (the presence test). Campaigns that embedded random bits carry no
        verification flag; there the rate is P(ba >= tau) alone, which undercounts by the soft-decoding
        margin (measured: 8.4% of verified reads sit at 0.85 to 0.88, below the hard limit 0.90) and
        never overcounts."""
        rec = self._perimage.get(self.perimage_key(f, a, stage))
        if rec is None: return None
        xs = list(rec["xs"]); ys = []
        for j in range(len(xs)):
            ba = np.asarray(rec["ba"][j], dtype=float)
            ver = np.asarray(rec["ver"][j], dtype=bool) if rec.get("ver") else np.zeros(len(ba), dtype=bool)
            ys.append(float(np.mean(ver | (ba >= float(tau) - 1e-12))))
        return PWL(xs, ys)
    # ---- request-scoped refinement -------------------------------------------------------------
    # A live measurement lands at ONE strength, but the solver reads a curve. Patching only that point
    # would let the next round step to s+eps and read the un-patched, optimistic value, so the loop
    # would never converge. The measurement is therefore applied as an OFFSET to the whole curve: the
    # shape stays, the level moves by however much reality disagreed. That also matches how the error
    # behaves -- the held-out validation found the discrepancy roughly level across strength rather
    # than concentrated at a point.
    @staticmethod
    def fe_key(stage, f, a):
        """The replacement curve the solver reads for fragment f under attack a with `stage` on."""
        return f"base_fe_{stage}_{f}|{a}"

    def with_live(self, offsets):
        """A copy whose curves are shifted by `offsets`.

        Keys are (frag, attack) for the plain curve or ("fe", stage, frag, attack) for the replacement
        curve of a front-end stage. The second form exists because a configuration with a front-end on
        is solved against the replacement curve, so a contradiction measured on it has to move THAT
        curve: shifting only the plain curve left the next solve reading the same optimistic value and
        the loop re-proposing the same configuration until its round budget ran out."""
        import copy as _c
        sg = _c.copy(self)
        sg._base = dict(self._base); sg._frontend = dict(self._frontend); sg._perimage = dict(self._perimage)
        def _shift_perimage(pk, d):
            rec = self._perimage.get(pk)
            if rec is None: return
            sg._perimage[pk] = {**rec, "ba": [[min(1.0, max(0.0, float(b) + d)) for b in row] for row in rec["ba"]]}
        for key, d in (offsets or {}).items():
            if abs(d) < 1e-12:
                continue
            if len(key) == 4 and key[0] == "fe":
                name = self.fe_key(*key[1:]); c = self._frontend.get(name)
                if c is not None:
                    sg._frontend[name] = PWL(list(c.xs), [min(1.0, max(0.0, y + d)) for y in c.ys])
                _shift_perimage(self.perimage_key(key[2], key[3], key[1]), d)
            else:
                c = self._base.get(tuple(key))
                if c is not None:
                    sg._base[tuple(key)] = PWL(list(c.xs), [min(1.0, max(0.0, y + d)) for y in c.ys])
                _shift_perimage(self.perimage_key(key[0], key[1]), d)
        return sg

    def live_offset(self, f, a, strength, measured, se_live=None, se_table=None,
                    prior_sd=None, k=2.0, asymmetric=True, fe=None):
        """The offset a live measurement implies, taking both sides' uncertainty seriously.

        Treating the live mean as ground truth is wrong twice over. It is itself an estimate -- at ten
        images its standard error is about 0.015, so a disagreement of 0.02 is barely more than noise --
        and the table it overrides was measured on a hundred, making it the more precise of the two. A
        raw overwrite therefore lets sampling noise rewrite a better-measured number.

        Three corrections, in order:

        gate       nothing is patched unless the disagreement exceeds k standard errors of the live
                   measurement. Below that the two are consistent and there is nothing to explain.
        shrinkage  what survives the gate is not applied whole. The live measurement and the table
                   estimate different things -- this user's images versus the pool -- so the question is
                   how much of the gap is a real distribution difference rather than sampling. With a
                   prior width for that difference the posterior mean shrinks the gap by
                   prior^2 / (prior^2 + se_live^2), which is close to the full gap when the live
                   measurement is precise and close to zero when it is not.
        asymmetry  a gap in the pessimistic direction is applied as computed; an optimistic one is
                   damped further. Over-correcting downward costs fidelity, over-correcting upward
                   manufactures a false SAT, and only the second is a safety failure.

        Returns 0.0 when the disagreement does not survive the gate. `fe` names the front-end stage
        whose replacement curve the solver read for this cell; the gap is then measured against it.
        """
        c = self._frontend.get(self.fe_key(fe, f, a)) if fe else None
        if c is None:
            c = self._base.get((f, a))
        if c is None:
            return 0.0
        gap = float(measured) - float(c.eval(strength))
        if se_live is None:                       # no uncertainty supplied -> behave as a raw overwrite
            return gap
        if abs(gap) < k * float(se_live):         # indistinguishable from sampling noise
            return 0.0
        p2 = float(prior_sd) ** 2 if prior_sd else float("inf")
        w = 1.0 if p2 == float("inf") else p2 / (p2 + float(se_live) ** 2)
        if asymmetric and gap > 0:                # reality better than the table: damp harder
            w *= 0.5
        return gap * w

    def range(self,f): return self._ranges[f]
    def base(self,f,a): return self._base[(f,a)]
    def delta(self,g,f,a): return self._delta[(g,f,a)]
    def d(self,f): return self._d[f]
    def e(self,f,g): return self._e[tuple(sorted((f,g)))]
    def cap(self,f,a): return self._cap.get((f,a))          # None when not measured
    def has_cap(self): return bool(self._cap)
    def frontend(self,key): return self._frontend.get(key)  # None when not measured
    def latency(self,key): return self._latency.get(key)    # ms, None when not measured
    def to_dict(self):
        pk=lambda p:{"xs":p.xs,"ys":p.ys}
        return {"fragments":self.fragments,"attacks":self.attacks,"ranges":self._ranges,
                "base":{f"{f}|{a}":pk(self._base[(f,a)]) for (f,a) in self._base},
                "delta":{f"{g}|{f}|{a}":pk(self._delta[(g,f,a)]) for (g,f,a) in self._delta},
                "d":{f:pk(self._d[f]) for f in self._d},
                "e":{f"{p[0]}|{p[1]}":pk(self._e[p]) for p in self._e},
                **({"cap":{f"{f}|{a}":pk(self._cap[(f,a)]) for (f,a) in self._cap}} if self._cap else {}),
                **({"frontend":{k:pk(v) for k,v in self._frontend.items()}} if self._frontend else {}),
                **({"latency":dict(self._latency)} if self._latency else {}),
                **({"perimage":dict(self._perimage)} if self._perimage else {})}
    @staticmethod
    def from_dict(dd):
        mk=lambda o:PWL(o["xs"],o["ys"])
        base={(k.split("|")[0],k.split("|")[1]):mk(v) for k,v in dd["base"].items()}
        delta={(k.split("|")[0],k.split("|")[1],k.split("|")[2]):mk(v) for k,v in dd["delta"].items()}
        d={k:mk(v) for k,v in dd["d"].items()}
        e={tuple(k.split("|")):mk(v) for k,v in dd["e"].items()}
        # optional blocks; entries without {xs,ys} (e.g. free-form meta) are skipped, not fatal
        cap={tuple(k.split("|")):mk(v) for k,v in (dd.get("cap") or {}).items()
             if isinstance(v,dict) and "xs" in v}
        frontend={k:mk(v) for k,v in (dd.get("frontend") or {}).items()
                  if isinstance(v,dict) and "xs" in v}
        latency={k:float(v) for k,v in (dd.get("latency") or {}).items()
                 if isinstance(v,(int,float))}
        perimage={k:v for k,v in (dd.get("perimage") or {}).items()
                  if isinstance(v,dict) and "xs" in v and "ba" in v}
        return Surrogate(dd["fragments"],dd["attacks"],dd["ranges"],base,delta,d,e,
                         cap=cap, frontend=frontend, latency=latency, perimage=perimage)

def synthetic_surrogate(fragments=("VINE","TrustMark","VideoSeal"), attacks=tuple(PHASE1_ATTACKS)):
    import numpy as np
    fragments=list(fragments); attacks=list(attacks); ranges={f:RANGES[f] for f in fragments}
    base,delta,d,e={},{},{},{}
    for fi,f in enumerate(fragments):
        lo,hi=ranges[f]; xs=list(np.linspace(lo,hi,5))
        for a in attacks:
            # monotone-increasing saturating base in [0.5,1.0], attack-specific offset
            off=0.04*(hash((f,a))%5)
            ys=[min(1.0,0.55+0.4*((x-lo)/(hi-lo))-off) for x in xs]
            base[(f,a)]=PWL(xs,ys)
        d[f]=PWL(xs,[ (0.5+ ((x-lo)/(hi-lo)) )*1.0 for x in xs])  # distortion rises with strength
    for gi,g in enumerate(fragments):
        for fi,f in enumerate(fragments):
            if g==f: continue
            lo,hi=ranges[g]; xs=list(np.linspace(lo,hi,5))
            for a in attacks:
                delta[(g,f,a)]=PWL(xs,[0.02*((x-lo)/(hi-lo)) for x in xs])  # small positive
    for i in range(len(fragments)):
        for j in range(i+1,len(fragments)):
            p=tuple(sorted((fragments[i],fragments[j])))
            xs=[0.6,1.25,1.9,2.55,3.2]; e[p]=PWL(xs,[0.1*x for x in xs])   # 1-D in s_f+s_g; spans up to the max pairwise strength sum (TrustMark.hi+VideoSeal.hi=3.1)
    return Surrogate(fragments,attacks,ranges,base,delta,d,e)
