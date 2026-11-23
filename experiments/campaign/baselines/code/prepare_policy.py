"""Freeze both historical baselines before any baseline final measurements.

Only source code, original offline tables, and current request inputs are read.
Historical modules are never imported.  AST extraction for the consistency
reference executes named pure functions only, without module-level statements.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from baseline_policy import (FRAGMENTS, METHODS, HistoricalOracle, POLICY_VERSION,
                             beta_from_fpr, canonical_hash, propose)

SC = Path(os.environ.get('TAILOR_WORKSPACE', 'workspace'))
CF = Path(os.environ.get('TAILOR_PROJECT', '.'))
N = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914')
B = Path(__file__).resolve().parents[1]
OLD_W = CF / "solver/watermark_smt_v2.py"
OLD_POLICY = SC / "rigorous_baselines.py"
OLD_FRONTIER = SC / "solve_frontier.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        try:
            os.link(temp, path)
        except FileExistsError:
            if read(path) != value:
                raise ValueError(f"immutable artifact differs: {path}")
    finally:
        os.unlink(temp)


def literal_constant(source, name):
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError((source, name))


def pure_function(source, name, namespace):
    tree = ast.parse(Path(source).read_text())
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one pure function: {source}:{name}")
    module = ast.Module(body=matches, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def build_oracle():
    data = read(CF / "inputs/smt_inputs.json")
    full = read(CF / "inputs/smt_inputs_full.json")
    table = read(SC / "baseline_table.json")
    fields = {"VINE": "vine", "TrustMark": "tm", "VideoSeal": "vs"}
    ba = {f: {a: row[fields[f]] for a, row in full["per_attack"].items()} for f in FRAGMENTS}
    for block in ("REGC", "CALIB", "ADV"):
        for f, values in literal_constant(OLD_W, block).items():
            ba[f].update(values)
    for f, value in literal_constant(OLD_W, "CROP50").items():
        ba[f]["crop50"] = value
    alpha = copy.deepcopy(data["alpha_sweep"])
    levels = literal_constant(OLD_W, "ALV")
    solo = {f: data["fragments"][f]["psnr"] for f in FRAGMENTS}
    cells = table["cells"]

    def cell(group, key):
        value = cells[group].get(key)
        return value.get("value") if isinstance(value, dict) else value

    attacks = sorted({key.split("/")[-1] for key in cells["bit_acc"]})
    for f in FRAGMENTS:
        value = cell("psnr_solo", f"{f}@1.0")
        if value is not None:
            solo[f] = value
        for attack in attacks:
            value = cell("bit_acc", f"{f}@1.0/{attack}")
            if value is not None:
                ba[f][attack] = value
    for level in levels:
        value = cell("psnr_solo", f"VINE@{level}")
        if value is not None:
            alpha[str(level)]["psnr"] = value
        for attack in attacks:
            value = cell("bit_acc", f"VINE@{level}/{attack}")
            if value is not None:
                alpha[str(level)][attack] = value
    # Original SOLO_Q explicitly uses alpha=1.0 for VINE's tie-break.
    solo["VINE"] = alpha[str(levels[3])]["psnr"]
    return dict(schema="historical_baseline_oracle_v1", fragments=list(FRAGMENTS),
                bit_acc=ba, alpha=alpha, alpha_levels=levels, solo_q=solo,
                swept=sorted(literal_constant(OLD_W, "SWEPT")),
                frontend_recovery_ba=0.96,
                unmarker_recovery_ba=literal_constant(OLD_W, "UNMK_REC"),
                unknown_numeric_fallback=0.0,
                unknown_semantics="proposal coverage unavailable; never a measured failure")


def reference_greedy(oracle):
    """Run original pure functions using equivalent Python boolean primitives."""
    env = dict(FR=list(FRAGMENTS), ALL3=FRAGMENTS,
               ALPHA=oracle["alpha"], ALV=oracle["alpha_levels"], SWEPT=set(oracle["swept"]),
               SOLO_Q=oracle["solo_q"], ROT={"rot9", "rot30"},
               NEST_ATT={"crop75", "crop90", "crop50", "unmarker"},
               UNMK_REC=oracle["unmarker_recovery_ba"],
               BA_full=lambda f, a: oracle["bit_acc"][f].get(a, 0.0),
               BoolVal=bool, And=lambda *x: all(x), Or=lambda *x: any(x), _tb=bool)
    pure_function(OLD_W, "frag_defends", env)
    pure_function(OLD_FRONTIER, "_attack_defended", env)
    return pure_function(OLD_POLICY, "greedy_subset", env)


def sources():
    paths = [OLD_W, OLD_POLICY, OLD_FRONTIER, SC / "baseline_table.json",
             CF / "inputs/smt_inputs.json", CF / "inputs/smt_inputs_full.json",
             CF / "inputs/smt_inputs_composite.json",
             N / "fixed/code/watermark_smt_v2.py", N / "fixed/code/capacity_protocol.py",
             Path(__file__).resolve(), Path(__file__).with_name("baseline_policy.py").resolve(), N / "code/unified_detector.py"]
    paths += [N / f"fixed/inputs/C{c}.json" for c in range(1, 6)]
    return {str(path): digest(path) for path in paths}


def prepare(root=B):
    root = Path(root).resolve()
    if any((root / method / "final/measurement_started.json").exists() for method in METHODS):
        raise ValueError("baseline final measurements started; cannot run policy preparation")
    fingerprints = sources()
    oracle_value = build_oracle()
    oracle = HistoricalOracle(oracle_value)
    # Historical gain and tie-break rule; thresholds now depend on each candidate configuration.

    write_once(root / "inputs/historical_oracle.json", oracle_value)
    method_rows = {method: [] for method in METHODS}
    audits = {method: dict(config_counts=collections.Counter(), unknown_attacks=collections.Counter(),
                           dual_fe_per_class=collections.Counter()) for method in METHODS}
    replayed = 0
    for class_index in range(1, 6):
        group = f"C{class_index}"
        records = read(N / f"fixed/inputs/{group}.json")["records"]
        if len(records) != 2000 or {r["i"] for r in records} != set(range(2000)):
            raise ValueError(f"incomplete original request coverage: {group}")
        for record in sorted(records, key=lambda r: r["i"]):
            if "min_bits" in record:
                raise ValueError("payload input is prohibited")
            request = {field: copy.deepcopy(record[field]) for field in ("attacks", "fpr", "min_psnr", "max_ms")}
            threshold = beta_from_fpr(request["fpr"])

            for method in METHODS:
                proposal = propose(request, method, oracle)
                config = proposal["config"]
                if method == "H2_rule":
                    replayed += 1
                cfg_id = canonical_hash(config)
                audits[method]["config_counts"][cfg_id] += 1
                audits[method]["unknown_attacks"].update(proposal["policy_evidence"]["coverage_unknown"])
                if len(config["fe_on"]) > 1:
                    audits[method]["dual_fe_per_class"][group] += 1
                for arm in ("04", "06"):
                    method_rows[method].append(dict(row_id=f"{group}/{arm}/{record['i']}", group=group,
                                                   arm=arm, request_index=record["i"], request=request,
                                                   config_id=cfg_id, **proposal))
    # Recheck every source after the complete replay before publishing frozen choices.
    if sources() != fingerprints:
        raise ValueError("source changed while preparing baseline policies")
    outputs = {}
    for method in METHODS:
        value = dict(schema="baseline_policy_proposals_v1", method=method, protocol=POLICY_VERSION,
                     source_hashes=fingerprints, oracle_path=str(root / "inputs/historical_oracle.json"),
                     oracle_sha256=digest(root / "inputs/historical_oracle.json"),
                     rows=method_rows[method], audit=dict(**audits[method],
                         requests=10000, comparison_rows=20000,
                         arms_are_paired_references=True, policy_uses_final_observations=False,
                         unified_greedy_requests=replayed if method == "H2_rule" else None))
        path = root / method / "plans/proposals.json"
        write_once(path, value)
        outputs[method] = dict(path=str(path), sha256=digest(path), rows=len(value["rows"]),
                               unique_configs=len(audits[method]["config_counts"]))
    manifest = dict(schema="baseline_policy_freeze_v1", source_hashes=fingerprints,
                    oracle=dict(path=str(root / "inputs/historical_oracle.json"),
                                sha256=digest(root / "inputs/historical_oracle.json")), methods=outputs,
                    assertions=dict(unified_h2_full_population=replayed == 10000,
                                    historical_configuration_equality_claimed=False,
                                    acceptance_protocol="unified_raw_ba_v1",
                                    all_proposals_frozen_before_baseline_final=True,
                                    capacity_is_output=True, no_main_final_observations_read=True,
                                    dual_fe_preserved_and_disclosed=True))
    write_once(root / "policy_freeze.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=B)
    args = parser.parse_args()
    prepare(args.root)
