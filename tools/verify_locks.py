#!/usr/bin/env python3
"""Check every lock file in this repository against the shipped files.

Standard library only. Run from anywhere:

    python3 tools/verify_locks.py            # summary per lock, plus every SANITIZED and FAIL entry
    python3 tools/verify_locks.py -v         # every entry
    python3 tools/verify_locks.py --json out.json

Each lock entry gets one status:

  OK           the shipped file's sha256 equals the lock entry.
  SANITIZED    the shipped file differs, SANITIZATION.json lists it, its recorded original_sha256
               equals the lock entry and its recorded public_sha256 equals the shipped file. The lock
               pins the private original; the public difference is documented, not hidden.
  NOT_SHIPPED  the file is not in this repository and SANITIZATION.json "not_shipped" names it
               (upstream code at a pinned commit, run outputs, held-back evidence).
  FAIL         anything else: a mismatch that is not documented, or a missing file nobody declared.

It also checks that every file SANITIZATION.json lists still has its recorded public sha256.
Exit status is 0 only when there are no FAILs.

Lock files read: every *.sha256 (except PUBLIC_MANIFEST.sha256), prerequisites/**/source-hashes.txt,
JSON source locks (*source-lock*.json), frozen_B_adapter_v1r2/manifest.json and the ABBA run's VM
copy manifests (evidence/abba_l4003/copy-manifest.*.json). docs/ and .git/ are skipped.
frozen_B_adapter_v1r2/files/models/model-manifest.json is not read: it pins model weights in the
Hugging Face cache, which are never shipped.

Path rules: entries under the runtime root /home/miodrag/robot-transfer/bowl/ map to this repository's
root. Outside experiments/, which ships only at the root, they fall back to frozen_B_adapter_v1r2/files/
and prerequisites/, which mirror that root for the pinned inputs (runs/, models/, RoboLab/output/).
lock-relative.sha256 is relative to frozen_B_adapter_v1r2/files/; the step 5 JSON locks are relative to
the repository root; every other relative entry is relative to its lock file's folder.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import posixpath
import sys

RUNTIME_ROOT = "/home/miodrag/robot-transfer/bowl/"
FALLBACK_ROOTS = ("", "frozen_B_adapter_v1r2/files/", "prerequisites/")
SKIP_DIRS = {".git", "docs", "__pycache__"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_locks(root):
    locks = []
    for d, dirs, files in os.walk(root):
        rel_d = os.path.relpath(d, root)
        dirs[:] = sorted(x for x in dirs if not (x in SKIP_DIRS and (x != "docs" or rel_d == ".")))
        for f in sorted(files):
            rel = posixpath.normpath(os.path.join(rel_d, f).replace(os.sep, "/"))
            if f.endswith(".sha256") and f != "PUBLIC_MANIFEST.sha256":
                locks.append(rel)
            elif f == "source-hashes.txt" and rel.startswith("prerequisites/"):
                locks.append(rel)
            elif f.endswith(".json") and ("source-lock" in f or f.startswith("copy-manifest.")):
                locks.append(rel)
            elif rel == "frozen_B_adapter_v1r2/manifest.json":
                locks.append(rel)
    return locks


def read_entries(root, lock):
    """Yield (key, sha256, where) for one lock file."""
    path = os.path.join(root, lock)
    if lock.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data.get("files"), dict):
            return [(k, v, "files." + k) for k, v in data["files"].items()]
        if isinstance(data.get("files"), list):
            return [(x["path"], x["sha256"], "files[%d]" % i) for i, x in enumerate(data["files"])]
        return [(k, v, k) for k, v in data.items() if isinstance(v, str) and len(v) == 64]
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            out.append((name.lstrip("*"), digest, "line %d" % n))
    return out


def rooted(rel):
    # experiments/ ships at the repository root and only there; the other runtime inputs
    # (runs/, models/, RoboLab/output/) ship under the mirror folders.
    if rel.startswith("experiments/"):
        return [rel]
    return [posixpath.normpath(r + rel) for r in FALLBACK_ROOTS]


def candidates(lock, key):
    if key.startswith(RUNTIME_ROOT):
        rel = key[len(RUNTIME_ROOT):]
        return rooted(rel), rel
    if key.startswith("/"):
        return [], key
    if lock == "frozen_B_adapter_v1r2/lock-relative.sha256":
        base = "frozen_B_adapter_v1r2/files"
    elif lock.startswith("experiments/bowl_step5/") and lock.endswith(".json"):
        base = ""
    else:
        base = posixpath.dirname(lock)
    rel = posixpath.normpath(posixpath.join(base, key))
    if base == "":
        return rooted(rel), rel
    return [rel], rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json", help="write a full per-entry report here")
    args = ap.parse_args()
    root = args.root

    san = json.load(open(os.path.join(root, "SANITIZATION.json"), encoding="utf-8"))
    documented = {e["public_path"]: e for e in san["files"]}
    not_shipped = san.get("not_shipped", [])

    def declared_absent(lock, rel):
        for rule in not_shipped:
            if fnmatch.fnmatch(rel, rule["pattern"]) and fnmatch.fnmatch(lock, rule.get("lock", "*")):
                return rule["reason"]
        return None

    hashes = {}

    def h(rel):
        if rel not in hashes:
            hashes[rel] = sha256(os.path.join(root, rel))
        return hashes[rel]

    report, totals = [], {"OK": 0, "SANITIZED": 0, "NOT_SHIPPED": 0, "FAIL": 0}
    for lock in find_locks(root):
        counts = {k: 0 for k in totals}
        rows = []
        for key, locked, where in read_entries(root, lock):
            cands, rel = candidates(lock, key)
            present = [c for c in cands if os.path.isfile(os.path.join(root, c))]
            status, detail, path = "FAIL", "", (present[0] if present else rel)
            match = next((c for c in present if h(c) == locked), None)
            if match:
                status, path = "OK", match
                if match != cands[0]:
                    detail = "found at " + match
            elif present:
                e = documented.get(path)
                if e and e["original_sha256"] == locked and e["public_sha256"] == h(path):
                    status = "SANITIZED"
                    detail = "%s; original %s, shipped %s; see SANITIZATION.json" % (
                        e["kind"], locked[:12], e["public_sha256"][:12])
                elif e:
                    detail = "listed in SANITIZATION.json but hashes do not match (lock %s, recorded original %s, recorded public %s, shipped %s)" % (
                        locked[:12], e["original_sha256"][:12], e["public_sha256"][:12], h(path)[:12])
                else:
                    detail = "sha256 %s differs from lock %s and is not documented" % (h(path)[:12], locked[:12])
            else:
                reason = declared_absent(lock, rel)
                if reason:
                    status, detail = "NOT_SHIPPED", reason
                else:
                    detail = "missing and not declared in SANITIZATION.json not_shipped"
            counts[status] += 1
            rows.append({"lock": lock, "entry": where, "path": path, "status": status, "detail": detail})
        for k in totals:
            totals[k] += counts[k]
        report.extend(rows)
        print("%-66s %3d entries  OK %3d  SANITIZED %d  NOT_SHIPPED %3d  FAIL %d" % (
            lock, len(rows), counts["OK"], counts["SANITIZED"], counts["NOT_SHIPPED"], counts["FAIL"]))
        for r in rows:
            if args.verbose or r["status"] in ("SANITIZED", "FAIL"):
                print("    %-11s %s (%s)%s" % (r["status"], r["path"], r["entry"], ": " + r["detail"] if r["detail"] else ""))

    print("\nSANITIZATION.json self-check")
    san_fail = 0
    for e in san["files"]:
        p = os.path.join(root, e["public_path"])
        if not os.path.isfile(p):
            ok, why = False, "file missing"
        elif h(e["public_path"]) != e["public_sha256"]:
            ok, why = False, "shipped sha256 %s != recorded public_sha256 %s" % (h(e["public_path"])[:12], e["public_sha256"][:12])
        else:
            ok, why = True, ""
        if not ok:
            san_fail += 1
            print("    FAIL        %s: %s" % (e["public_path"], why))
        elif args.verbose:
            print("    OK          %s" % e["public_path"])
    print("    %d documented files, %d match their recorded public sha256" % (len(san["files"]), len(san["files"]) - san_fail))

    totals["FAIL"] += san_fail
    print("\nTOTAL lock entries: %d  OK %d  SANITIZED %d  NOT_SHIPPED %d  FAIL %d (including %d SANITIZATION.json self-check failures)" % (
        len(report), totals["OK"], totals["SANITIZED"], totals["NOT_SHIPPED"], totals["FAIL"], san_fail))
    if args.json:
        json.dump({"totals": totals, "entries": report}, open(args.json, "w"), indent=1)
    print("RESULT: " + ("PASS" if totals["FAIL"] == 0 else "FAIL"))
    return 0 if totals["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
