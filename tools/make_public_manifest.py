#!/usr/bin/env python3
"""Write PUBLIC_MANIFEST.sha256: the sha256 of every file this repository ships.

Standard library only (plus the git command line when available). Re-run it after any change,
including after the website lands in docs/:

    python3 tools/make_public_manifest.py
    shasum -a 256 -c PUBLIC_MANIFEST.sha256        # from the repository root; every line must say OK

"Shipped" means what git would commit: inside a git work tree the file list comes from
`git ls-files --cached --others --exclude-standard`, so .gitignore applies. Without git it walks the
folder and skips .git/, __pycache__/, *.pyc and .DS_Store. Either way .git/ and the manifest itself
are never listed, and docs/ is included. Files present on disk but ignored by .gitignore are printed,
so nothing drops out unnoticed.

This manifest describes the public, sanitized copy. It is not the experiment's lock: the original
locks ship unchanged, and tools/verify_locks.py maps the few sanitized files back to the hashes those
locks record (see SANITIZATION.md).
"""
import hashlib
import os
import subprocess
import sys

NAME = "PUBLIC_MANIFEST.sha256"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def walk_all():
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
        for f in files:
            if f.endswith(".pyc") or f == ".DS_Store":
                continue
            out.append(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/"))
    return out


def git_list(*extra):
    res = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"] + list(extra),
                         capture_output=True, check=True)
    return [p for p in res.stdout.decode("utf-8").split("\0") if p]


def main():
    try:
        inside = subprocess.run(["git", "-C", ROOT, "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True, check=True).stdout.strip()
        use_git = os.path.realpath(inside) == os.path.realpath(ROOT)
    except (OSError, subprocess.CalledProcessError):
        use_git = False
    if use_git:
        files = git_list("--cached", "--others", "--exclude-standard")
        ignored = git_list("--others", "--ignored", "--exclude-standard")
        source = "git ls-files (respects .gitignore)"
    else:
        files, ignored, source = walk_all(), [], "directory walk (git not available)"
    files = sorted({f for f in files if f != NAME and not f.startswith(".git/")
                    and os.path.isfile(os.path.join(ROOT, f))})
    for f in files:
        if "\n" in f or "\\" in f:
            sys.exit("unsupported file name: %r" % f)
    lines = ["%s  %s\n" % (sha256(os.path.join(ROOT, f)), f) for f in files]
    tmp = os.path.join(ROOT, NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)
    os.replace(tmp, os.path.join(ROOT, NAME))
    docs = sum(1 for f in files if f.startswith("docs/"))
    print("wrote %s: %d files (%d under docs/), file list from %s" % (NAME, len(files), docs, source))
    real_ignored = sorted(f for f in ignored if "__pycache__/" not in f and not f.endswith((".pyc", ".DS_Store")))
    if real_ignored:
        print("ignored by .gitignore and NOT in the manifest (%d):" % len(real_ignored))
        for f in real_ignored:
            print("  " + f)
    print("verify: cd %s && shasum -a 256 -c %s" % (os.path.basename(ROOT) or ".", NAME))


if __name__ == "__main__":
    main()
