# Sanitization record

This repository is a sanitized public copy of the experiment, not a byte-identical one. 21 shipped files differ from their originals in the private project; every other file with a private original is byte-identical to it. The private originals are preserved unchanged. `SANITIZATION.json` is the machine-readable form of this page: for each file it gives the private source path, both sha256 values, the changed line numbers, the pattern class and placeholder, the reason, and the lock lines that pin it. Private source paths are written relative to the private project folder as `<project-root>/`. No replaced value is printed here or there.

## Placeholders

| Placeholder | Replaces | Why |
| --- | --- | --- |
| `<project-root>/` | the absolute path of the private project folder on the author's Mac (`/Users/<user>/Desktop/<project>/`) | A personal machine path. Nothing reads it to score or reproduce; the scorers take the evidence root as an argument. |
| `<vm-instance-id>` | the Nebius compute instance ID of the project VM, which was also its hostname | It ties the repository to one cloud account resource. A re-runner needs their own VM and ID. |

The runtime root `/home/miodrag/robot-transfer/bowl/` is kept everywhere, as decided on 24 October: the runners hardcode it and every source lock records it (README, "Root path").

## Files that differ from their originals

Five kinds of change. Only the redactions remove information; the rest are labelled public-copy edits.

| File | Kind | Change | Pinned by a lock | Original sha256 | Public sha256 |
| --- | --- | --- | --- | --- | --- |
| `experiments/scoring_v5/score_step8_mock/score_step8-mock.json` | Redaction | 53× local Mac project prefix → `<project-root>/`; 50 line ranges, 5-8 to 1039 | `r1-lock-package.sha256` line 13 | `6e4443faba0079aa0b37d1b2740d4e375822b722cd97f44966bd1ff9bb0cac31` | `b30aeacc795ebb5bd025fb4e73aa0d8b515ddb3e3490272661cae92e4434d030` |
| `experiments/scoring_v5/validation-pose-variant.json` | Redaction | 39× local Mac project prefix → `<project-root>/`; 39 line ranges, 13 to 608 | `pose-variant.sha256` line 4 | `1322c5493435c4139ac5abc903197c279c18e73219f268a62330220d2221fc5c` | `3324b603fb65ebf81eaab410cb4542d4e1e5b1dea50c2a945b4154e9cbf9e675` |
| `experiments/scoring_v5/score_step8_before/score_step8-before.json` | Redaction | 85× local Mac project prefix → `<project-root>/`; 73 line ranges, 5 to 1481 | no | `372bdd0f10424179e8c47e1966a542dfa3f2f8682479d64623d73be697443aa5` | `c37b17f38876f8a9c5ca802c0a7595f29e8eef8705c5d09c937051d33f682df4` |
| `experiments/scoring_v5/pose_variant_before` | Redaction | 1× local Mac project prefix → `<project-root>/`; line 4 | no | `fa81c88bb346bd2607432a9f1fb7348d6451b6f5cf81830d8f6d0f1700412d78` | `5c589173af85785778a9eb4255aa009ce1f46986078f0680aa43660b125ddf86` |
| `experiments/scoring_v5/score_step8_after/score_step8-after.json` | Redaction | 178× local Mac project prefix → `<project-root>/`; 151 line ranges, 5-6 to 2924 | no | `3f945e1fe3b848848a83025788e01d4079b0457e9d83605bcdabadd8652b7198` | `94d85a79b2643a662c20feca89d551fcf978e5064fe8195f87bd5084a2124040` |
| `experiments/scoring_v5/score_select_v1_raw/score_step8-select.json` | Redaction | 161× local Mac project prefix → `<project-root>/`; 121 line ranges, 5 to 2150 | no | `534b54196e57ecc05d10f841d8680988a9135aa3da791dd04a6c82302782f1c4` | `b2b4c4fed1a27a042479d74347bac03825ed163f83b156f2de46d3e95a6c9a12` |
| `experiments/scoring_v5/score_select_v1_adj/score_step8-select.json` | Redaction | 161× local Mac project prefix → `<project-root>/`; 121 line ranges, 5 to 2150 | no | `194c47eec8aa7a846ba710edeb700da381aee27423b2e2b6c2af9949b811e1f3` | `099e248791610bffaa3caa7e29cf974d7d65a03e2ff777f05cde8432da1018da` |
| `experiments/charlie_bravo_perception_v1/runner-provenance.json` | Redaction | 1× local Mac project prefix → `<project-root>/`; line 2 | no | `8fe858cf89ffd17c1ba4f85fba13c9292495d8b8904d0a54d7abe267af4e31f6` | `56f495d99b5970973f4c7802b923f40db5dc4857b2fe5b89a708253134edfba0` |
| `experiments/charlie_bravo_perception_v1/preflight.py` | Redaction | 1× VM instance ID → `<vm-instance-id>`; line 5 | `source-lock.original.json` line 9 | `4c26d74efd14af3013ecdbc9a3418743ab6734fa284692b6fa55b613400e5ae4` | `3778689d2240603275f12abb77a97a0802ab0d1050155ec08bce608955c76a8d` |
| `experiments/charlie_bravo_perception_v1/permit.example.json` | Redaction | 1× VM instance ID → `<vm-instance-id>`; line 15 | no | `fef19bfaf76b03b9f1242c98514575e43d9bfd1da9b506a011294c17c1e3a35e` | `2600a1510f615c9727c47fdecaf9bbf69d9320354904ed194dbdf22739fb5355` |
| `experiments/roger_bravo_one_shot_v1/preflight.py` | Redaction | 1× VM instance ID → `<vm-instance-id>`; line 5 | `source-lock.json` line 6 | `818896a1786120873300b56ef9fb6f536b69acb7ed8666fc209ff78eaa4e25a2` | `67ce9c7f497633d1c182588e19b6a80991e56a04c8427fb02650b6fa313d39aa` |
| `experiments/roger_bravo_one_shot_v1/permit.example.json` | Redaction | 1× VM instance ID → `<vm-instance-id>`; line 15 | no | `3a54b1a7647dacec4c893aac37c95e6af72ff33b159f392e0a0ca4f124fbf70e` | `ea92fe6103b80f2d3a9506f700e58cc95a04ade458b2014565d978c0e2b16fa8` |
| `experiments/bowl_step6/B_P0_DIAGNOSIS.md` | Redaction | internal model nickname → neutral wording; lines 55, 61, 68 | no | `b1d947cc1abf3ffcae214b3c24c37ad43c858d41b177b554f3f066f824165d22` | `785a0066449e68c336602d1f77dbf12bdd5efed011ae95c3fadae7b7b6c33530` |
| `experiments/bowl_step6/inspect_kinova_mechanics.py` | Licence header | inserted public lines 1-5 | no | `7aa19c4e7629b183a85bf2dbc595cd02d3e2e429c7d896940ceff1e35ec032a6` | `77ab6744fc7ec38d3bab48984811e7c80b7f576d2b0b0821907f9d65e8eeb17b` |
| `experiments/charlie_bravo_v1/tests/test_vision_images.py` | Test adjustment | inserted public lines 11 | `source-lock.original.json` line 12 | `9d6bd0e78ac47ded3e1bd37df2ba925203ff1baba14b43256291ea52e56dbb00` | `9bc87abe6a3ea6affaedfa0ec8c9fa04b49824d4621a5b5664d337edcd669799` |
| `experiments/charlie_bravo_v1/source-lock.json` | Export lock | removed original lines 2; original lines 12 → public 11 | no | `e6019e48ac93d8d3bd731e3fbcb3229382fa6cb7f79058784ed30d2ea8e16505` | `08b53827dbf480a591370d8c6c61ce5f0d20708703b5bfabd973c43d84edbab7` |
| `experiments/charlie_bravo_perception_v1/source-lock.json` | Export lock | original lines 9 → public 9 | no | `240e60c7f3881a31b852a4561fc202481e0cb84545cb93a48f7efa88718b849e` | `ad8db1f8287442954ffff3982f42f1b7dd97f1cfe44c6291619e58d103546ae6` |
| `experiments/roger_bravo_one_shot_v1/README.md` | Public README | inserted public lines 1-2 | no | `28baa324a6febeda2bb0bca86ee7a182f0ed1090483f4a31e76777fa36176064` | `4f473328813f89d1feb74d8a7507c0bc72f3c3d54fe51df3823c12c315761f76` |
| `experiments/charlie_bravo_v1/README.md` | Public README | original lines 1 → public 1; original lines 3 → public 3; removed original lines 5-6 | no | `8f0472af1176df0847e6083cdf198dfa25c140ec5e9917a02e007978ec942689` | `5ea4594675e16dd64326cc0f4103498db39a14fc29793507a4bef58eff2cb5ea` |
| `experiments/charlie_bravo_perception_v1/README.md` | Public README | original lines 1 → public 1; original lines 3 → public 3; removed original lines 5-41 | no | `3d8b3abcec3b641bade8643e48f9da204162aabaeb4a1930a818854f8e46bc41` | `35c66a06273a90b88c3c3ae9c7e4ed3d89edebcd7f684a6b7fc110ae4e47825e` |
| `nemotron/README.md` | Public README | rewritten (original lines 1-81 → public 1-18) | no | `08169c0719caa523b2c279101985e1a51f89c59b281ac201e5d9748ae4fb1dc4` | `41d819643c3002aa39076aca9af26d7162cae1f8f95e5eb90b05e5c99b6e1475` |

Dates: the two locked scoring JSONs (first two rows) were sanitized on 27 October; the ABBA rows on 26 October; the rest on 24–25 October (see `SCAN.md`). Each JSON redaction is a plain prefix replacement: the file still parses and differs from the original only in that prefix.

Notes on individual files:

- `score_step8-mock.json`: `experiments/scoring_v5/SCORING_V5.md` line 477, itself locked and left as it was, still cites the original sha256 `6e4443fa…`.
- `validation-pose-variant.json`: running `validate_pose_variant.py` writes this file again, with local absolute paths.
- `score_step8-before.json`: `experiments/intent_demo_v1/validation/checker-offline-v1.json` records the original sha256 `372bdd0f…`, i.e. the unredacted score file.
- The two Charlie `source-lock.json` files are export snapshot locks that describe the shipped files. The original locks ship unchanged beside them as `source-lock.original.json`.
- The ABBA `source-lock.json` is unchanged (sha256 `166a1bb9fc5e2efdbb7befae002bc59ca492f1f7c09bdc340fd613c4a1a91b83`, the value the run's launch permit pinned); its `preflight.py` entry pins the original.

## Renamed, bytes unchanged

| Public path | Private source |
| --- | --- |
| `experiments/charlie_bravo_v1/source-lock.original.json` | `<project-root>/experiments/charlie_bravo_v1/source-lock.json` |
| `experiments/charlie_bravo_perception_v1/source-lock.original.json` | `<project-root>/experiments/charlie_bravo_perception_v1/source-lock.json` |
| `evidence/abba_l4003/copy-manifest.run.json` | `<project-root>/review/roger_bravo_one_shot_v1/evidence/roger_B_P1_l4003_assisted_v1/copy-manifest.json` |
| `evidence/abba_l4003/copy-manifest.supervisor.json` | `<project-root>/review/roger_bravo_one_shot_v1/evidence/roger_B_P1_l4003_assisted_v1-supervisor/copy-manifest.json` |

`frozen_B_adapter_v1r2/`, `setup/`, `prerequisites/` and `evidence/abba_l4003/` were copied byte for byte from the private evidence folders listed under `relocated_byte_identical` in `SANITIZATION.json`. Files written for this repository (README, NOTICE, SCAN.md, this record, `tools/`, `docs/` and the others under `public_only`) have no private original.

## How to check

From the repository root, Python 3 standard library only:

```bash
shasum -a 256 -c PUBLIC_MANIFEST.sha256     # the public copy is complete and unmodified: every line OK
python3 tools/verify_locks.py               # every lock entry: OK, SANITIZED or NOT_SHIPPED; exit 0 means no FAIL
```

`PUBLIC_MANIFEST.sha256` describes this sanitized copy; `tools/make_public_manifest.py` rewrites it after any change. `tools/verify_locks.py` reads every lock and maps each sanitized file back to its original: an entry is SANITIZED only when the file is listed here, the recorded original sha256 equals the lock entry, and the shipped file still has the recorded public sha256. An entry whose file is not shipped passes only when `SANITIZATION.json` (`not_shipped`) says why: upstream RoboLab code at the pinned commit, regenerable or bulky run outputs, one private review record, and the ABBA files held back from the repository. Anything else fails.

Result on 27 October: 36 lock files, 855 entries: 553 OK, 5 SANITIZED, 297 NOT_SHIPPED, 0 FAIL. The 5 SANITIZED entries are `validation-pose-variant.json`, `score_step8-mock.json`, ABBA's `preflight.py`, and, against the two Charlie original locks, the perception `preflight.py` and `tests/test_vision_images.py`.

To get the experiment's exact bytes back for a sanitized file, take it from the private original; for the JSON redactions, replacing `<project-root>/` with the original folder path restores the original sha256.
