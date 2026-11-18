# Evidence behind the plan

This folder contains the material that the plan was built from. Nothing here is a step.

| File | What it is |
|---|---|
| `audit_*.md` | Six audits of the package, one per area, written by reading the code at commit `0edcad6`. |
| `verification.md` | A second pass that ran code to confirm or refute 35 claims of the audits. Where it disagrees with an audit, the verification is right. |
| `verify_scripts/` | The scripts of that pass. They reproduce the wrong behaviours of phase B and can become its tests. Their paths point to a scratch folder and need editing. |
| `conversions.txt` | The output of a script that called the nine expressions of W-01. |
| `literal_defaults.tsv` | Keyword defaults found in the signatures of the package, for S-05 and E-12. |
| `import_census.json` | The imports of every module of the package and of the tests, for S-03 and phase L. |

The last three files come from agents that were interrupted before they reported, so nobody has
reviewed them. Treat them as a starting point and produce them again before you rely on them.
The only commit after `0edcad6` changed logo files and three text files, so the line numbers of
the audits are valid at `39f42d1`.
