# VDI → GitHub → local

Author: Winston Cai.

No packaging or helper scripts: copy a selected folder, push it, pull locally.

| Folder | Purpose | Git |
|---|---|---|
| `results/` | Raw VDI/Modal outputs; scripts unchanged | Ignored |
| `transfer/<original-folder-name>/` | Selected files in transit | Tracked; LFS for large reports |
| `raw/` | Local downloaded/archive copies | Ignored |
| `benchmarks/` | Curated date-labeled evidence | Tracked; LFS for profiler reports |

## VDI

Select only needed summaries, manifests and reports. Preserve original run IDs.
Do not include credentials, raw prompts, model weights, builds or copied sources.
Use a new destination; never overwrite an existing transfer.

```bash
cd /home/jovyan/winstonxcai/flash-optimizations
test "$(git branch --show-current)" = remnant
run=YOUR_SELECTED_FOLDER
test -d "results/$run" && test ! -e "transfer/$run"
mkdir -p transfer
cp -a "results/$run" "transfer/$run"
git add -- .gitattributes "transfer/$run"
git diff --cached --stat -- .gitattributes "transfer/$run"
git commit --only -m "Transfer $run" -- .gitattributes "transfer/$run"
git push origin HEAD:remnant
```

LFS rules already cover `.nsys-rep`, transferred `.ncu-rep` and `.sqlite.gz`.
For another large file type, add a narrow `git lfs track` rule before staging.
Review the copied folder before committing; LFS does not make secrets safe.

## Local

Protect local changes before pulling. If a pull conflicts, stop rather than
overwriting work. Fetch/download only the transfer paths when a full pull is unsafe.

```bash
git pull --ff-only origin remnant
run=YOUR_SELECTED_FOLDER
git lfs pull --include="transfer/$run/**" --exclude=""
test -d "transfer/$run" && test ! -e "raw/$run"
mkdir -p raw
mv "transfer/$run" "raw/$run"
```

Verify sizes/checksums against existing manifests, then manually copy selected
evidence from `raw/` into `benchmarks/`. Keep full originals in VDI `results/`.
Review and commit the curated files and removal of the downloaded transfer paths;
do not add the ignored archive. Keep this README in `transfer/`.

No automatic curation, folder flattening or fabricated provenance. Record missing
coverage and screenshots. Removing transfer paths from Git does not necessarily
reclaim GitHub LFS storage; transfer only what is needed.
