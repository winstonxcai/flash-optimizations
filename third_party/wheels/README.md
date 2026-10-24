<!-- Author: Winston Cai. -->

# Offline serving-image dependency

Both root versioned Dockerfiles install `typing_extensions` from this directory
because VDI build workers cannot reliably access PyPI or its mirrors. This 45,571-byte
universal Python wheel is a deliberate Git-tracked build input, not a generated
benchmark artifact. It does not use Git LFS, so a normal pull includes its bytes.
Keep the dependency copy/install steps after the CUDA build to preserve caching.

## Provenance

- Package: `typing_extensions` 4.16.0, universal `py3-none-any` wheel.
- Upstream: <https://pypi.org/project/typing-extensions/4.16.0/>.
- Downloaded from the Tsinghua mirror; its SHA-256 matches the upstream release.
- The canonical version and wheel digest are in `requirements.txt`. Pip enforces
  that digest with `--require-hashes` and uses `--no-index --no-deps`.
- Upstream PSF-2.0 license is preserved inside the wheel at
  `typing_extensions-4.16.0.dist-info/licenses/LICENSE`.

For an intentional version update, obtain the upstream universal wheel on a
networked machine, verify its published SHA-256, update `requirements.txt`, and
replace the old wheel here. Commit the wheel and manifest together. Do not make
the VDI build download dependencies or silently fall back to another version.
