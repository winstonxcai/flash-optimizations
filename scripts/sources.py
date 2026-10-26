"""Read-only source gate shared by local and Modal launchers. Author: Winston Cai.

Run before builds or host-side benchmarks: python3 scripts/sources.py .
Explicit Modal source overrides must be Git checkouts; their dirty contents and
nested sources are fingerprinted, not silently treated as a pinned release.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def compiled_source_digest(roots: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in map(Path, roots):
        if not root.is_dir():
            raise RuntimeError(f"compiled source directory absent: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {
                ".cu",
                ".cuh",
                ".cc",
                ".cpp",
                ".h",
                ".hpp",
                ".cmake",
                ".txt",
                ".py",
            }:
                digest.update(str(path.relative_to(root)).encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()



def verify_compiled_image_sources(expected: str, roots: list[str], integration: str) -> None:
    """CPU-only reuse gate; never overlay runtime code on mismatched kernels."""
    if os.environ.get("REMNANT_BUILT_INTEGRATION", "v0.5.18") != integration:
        raise RuntimeError("Compiled image belongs to a different integration")
    if compiled_source_digest([Path(root) for root in roots]) != expected:
        raise RuntimeError("Compiled source changed; rebuild the selected Dockerfile")

def glm_rebuild_source_digest(roots: list[Path]) -> str:
    """Hash every compiled source except the two authorized GLM JIT inputs."""
    allowed = {"ops/attention/dsa/tilelang_kernel.py", "jit/include/sgl_kernel/glm_packed_decode.cuh"}
    digest = hashlib.sha256()
    for i, root in enumerate(map(Path, roots)):
        if not root.is_dir():
            raise RuntimeError(f"Missing CPU rebuild source root: {root}")
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            if i == 0 and relative in allowed:
                continue
            if path.is_file() and path.suffix in {".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp", ".cmake", ".txt", ".py"}:
                digest.update(f"{i}/{relative}".encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_glm_cpu_rebuild(expected_base: dict, candidate: str, roots: list[str]) -> None:
    """Verify a CPU JIT-only overlay; ordinary compiled-image reuse stays strict."""
    if os.environ.get("REMNANT_BUILT_INTEGRATION") != "v0.5.22":
        raise RuntimeError("GLM CPU rebuild requires the verified v0.5.22 image")
    manifest_path = Path("/opt/remnant-build.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_identities"]["sglang"]["commit"] != expected_base["sglang"] or manifest["source_identities"]["flashmla"]["commit"] != expected_base["flashmla"]:
        raise RuntimeError("CPU rebuild base source identity mismatch")
    for name, digest in {**manifest["binaries"], **manifest["glm_compile"]["cache_files"]}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"CPU rebuild base artifact changed: {name}")
    if glm_rebuild_source_digest([Path(root) for root in roots]) != candidate:
        raise RuntimeError("CPU rebuild found changes outside the authorized GLM loader/header")
    Path("/opt/remnant-cpu-rebuild-base.json").write_text(json.dumps(manifest))


def glm_compile_owner_digest(sglang: Path) -> str:
    """Hash JIT probe/factory owners, allowing unrelated test-harness edits.

    Production kernel trees have a separate whole-tree gate before any overlay.
    These test definitions own additional compile artifacts; changing one forces
    CPU compilation, while changing fixtures or assertions does not.
    """
    import ast

    test = sglang / "test/registered/attention/unittests/test_glm5_remnant.py"
    text = test.read_text()
    nodes = ast.parse(text).body
    names = {"_packed_loader_probe", "_compile_sm90_variant", "precompile_sm90", "ATTENTION_SCALE"}
    owners = {}
    for node in nodes:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = next((t.id for t in node.targets if isinstance(t, ast.Name) and t.id in names), None)
        if name in names:
            owners[name] = ast.get_source_segment(text, node)
    if set(owners) != names:
        raise RuntimeError("Missing GLM CPU compile owner definitions")
    owners["runtime"] = (sglang / "python/sglang/srt/remnant/glm.py").read_text()
    return hashlib.sha256(json.dumps(owners, sort_keys=True).encode()).hexdigest()


def snapshot_glm_artifacts(sglang: str) -> None:
    """Called on the verified base BEFORE overlaying candidate test sources."""
    manifest = json.loads(Path("/opt/remnant-build.json").read_text())
    manifest["compile_owner_digest"] = glm_compile_owner_digest(Path(sglang))
    Path("/opt/remnant-glm-reuse-base.json").write_text(json.dumps(manifest))


def glm_partial_launch_metadata(compiled: dict) -> None:
    """Bind exact production cubins, generated ABI and dynamic shared bytes.

    The qualified TileLang version uses this public CUDA cache key API. Resolve
    it on CPU, require a manifest hash, and fail before H100 on any mismatch.
    """
    import tilelang
    from tilelang.cache.cuda_binary_cache import CUDABinaryCache

    package = Path(tilelang.__file__).parent
    options = ["-std=c++20", f"-I{package}/3rdparty/../src", f"-I{package}/3rdparty/cutlass/include"]
    cache = compiled["cache_files"]
    for variant in compiled["tilelang"]:
        matches = [Path(p) for p,h in cache.items()
                   if p.endswith("/device_kernel.cu") and h == variant["source_sha256"]]
        if not matches:
            raise RuntimeError(f"Missing generated GLM source: {variant}")
        source = matches[0].read_text()
        key = CUDABinaryCache.make_key(code=source, target_kind="cuda", target_arch="90a",
                target_code=[], compile_format="cubin", options=options)
        cubin = Path(CUDABinaryCache.get_path(key, "cubin"))
        if str(cubin) not in cache or hashlib.sha256(cubin.read_bytes()).hexdigest() != cache[str(cubin)]:
            raise RuntimeError(f"Unverified production GLM cubin: {cubin}")
        symbol = "glm_packed_partial_kernel" if variant["packed"] else "main_kernel"
        match = re.search(r"\b" + symbol + r"\(([^)]*)\)\s*\{", source)
        if not match:
            raise RuntimeError(f"Missing production CUDA definition: {symbol}")
        parameters = []
        for declaration in match[1].split(","):
            declaration = declaration.strip()
            name = declaration.split()[-1]
            pointer = "*" in declaration
            if not pointer and declaration != f"int {name}":
                raise RuntimeError(f"Unsupported GLM CUDA ABI: {declaration}")
            parameters.append(dict(name=name, pointer=pointer, type="pointer" if pointer else "int"))
        expected = (["bitmap"] if variant["packed"] else []) + ["indices","kv_fp8","partial_lse","partial_o","q_fp8","num_pages","seq_len"]
        if [p["name"] for p in parameters] != expected:
            raise RuntimeError(f"GLM partial ABI changed: {parameters}")
        host = (matches[0].parent / "host_kernel.cu").read_text()
        sizes = re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\);", host)
        if not sizes:
            raise RuntimeError("Missing generated dynamic shared-memory size")
        variant.update(cubin_path=str(cubin), cubin_sha256=cache[str(cubin)], symbol=symbol,
                       parameters=parameters, dynamic_shared=int(sizes[-1]))


def integration_profile(value: str | None = None) -> dict[str, str]:
    """One explicit source/image pair; Native defaults remain v0.5.18."""
    version = (
        value if value is not None else os.environ.get("REMNANT_INTEGRATION", "v0.5.18")
    )
    if version not in ("v0.5.18", "v0.5.22"):
        raise ValueError(
            f"Unsupported REMNANT_INTEGRATION={version!r}; use v0.5.18 or v0.5.22"
        )
    newer = version == "v0.5.22"
    return {
        "version": version,
        "sglang": "third_party/sglang-v0.5.22" if newer else "third_party/sglang-v0.5.18",
        "flashmla": "third_party/flashmla-v0.5.22" if newer else "third_party/flashmla-v0.5.18",
        "base_image": (
            "lmsysorg/sglang:v0.5.22" if newer else "lmsysorg/sglang:v0.5.18-cu130"
        ),
        "container": "remnant-v0522" if newer else "remnant",
        "dockerfile": f"Dockerfile.{version}",
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            stderr=subprocess.PIPE,
            timeout=30,
        ).rstrip("\n")
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Source preflight cannot inspect {root}: {error}"
        ) from error


def check_sources(
    repo: Path,
    overrides: dict[str, Path] | None = None,
    *,
    integration: str | None = None,
) -> dict:
    """Require exact clean fork pins, or record explicitly overridden Git trees.

    Parent edits are allowed. Fork staged, unstaged and untracked changes are not,
    unless the caller explicitly selects that fork through an experiment override.
    This checks submitted source, not the identity of an already-built image.
    """
    repo = repo.resolve()
    profile = integration_profile(integration)
    overrides = overrides or {}
    unknown = overrides.keys() - {"sglang", "flashmla"}
    if unknown:
        raise ValueError(f"Unknown source overrides: {sorted(unknown)}")
    records = {"parent": {"path": str(repo), "commit": _git(repo, "rev-parse", "HEAD")}}
    recovery = (
        "Inspect and preserve changes first. For a clean checkout, run "
        "git submodule update --init --recursive --checkout from the parent; "
        "this preflight never repairs, stashes or discards work."
    )

    def inspect(root: Path, name: str, expected: str, override: bool) -> None:
        root = root.resolve()
        # Git otherwise walks up to the parent when a submodule is uninitialized.
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise RuntimeError(
                f"Source {name} is not an initialized Git checkout: {root}. {recovery}"
            )
        actual = _git(root, "rev-parse", "HEAD")
        if actual != expected and not override:
            raise RuntimeError(
                f"Source {name} pin mismatch: expected {expected}, actual {actual} at {root}. {recovery}"
            )
        for entry in _git(root, "ls-tree", "-rz", "HEAD").split("\0"):
            if entry.startswith("160000 "):
                metadata, child = entry.split("\t", 1)
                inspect(root / child, f"{name}/{child}", metadata.split()[2], override)
        dirty = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        if dirty and not override:
            raise RuntimeError(
                f"Source {name} is dirty at {root} (expected/actual {expected}):\n{dirty}\n{recovery}"
            )
        record = {
            "path": str(root),
            "expected_commit": expected,
            "commit": actual,
            "override": override,
            "dirty": bool(dirty),
        }
        if override:
            digest = hashlib.sha256()
            files = _git(
                root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
            ).split("\0")
            for relative in sorted(set(files) - {""}):
                path = root / relative
                digest.update(relative.encode() + b"\0")
                if path.is_symlink():
                    digest.update(
                        b"link\0"
                        + hashlib.sha256(str(path.readlink()).encode()).digest()
                    )
                elif path.is_file():
                    digest.update(
                        b"file\0" + hashlib.sha256(path.read_bytes()).digest()
                    )
                elif not path.exists():
                    digest.update(b"deleted\0")
            record["sha256"] = digest.hexdigest()
        records[name] = record

    for name in ("sglang", "flashmla"):
        relative = profile[name]
        entry = _git(repo, "ls-tree", "HEAD", "--", relative)
        if not entry.startswith("160000 "):
            raise RuntimeError(f"Parent HEAD has no submodule pin for {relative}")
        source = overrides.get(name, repo / relative)
        inspect(source, name, entry.split()[2], name in overrides)
        records[name]["integration"] = profile["version"]
        if name == "sglang":
            version_file = source / "python/sglang/version.py"
            if version_file.is_file():
                match = re.search(
                    r"^__version__\s*=\s*[\"\x27]([^\"\x27]+)",
                    version_file.read_text(),
                    re.MULTILINE,
                )
                if match is not None and match[1] != profile["version"].lstrip("v"):
                    raise RuntimeError(
                        f"SGLang version does not match {profile['version']}: {version_file}"
                    )
            if version_file.exists():
                grouped_args = (
                    source / "python/sglang/srt/arg_groups/fields/model.py"
                ).is_file()
                if grouped_args != (profile["version"] == "v0.5.22"):
                    raise RuntimeError(
                        f"SGLang argument-group layout does not match {profile['version']}: {source}"
                    )
        # Overrides must not mix the incompatible FlashMLA source layouts.
        if name == "flashmla":
            newer_layout = (
                source / "csrc/kernels/sm90/decode/sparse/config.h"
            ).is_file()
            legacy_layout = (source / "csrc/sm90/decode/sparse_fp8/config.h").is_file()
            if (newer_layout or legacy_layout) and newer_layout != (
                profile["version"] == "v0.5.22"
            ):
                raise RuntimeError(
                    f"FlashMLA source layout does not match {profile['version']}: {source}"
                )
    return records


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: python3 scripts/sources.py <repository>")
        print(json.dumps(check_sources(Path(sys.argv[1])), indent=2))
    except (RuntimeError, ValueError) as error:
        sys.exit(str(error))
