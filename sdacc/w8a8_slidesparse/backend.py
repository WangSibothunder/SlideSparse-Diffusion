"""Explicit current-ABI build/load; never download/load foreign ABI binaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch

CUTLASS_REVISION = "0b55a2f691d69981583568fd9eb69687b1f0de8a"
_loaded = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_native(output, cutlass_headers, headers_manifest=None, fused_projection=False):
    """Offline build, without initializing CUDA; call before measurement.

    cutlass_headers must be the pinned official CUTLASS checkout. Supply the
    frozen headers_manifest to verify all dependency contents before building.
    """
    from torch.utils.cpp_extension import include_paths, library_paths
    if torch.cuda.is_initialized():
        raise RuntimeError("build in a fresh CPU-only process")
    output, headers = Path(output), Path(cutlass_headers)
    source = Path(__file__).parent / "csrc/dualscale_sm80.cu"
    compiled_source = source.parent / "projection_sm80.cu" if fused_projection else source
    pinned_manifest = Path(__file__).parent / "reference/cutlass_headers_manifest.json"
    headers_manifest = Path(headers_manifest) if headers_manifest else pinned_manifest
    if sha(headers_manifest) != sha(pinned_manifest):
        raise ValueError("dependency manifest differs from the qualified CUTLASS pin")
    if headers_manifest:
        manifest = json.loads(Path(headers_manifest).read_text())
        entries = (list(dict(path=p, sha256=s) for p, s in manifest.items())
                   if all(isinstance(v, str) for v in manifest.values())
                   else manifest.get("files", manifest.get("entries", [])))
        if not entries:
            raise ValueError("empty dependency identity")
        for e in entries:
            relative = e.get("relative", e.get("path"))
            expected = e.get("sha256", e.get("sha"))
            if sha(headers / relative) != expected:
                raise ValueError("CUTLASS dependency hash mismatch: " + relative)
    output.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"),
           "-std=c++17", "-O3", "--shared", "--expt-relaxed-constexpr",
           "--expt-extended-lambda", "-Xcompiler", "-fPIC", "-Xptxas=-v",
           "-gencode=arch=compute_80,code=sm_80",
           f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
           f"-I{headers}/include", f"-I{headers}/tools/util/include"]
    cmd += [f"-I{x}" for x in include_paths(cuda=True)]
    cmd += [f"-L{x}" for x in library_paths(cuda=True)]
    cmd += ["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
            "-Xlinker", f"-rpath,{Path(torch.__file__).parent}/lib",
            str(compiled_source), "-o", str(output / "dualscale_sm80.so")]
    log = output / "compile.log"
    with log.open("w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    receipt = dict(complete=True, source_sha=sha(source), library_sha=sha(output / "dualscale_sm80.so"),
                   torch=torch.__version__, cuda=torch.version.cuda,
                   abi=int(torch._C._GLIBCXX_USE_CXX11_ABI), command=cmd,
                   cutlass_revision=CUTLASS_REVISION,
                   headers_manifest_sha=sha(headers_manifest) if headers_manifest else None,
                   compile_log_sha=sha(log), cuda_initialized=torch.cuda.is_initialized())
    receipt['source_dependencies']={p.name:sha(p) for p in sorted(source.parent.iterdir()) if p.suffix in ('.cu','.cuh') and not p.name.startswith('._')}
    if fused_projection:
        receipt['projection_source_sha'] = sha(compiled_source)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_native(directory):
    """Load once and register the native op's compile-time shape inference."""
    global _loaded
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    revision = receipt.get("cutlass_revision", receipt.get("cutlass_pin", {}).get("revision"))
    dependencies={p.name:sha(p) for p in sorted((Path(__file__).parent/'csrc').iterdir()) if p.suffix in ('.cu','.cuh') and not p.name.startswith('._')}
    if receipt.get('source_dependencies') != dependencies:
        raise ValueError('native source dependency identity differs; rebuild this release')
    if receipt.get('projection_source_sha') and receipt['projection_source_sha'] != sha(Path(__file__).parent / 'csrc/projection_sm80.cu'):
        raise ValueError('native projection frontend source differs')
    if (not receipt.get("complete") or receipt["torch"] != torch.__version__
            or receipt["cuda"] != torch.version.cuda
            or receipt["abi"] != int(torch._C._GLIBCXX_USE_CXX11_ABI)
            or receipt["source_sha"] != sha(Path(__file__).parent / "csrc/dualscale_sm80.cu")
            or receipt["library_sha"] != sha(directory / "dualscale_sm80.so")
            or revision != CUTLASS_REVISION
            or receipt["headers_manifest_sha"] != sha(Path(__file__).parent / "reference/cutlass_headers_manifest.json")):
        raise ValueError("native source, binary, or Torch/CUDA ABI identity differs")
    if _loaded is not None:
        if _loaded != receipt["library_sha"]:
            raise RuntimeError("a different native binary is already loaded in this process")
        return receipt
    torch.ops.load_library(str(directory / "dualscale_sm80.so"))

    @torch.library.register_fake("sdacc_dualscale_sm80::run")
    def _fake(q, packed, metadata, sa, sw, bias, m, n, config, output_addend=None):
        return q.new_empty((m, n), dtype=torch.bfloat16)

    if receipt.get('projection_source_sha'):
        @torch.library.register_fake('sdacc_mild_projection_sm80::projection')
        def _projection_fake(x, packed, metadata, scales, bias, group, n, config, output_addend=None):
            return x.new_empty((*x.shape[:-1], n))
        @torch.library.register_fake('sdacc_mild_projection_sm80::projection_scalar')
        def _scalar_fake(x, packed, metadata, scales, bias, group, n, config, output_addend=None):
            return x.new_empty((*x.shape[:-1], n))

    @torch.library.register_fake('sdacc_mild_projection_sm80::projection_packed16')
    def _packed16_fake(x, packed, metadata, scales, bias, group, n, config, output_addend=None):
        return x.new_empty((*x.shape[:-1], n))
    @torch.library.register_fake('sdacc_prepared_projection_sm80::run')
    def _graph_fake(x, handle, n):
        return x.new_empty((*x.shape[:-1], n))

    _loaded = receipt["library_sha"]
    return receipt
