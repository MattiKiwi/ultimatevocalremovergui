from __future__ import annotations

import argparse
import logging
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import urllib.request
import math

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from lib_v5.tfc_tdf_v3 import STFT

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = "UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_DATA_PATH = REPO_ROOT / "models" / "MDX_Net_Models" / "model_data" / "model_data.json"
MANUAL_DOWNLOAD_PATH = REPO_ROOT / "gui_data" / "model_manual_download.json"
MANUAL_BASE_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"

_STEM_PAIRS = {
    "Vocals": "Instrumental",
    "Instrumental": "Vocals",
    "Other": "No Other",
    "Bass": "No Bass",
    "Drums": "No Drums",
    "Guitar": "No Guitar",
    "Piano": "No Piano",
    "Synth": "No Synth",
    "Strings": "No Strings",
    "Woodwinds": "No Woodwinds",
    "Brass": "No Brass",
    "Wind": "No Wind",
    "lead_only": "backing_only",
    "backing_only": "lead_only",
}


LOG = logging.getLogger("uvr_cli")


def _stem_pair(name: str) -> str:
    """Return the complementary stem name."""
    if name in _STEM_PAIRS:
        return _STEM_PAIRS[name]
    for key, value in _STEM_PAIRS.items():
        if name == value:
            return key
    if name.startswith("No "):
        return name[3:]
    return f"No {name}"


@dataclass(frozen=True)
class Separations:
    """Holds the file paths for the two separated stems."""

    primary: Path
    secondary: Path


def _hash_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_mdx_metadata(model_path: Path) -> dict:
    if not MODEL_DATA_PATH.is_file():
        raise FileNotFoundError(f"metadata file not found: {MODEL_DATA_PATH}")
    data = json.loads(MODEL_DATA_PATH.read_text())
    model_hash = _hash_file(model_path)
    if model_hash not in data:
        inferred = _infer_mdx_metadata(model_path)
        LOG.warning(
            "Model hash %s not recognised; using inferred metadata (dim_f=%s, n_fft=%s).",
            model_hash,
            inferred["mdx_dim_f_set"],
            inferred["mdx_n_fft_scale_set"],
        )
        return inferred
    entry = data[model_hash]
    required = ("compensate", "mdx_dim_f_set", "mdx_dim_t_set", "mdx_n_fft_scale_set", "primary_stem")
    missing = [key for key in required if key not in entry]
    if missing:
        raise KeyError(f"model metadata missing keys: {missing}")
    return entry


def _as_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure (channels, samples) stereo layout."""
    if audio.ndim == 1:
        return np.vstack([audio, audio])
    channels, _ = audio.shape
    if channels == 2:
        return audio
    if audio.shape[1] == 2:
        return audio.T
    mono = np.mean(audio, axis=0, keepdims=False)
    return np.vstack([mono, mono])


def _load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.T.astype(np.float32, copy=False)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = signal.resample_poly(audio, up, down, axis=-1)
        sr = target_sr
    return _as_stereo(audio), sr


def _default_providers() -> Sequence[str]:
    try:
        import onnxruntime as ort  # type: ignore

        available = ort.get_available_providers()
    except Exception:
        return []
    preferred: Iterable[str] = (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    return [provider for provider in preferred if provider in available]


class _OrtRunner:
    def __init__(self, session):
        self.session = session

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        output = self.session.run(None, {"input": spectrogram.cpu().numpy()})[0]
        return torch.from_numpy(output)


class _TorchRunner:
    def __init__(self, module: torch.nn.Module, device: torch.device):
        self.module = module.to(device).eval()
        self.device = device

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.module(spectrogram.to(self.device))
        return output.cpu()


def _create_model_runner(
    model_path: Path,
    *,
    providers: Sequence[str] | None,
    segment_size: int,
    dim_t: int,
) -> tuple[object, str]:
    # Prefer ONNX Runtime when segment size matches the exported model.
    if segment_size == dim_t:
        try:
            import onnxruntime as ort  # type: ignore

            session = ort.InferenceSession(str(model_path), providers=providers or _default_providers())
            return _OrtRunner(session), "onnxruntime"
        except Exception as exc:
            LOG.warning(
                "Failed to initialise ONNX Runtime (%s). Falling back to PyTorch execution.",
                exc,
            )

    from onnx import load
    from onnx2pytorch import ConvertModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = ConvertModel(load(model_path))
    return _TorchRunner(module, device), "torch"


def _run_mdx(
    mix: np.ndarray,
    model_runner,
    stft: STFT,
    segment_size: int,
    n_fft: int,
    overlap: float,
    compensate: float,
) -> np.ndarray:
    hop = 1024
    trim = n_fft // 2
    chunk_size = hop * (segment_size - 1)
    gen_size = chunk_size - 2 * trim
    pad = (gen_size + trim) - (mix.shape[-1] % gen_size)
    mixture = np.concatenate(
        (np.zeros((2, trim), dtype=np.float32), mix, np.zeros((2, pad), dtype=np.float32)),
        axis=1,
    )

    step = max(1, int((1.0 - overlap) * chunk_size))
    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros_like(result)

    for start in range(0, mixture.shape[-1], step):
        end = min(start + chunk_size, mixture.shape[-1])
        chunk_size_actual = end - start
        chunk = mixture[:, start:end]
        if chunk.shape[-1] < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[-1])), mode="constant")

        window = np.hanning(chunk_size_actual).astype(np.float32)
        window = np.tile(window[None, :], (2, 1))

        input_tensor = torch.from_numpy(chunk[None])
        spec = stft(input_tensor)
        if spec.shape[-1] > segment_size:
            spec = spec[..., :segment_size]
        prediction = model_runner(spec)
        audio = stft.inverse(prediction).cpu().numpy()
        audio = audio[..., :chunk_size_actual]

        result[..., start:end] += audio * window[None]
        divider[..., start:end] += window[None]

    np.maximum(divider, 1e-8, out=divider)
    separated = (result / divider)[:, :, trim:-trim]
    separated = separated[:, :, : mix.shape[-1]]
    return separated[0] * compensate


def _ensure_model_file(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    if not MANUAL_DOWNLOAD_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Manual download manifest missing: {MANUAL_DOWNLOAD_PATH}"
        )

    manual_data = json.loads(MANUAL_DOWNLOAD_PATH.read_text())
    mdx_files = manual_data.get("mdx_download_list", {})
    filename = model_path.name
    if filename not in mdx_files.values():
        raise FileNotFoundError(
            f"Model '{filename}' not available locally and not listed in manual downloads."
        )

    url = f"{MANUAL_BASE_URL}{filename}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.warning("Downloading %s from %s", filename, url)

    try:
        with urllib.request.urlopen(url) as response, open(model_path, "wb") as fh:
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        raise FileNotFoundError(f"Failed to download model from {url}: {exc}") from exc

    return model_path


def _infer_mdx_metadata(model_path: Path) -> dict:
    from onnx import load

    model = load(str(model_path))
    if not model.graph.input:
        raise KeyError("Unable to infer MDX metadata: model has no inputs.")
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) < 4:
        raise KeyError("Unable to infer MDX metadata: unexpected input rank.")

    def _dim_value(index: int, default: int) -> int:
        value = dims[index].dim_value
        return value if value else default

    dim_f = _dim_value(2, 3072)
    segment = _dim_value(3, 256)
    mdx_dim_t_set = int(round(math.log2(segment))) if segment else 8
    n_fft = dim_f * 2
    filename = model_path.name.lower()
    primary = "Instrumental" if "inst" in filename and "voc" not in filename else "Vocals"

    return {
        "compensate": 1.0,
        "mdx_dim_f_set": dim_f,
        "mdx_dim_t_set": mdx_dim_t_set,
        "mdx_n_fft_scale_set": n_fft,
        "primary_stem": primary,
    }


def separate(
    input_path: str | Path,
    model_path: str | Path | None = None,
    *,
    output_dir: str | Path = "UVR_separated",
    overlap: float = 0.25,
    sample_rate: int = 44100,
    providers: Sequence[str] | None = None,
) -> Separations:
    input_path = Path(input_path)
    if model_path is None:
        model_path = REPO_ROOT / "models" / "MDX_Net_Models" / DEFAULT_MODEL_NAME
    model_path = Path(model_path)
    model_path = _ensure_model_file(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if overlap <= 0 or overlap >= 1:
        raise ValueError("overlap must be between 0 and 1 (exclusive)")

    mix, sr = _load_audio(input_path, sample_rate)
    metadata = _load_mdx_metadata(model_path)
    stft = STFT(metadata["mdx_n_fft_scale_set"], 1024, metadata["mdx_dim_f_set"])
    segment_size = 2 ** metadata["mdx_dim_t_set"]
    runner, backend = _create_model_runner(
        model_path,
        providers=providers,
        segment_size=segment_size,
        dim_t=segment_size,
    )

    if backend != "onnxruntime":
        LOG.info("Using PyTorch backend for MDX inference (providers %s).", providers or [])

    vocals = _run_mdx(
        mix,
        runner,
        stft,
        segment_size=segment_size,
        n_fft=metadata["mdx_n_fft_scale_set"],
        overlap=overlap,
        compensate=metadata["compensate"],
    )

    accompaniment = mix - vocals
    primary_name = metadata["primary_stem"]
    secondary_name = _stem_pair(primary_name)
    stem = input_path.stem

    primary_path = output_dir / f"{stem}_({primary_name}).wav"
    secondary_path = output_dir / f"{stem}_({secondary_name}).wav"

    sf.write(str(primary_path), vocals.T, sr)
    sf.write(str(secondary_path), accompaniment.T, sr)

    return Separations(primary=primary_path, secondary=secondary_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UVR separation without the GUI.")
    parser.add_argument("input", help="Audio file to process.")
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "MDX_Net_Models" / DEFAULT_MODEL_NAME),
        help="Path to the MDX-Net ONNX model to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="UVR_separated",
        help="Directory to store separated stems.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap factor between 0 and 1; higher smooths transitions at the cost of speed.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Target sample rate.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Custom ONNX Runtime execution providers (defaults to CUDA->CPU fallbacks).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Separations:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return separate(
        args.input,
        args.model,
        output_dir=args.output_dir,
        overlap=args.overlap,
        sample_rate=args.sample_rate,
        providers=args.providers,
    )
