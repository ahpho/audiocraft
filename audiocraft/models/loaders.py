# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path
import typing as tp
import os
import tempfile

from huggingface_hub import hf_hub_download, HfApi
from omegaconf import OmegaConf, DictConfig
import torch

import audiocraft

from . import builders
from .encodec import CompressionModel


def get_audiocraft_cache_dir() -> tp.Optional[str]:
    return os.environ.get('AUDIOCRAFT_CACHE_DIR', None)


def _hf_hub_download_kwargs() -> dict:
    """Build kwargs for hf_hub_download, including HF_ENDPOINT when set (e.g. for mirror)."""
    kwargs = {
        "library_name": "audiocraft",
        "library_version": audiocraft.__version__,
    }
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        kwargs["endpoint"] = endpoint
    return kwargs


def _download_from_mirror(
    repo_id: str,
    filename: str,
    endpoint: str,
    cache_dir: tp.Optional[str] = None,
) -> str:
    """Download a single file from HF mirror via direct URL; returns path to local file.
    If cache_dir is set (e.g. AUDIOCRAFT_CACHE_DIR), file is saved under cache_dir/mirror/<repo_id>/<filename>
    and reused on next run; otherwise a temp file is used (caller should delete after use).
    """
    import gzip
    import urllib.request

    base = endpoint.rstrip("/")

    def _fetch(url: str) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "audiocraft/1.0 (https://github.com/facebookresearch/audiocraft)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            ce = resp.headers.get("Content-Encoding", "").strip().lower()
            if ce == "gzip" or (len(data) >= 2 and data[:2] == b"\x1f\x8b"):
                data = gzip.decompress(data)
        return data

    def _valid_json_content(data: bytes) -> bool:
        if not filename.lower().endswith(".json"):
            return True
        try:
            text = data.decode("utf-8").strip()
            return bool(text) and (text[0] in "{[")
        except Exception:
            return False

    url = f"{base}/{repo_id}/resolve/main/{filename}"
    try:
        data = _fetch(url)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    if not _valid_json_content(data):
        if data.lstrip()[:1] == b"<":
            alt_url = f"{base}/{repo_id}/raw/main/{filename}"
            try:
                data = _fetch(alt_url)
            except Exception:
                pass
        if not _valid_json_content(data):
            raise RuntimeError(
                f"Mirror returned non-JSON for {url} (e.g. HTML error page). "
                f"Check HF_ENDPOINT={endpoint} or try without mirror."
            )

    if cache_dir:
        safe_repo = repo_id.replace("/", "--")
        cache_path = Path(cache_dir) / "mirror" / safe_repo / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return str(cache_path)
        cache_path.write_bytes(data)
        return str(cache_path)

    fd, path = tempfile.mkstemp(suffix=Path(filename).suffix or ".bin")
    try:
        os.write(fd, data)
        os.close(fd)
        return path
    except Exception:
        os.close(fd)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise


# Fallback file list when HfApi.list_repo_files fails on mirror (e.g. API not available)
_T5_REPO_FILES_FALLBACK: tp.Dict[str, tp.List[str]] = {
    "t5-small": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "t5-base": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "t5-large": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "t5-3b": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "t5-11b": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "google/flan-t5-small": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "google/flan-t5-base": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "google/flan-t5-large": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "google/flan-t5-xl": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
    "google/flan-t5-xxl": [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "pytorch_model.bin",
    ],
}

# Encodec repo file list (when HfApi.list_repo_files fails on mirror)
_ENCODEC_REPO_FILES_FALLBACK: tp.Dict[str, tp.List[str]] = {
    "facebook/encodec_24khz": [
        "config.json", "preprocessor_config.json", "pytorch_model.bin",
    ],
    "facebook/encodec_32khz": [
        "config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors",
    ],
}

# ModelScope (魔搭) model_id mapping: HF repo_id -> modelscope.cn model_id
# ModelScope is not an HF-compatible mirror; use its SDK when HF_ENDPOINT points to modelscope.cn
_MODELSCOPE_REPO_MAP: tp.Dict[str, str] = {
    "t5-small": "AI-ModelScope/t5-small",
    "t5-base": "AI-ModelScope/t5-base",
    "t5-large": "AI-ModelScope/t5-large",
    "t5-3b": "AI-ModelScope/t5-3b",
    "t5-11b": "AI-ModelScope/t5-11b",
    "google/flan-t5-small": "AI-ModelScope/flan-t5-small",
    "google/flan-t5-base": "AI-ModelScope/flan-t5-base",
    "google/flan-t5-large": "AI-ModelScope/flan-t5-large",
    "google/flan-t5-xl": "AI-ModelScope/flan-t5-xl",
    "google/flan-t5-xxl": "AI-ModelScope/flan-t5-xxl",
}


def ensure_hf_model_cached(
    repo_id: str,
    endpoint: str,
    cache_dir: tp.Optional[str] = None,
) -> str:
    """Download a full HuggingFace model repo from mirror to a local dir; return path to that dir.
    Used when transformers.from_pretrained(repo_id) fails (e.g. mirror metadata issue).
    If HF_ENDPOINT is modelscope.cn (魔搭), uses ModelScope SDK when available.
    """
    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "audiocraft")
    cache_dir = str(cache_dir)
    safe_repo = repo_id.replace("/", "--")
    model_dir = Path(cache_dir) / "mirror" / safe_repo
    if model_dir.exists() and (model_dir / "config.json").exists():
        return str(model_dir)

    # ModelScope (魔搭): different API and URL; use its SDK instead of HF-style direct URLs
    if "modelscope.cn" in endpoint:
        ms_model_id = _MODELSCOPE_REPO_MAP.get(repo_id)
        if ms_model_id is None:
            raise RuntimeError(
                f"ModelScope endpoint detected but no mapping for repo_id={repo_id}. "
                f"For Hugging Face models use HF_ENDPOINT=https://hf-mirror.com instead."
            )
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            raise RuntimeError(
                "HF_ENDPOINT points to ModelScope (modelscope.cn). "
                "Install ModelScope SDK: pip install modelscope"
            ) from None
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(ms_model_id, local_dir=str(model_dir), revision="master")
        return str(model_dir)

    try:
        api = HfApi(endpoint=endpoint)
        files = api.list_repo_files(repo_id, revision="main")
    except Exception:
        files = _T5_REPO_FILES_FALLBACK.get(repo_id) or _ENCODEC_REPO_FILES_FALLBACK.get(repo_id)
        if files is None:
            raise RuntimeError(
                f"Cannot list files for {repo_id} from mirror and no fallback file list. "
                "For Hugging Face models (e.g. encodec) use HF_ENDPOINT=https://hf-mirror.com ."
            ) from None

    for f in files:
        _download_from_mirror(repo_id, f, endpoint, cache_dir=cache_dir)

    return str(model_dir)


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location=device)

    if os.path.isdir(file_or_url_or_id):
        file = f"{file_or_url_or_id}/{filename}"
        return torch.load(file, map_location=device)

    elif file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    else:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"
        repo_id = file_or_url_or_id
        # Use a default cache root when unset so mirror fallback always persists
        effective_cache_dir = cache_dir
        if effective_cache_dir is None:
            effective_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "audiocraft")
        # Prefer existing mirror cache to avoid re-download and failed hf_hub_download
        safe_repo = repo_id.replace("/", "--")
        mirror_path = Path(effective_cache_dir) / "mirror" / safe_repo / filename
        if mirror_path.exists():
            return torch.load(str(mirror_path), map_location=device)
        try:
            file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                **_hf_hub_download_kwargs(),
            )
            return torch.load(file, map_location=device)
        except Exception as e:
            endpoint = os.environ.get("HF_ENDPOINT")
            if not endpoint:
                raise
            err_msg = str(e).lower()
            if "huggingface.co" not in err_msg and "cannot find" not in err_msg and "localentrynotfound" not in err_msg and "filemetadata" not in err_msg:
                raise
            # Fallback: mirror download; always use effective_cache_dir so file is persisted
            path = _download_from_mirror(repo_id, filename, endpoint, cache_dir=effective_cache_dir)
            return torch.load(path, map_location=device)


def load_compression_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)


def load_compression_model(
    file_or_url_or_id: tp.Union[Path, str],
    device="cpu",
    cache_dir: tp.Optional[str] = None,
):
    pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    if 'pretrained' in pkg:
        return CompressionModel.get_pretrained(pkg['pretrained'], device=device)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg["best_state"])
    model.eval()
    return model


def load_lm_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)


def _delete_param(cfg: DictConfig, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)


def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_lm_model_magnet(file_or_url_or_id: tp.Union[Path, str], compression_model_frame_rate: int,
                         device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    cfg.transformer_lm.compression_model_framerate = compression_model_frame_rate
    cfg.transformer_lm.segment_duration = cfg.dataset.segment_duration
    cfg.transformer_lm.span_len = cfg.masking.span_len

    # MAGNeT models v1 support only xformers backend.
    from audiocraft.modules.transformer import set_efficient_attention_backend

    if cfg.transformer_lm.memory_efficient:
        set_efficient_attention_backend("xformers")

    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_jasco_model(file_or_url_or_id: tp.Union[Path, str],
                     compression_model: CompressionModel,
                     device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_jasco_model(cfg, compression_model)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_mbd_ckpt(file_or_url_or_id: tp.Union[Path, str],
                  filename: tp.Optional[str] = None,
                  cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename=filename, cache_dir=cache_dir)


def load_diffusion_models(file_or_url_or_id: tp.Union[Path, str],
                          device='cpu',
                          filename: tp.Optional[str] = None,
                          cache_dir: tp.Optional[str] = None):
    pkg = load_mbd_ckpt(file_or_url_or_id, filename=filename, cache_dir=cache_dir)
    models = []
    processors = []
    cfgs = []
    sample_rate = pkg['sample_rate']
    for i in range(pkg['n_bands']):
        cfg = pkg[i]['cfg']
        model = builders.get_diffusion_model(cfg)
        model_dict = pkg[i]['model_state']
        model.load_state_dict(model_dict)
        model.to(device)
        processor = builders.get_processor(cfg=cfg.processor, sample_rate=sample_rate)
        processor_dict = pkg[i]['processor_state']
        processor.load_state_dict(processor_dict)
        processor.to(device)
        models.append(model)
        processors.append(processor)
        cfgs.append(cfg)
    return models, processors, cfgs


def load_audioseal_models(
    file_or_url_or_id: tp.Union[Path, str],
    device="cpu",
    filename: tp.Optional[str] = None,
    cache_dir: tp.Optional[str] = None,
):

    detector_ckpt = _get_state_dict(
        file_or_url_or_id,
        filename=f"detector_{filename}.pth",
        device=device,
        cache_dir=cache_dir,
    )
    assert (
        "model" in detector_ckpt
    ), f"No model state dict found in {file_or_url_or_id}/detector_{filename}.pth"
    detector_state = detector_ckpt["model"]

    generator_ckpt = _get_state_dict(
        file_or_url_or_id,
        filename=f"generator_{filename}.pth",
        device=device,
        cache_dir=cache_dir,
    )
    assert (
        "model" in generator_ckpt
    ), f"No model state dict found in {file_or_url_or_id}/generator_{filename}.pth"
    generator_state = generator_ckpt["model"]

    def load_model_config():
        if Path(file_or_url_or_id).joinpath(f"{filename}.yaml").is_file():
            return OmegaConf.load(Path(file_or_url_or_id).joinpath(f"{filename}.yaml"))
        elif file_or_url_or_id.startswith("https://"):
            import requests  # type: ignore

            resp = requests.get(f"{file_or_url_or_id}/{filename}.yaml")
            return OmegaConf.create(resp.text)
        else:
            try:
                file = hf_hub_download(
                    repo_id=file_or_url_or_id,
                    filename=f"{filename}.yaml",
                    cache_dir=cache_dir,
                    **_hf_hub_download_kwargs(),
                )
                return OmegaConf.load(file)
            except Exception as e:
                endpoint = os.environ.get("HF_ENDPOINT")
                if endpoint:
                    err_msg = str(e).lower()
                    if "huggingface.co" in err_msg or "cannot find" in err_msg or "localentrynotfound" in err_msg or "filemetadata" in err_msg:
                        path = _download_from_mirror(
                            file_or_url_or_id, f"{filename}.yaml", endpoint, cache_dir=cache_dir
                        )
                        try:
                            return OmegaConf.load(path)
                        finally:
                            if cache_dir is None:
                                try:
                                    os.remove(path)
                                except OSError:
                                    pass
                raise

    try:
        cfg = load_model_config()
    except Exception as exc:  # noqa
        cfg_fp = (
            Path(__file__)
            .parents[2]
            .joinpath("config", "model", "watermark", "default.yaml")
        )
        cfg = OmegaConf.load(cfg_fp)

    OmegaConf.resolve(cfg)
    model = builders.get_watermark_model(cfg)

    model.generator.load_state_dict(generator_state)
    model.detector.load_state_dict(detector_state)
    return model.to(device)
