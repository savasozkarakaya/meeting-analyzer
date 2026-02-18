import torch
import logging
import os
import math
from speechbrain.inference.speaker import SpeakerRecognition
import torch.nn.functional as F
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)
EMBEDDING_MODEL_VERSION = "speechbrain/spkrec-ecapa-voxceleb"

class Embedder:
    def __init__(self, device="cpu", savedir=None):
        if savedir is None:
            # Use a default cache dir or let speechbrain decide (usually ~/.cache/speechbrain)
            # But we want to be offline friendly, so maybe we should specify a local dir if needed.
            # For now, let's use the default or env var.
            pass
            
        self.device = device
        logger.info(f"Loading SpeakerRecognition model on {device}...")
        # We use run_opts to specify device
        run_opts = {"device": device}
        
        # Workaround for Windows Symlink (WinError 1314)
        # We download the model to a local folder in the project directory, forcing copies instead of symlinks.
        try:
            logger.info("Downloading/Locating model via huggingface_hub (local_dir mode)...")
            
            # Create a local models directory
            local_model_dir = os.path.join(os.getcwd(), "models", "speechbrain_ecapa")
            os.makedirs(local_model_dir, exist_ok=True)
            
            source_path = snapshot_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb",
                local_dir=local_model_dir,
                local_dir_use_symlinks=False  # CRITICAL: This prevents WinError 1314
            )
            logger.info(f"Model found/downloaded at: {source_path}")
            
            self.verification = SpeakerRecognition.from_hparams(
                source=source_path, 
                savedir=os.path.join(local_model_dir, "savedir"),
                run_opts=run_opts
            )
        except Exception as e:
            logger.error(f"Failed to load model with local_dir workaround: {e}")
            # Fallback (might fail if symlinks are strictly required by default cache)
            self.verification = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=os.path.join(os.environ.get("HF_HOME", "."), "speechbrain_ecapa"),
                run_opts=run_opts
            )

    def get_embedding(self, wav_tensor):
        """
        Computes embedding for a waveform tensor.
        wav_tensor: (1, T) or (T,)
        """
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        # Ensure it's on the right device
        wav_tensor = wav_tensor.to(self.device)
        
        # SpeechBrain expects (batch, time)
        embedding = self.verification.encode_batch(wav_tensor)
        # embedding is (batch, 1, emb_dim) -> (batch, emb_dim)
        return embedding.squeeze(1)

    def compute_similarity(self, emb1, emb2):
        """
        Computes cosine similarity between two embeddings.
        """
        # Be tolerant to shapes: (1, D) vs (D,)
        if hasattr(emb1, "dim") and emb1.dim() == 2 and emb1.shape[0] == 1:
            emb1 = emb1.squeeze(0)
        if hasattr(emb2, "dim") and emb2.dim() == 2 and emb2.shape[0] == 1:
            emb2 = emb2.squeeze(0)

        # Ensure 1-D
        emb1 = emb1.view(-1)
        emb2 = emb2.view(-1)

        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)

        score = (emb1 * emb2).sum()
        return float(score.item())

def extract_reference_embedding(embedder, audio_path, max_duration=40.0):
    """
    Loads reference audio and extracts embedding.
    Simple strategy: load, crop to max_duration, embed.
    """
    import torchaudio
    
    # We use torchaudio here to load directly to tensor
    # But we might need to resample if not 16k. 
    # SpeechBrain ECAPA expects 16k.
    
    # Let's use our audio.py to convert/load if we want consistency, 
    # but audio.py returns numpy. 
    # Let's just use torchaudio and resample if needed.
    
    logger.info(f"Loading reference audio: {audio_path}")
    wav, sr = torchaudio.load(audio_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        
    # Mix to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # Crop if too long
    max_samples = int(max_duration * 16000)
    if wav.shape[1] > max_samples:
        wav = wav[:, :max_samples]
        
    emb = embedder.get_embedding(wav)
    return emb

def extract_reference_embedding_with_info(embedder, audio_path, max_duration=40.0):
    """
    Like extract_reference_embedding, but also returns basic quality metadata.
    """
    import torchaudio

    logger.info(f"Loading reference audio: {audio_path}")
    wav, sr = torchaudio.load(audio_path)

    # Resample to 16k for SpeechBrain ECAPA
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000

    # Mix to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Crop if too long
    max_samples = int(max_duration * sr)
    if wav.shape[1] > max_samples:
        wav = wav[:, :max_samples]

    # Quality checks (best-effort, lightweight)
    flags = []
    try:
        duration_sec = float(wav.shape[1]) / float(sr)
    except Exception:
        duration_sec = None

    try:
        peak = float(wav.abs().max().item())
        rms = float(wav.pow(2).mean().sqrt().item())
    except Exception:
        peak, rms = None, None

    try:
        snr_db = _estimate_snr_db(wav)
    except Exception:
        snr_db = None

    if duration_sec is not None and duration_sec < 1.0:
        flags.append("too_short")
    if rms is not None and rms < 0.005:
        flags.append("low_energy")
    if peak is not None and peak >= 0.99:
        flags.append("clipped")
    if snr_db is not None and snr_db < 8.0:
        flags.append("low_snr")

    emb = embedder.get_embedding(wav)

    info = {
        "duration_sec": duration_sec,
        "rms": rms,
        "peak": peak,
        "snr_db": snr_db,
        "flags": flags,
        "sr": sr,
        "samples": int(wav.shape[1]),
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
    }
    info["quality_score"] = compute_reference_quality_score(info)
    return emb, info


def _estimate_snr_db(wav):
    """
    Lightweight SNR proxy from waveform statistics.
    Uses lower-energy percentile as a noise-floor estimate.
    """
    flat = wav.abs().view(-1)
    if flat.numel() == 0:
        return None
    signal_rms = float(wav.pow(2).mean().sqrt().item())
    noise_floor = float(torch.quantile(flat, 0.2).item())
    signal_rms = max(signal_rms, 1e-8)
    noise_floor = max(noise_floor, 1e-8)
    return float(20.0 * math.log10(signal_rms / noise_floor))


def compute_reference_quality_score(info):
    """
    Produces a [0,1] quality score using duration/energy/SNR and flags.
    """
    score = 1.0

    duration_sec = info.get("duration_sec")
    rms = info.get("rms")
    snr_db = info.get("snr_db")
    flags = set(info.get("flags", []))

    if duration_sec is not None:
        if duration_sec < 1.0:
            score *= 0.35
        elif duration_sec < 2.0:
            score *= 0.7

    if rms is not None:
        if rms < 0.003:
            score *= 0.4
        elif rms < 0.008:
            score *= 0.8

    if snr_db is not None:
        if snr_db < 5.0:
            score *= 0.5
        elif snr_db < 10.0:
            score *= 0.8

    if "clipped" in flags:
        score *= 0.8
    if "low_energy" in flags:
        score *= 0.75
    if "low_snr" in flags:
        score *= 0.8

    return float(max(0.05, min(1.0, score)))


def combine_reference_embeddings(embeddings, infos):
    """
    Weighted aggregation for multiple references of one identity.
    Returns:
      - combined_embedding: (D,)
      - combine_info: metadata for explainability
    """
    if not embeddings:
        raise ValueError("No reference embeddings to combine.")

    clean_embs = []
    weights = []
    used_infos = []

    for idx, emb in enumerate(embeddings):
        if emb is None:
            continue
        v = emb.squeeze(0) if hasattr(emb, "dim") and emb.dim() > 1 else emb
        v = v.view(-1)
        v = F.normalize(v, p=2, dim=0)

        info = infos[idx] if idx < len(infos) else {}
        q = float(info.get("quality_score", compute_reference_quality_score(info)))
        q = max(0.05, min(1.0, q))

        clean_embs.append(v)
        weights.append(q)
        used_infos.append(info)

    if not clean_embs:
        raise ValueError("No valid reference embeddings after filtering.")

    stacked = torch.stack(clean_embs, dim=0)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).unsqueeze(1)
    weighted = stacked * weight_tensor
    combined = weighted.sum(dim=0) / weight_tensor.sum().clamp_min(1e-8)
    combined = F.normalize(combined, p=2, dim=0)

    combine_info = {
        "num_references": len(clean_embs),
        "method": "quality_weighted_mean",
        "weights": [float(w) for w in weights],
        "quality_scores": [float(i.get("quality_score", 0.0)) for i in used_infos],
        "reference_flags": [list(i.get("flags", [])) for i in used_infos],
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
    }
    return combined, combine_info
