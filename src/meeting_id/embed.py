import torch
import logging
import os
from speechbrain.inference.speaker import SpeakerRecognition
import torch.nn.functional as F
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

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
        # Cosine similarity
        score = F.cosine_similarity(emb1, emb2)
        return score.item()

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
