import abc
import os
import threading

from abc import abstractmethod

import librosa
import numpy as np
import torch
from modules.torch_utils import get_device, is_mps_available

STUDIO_MODELS_ROOT = "studio_models"
lock = threading.Lock()

# Base class for Pipeline
class StudioModelConversionPipeline(abc.ABC):
    def __init__(self):
        self.p_sampling_rate = 16000
        self.pp_sampling_rate = 24000
        self.sampling_rate = 22050

        self.device = get_device()
        self.mac_silicon_device = is_mps_available()

        # Load the model.
        self.model = torch.jit.load(os.path.join(STUDIO_MODELS_ROOT, "model.pt")).to(self.device)

        # Load the preprocessor.
        if self.mac_silicon_device:
            import coremltools as ct

            self.pmodel = ct.models.MLModel(os.path.join(STUDIO_MODELS_ROOT, "model.mlpackage"))
        else:
            self.pmodel = torch.jit.load(os.path.join(STUDIO_MODELS_ROOT, "b_model.pt")).to(self.device)
            
        self.targetSet = False
        
    def isTargetSet(self):
        return self.targetSet

    def set_target(self, speaker_id: str):
        path_studio = os.path.join(STUDIO_MODELS_ROOT, f"targets/{speaker_id}.npy")
        path = path_studio

        if not os.path.exists(path):
            raise FileNotFoundError(f"Target speaker {speaker_id} not found in {path_studio}.")

        with lock:
            self.target = np.load(path)
            self.target = torch.from_numpy(self.target).unsqueeze(0).to(self.device)
        
        self.targetSet = True

    def infer(self, wav: np.ndarray) -> np.ndarray:
        # Audio comes from PyAudio at 22050Hz, which we resample to 16khz for the preprocessor. The inference returns 24khz, which we resample back to 22050Hz to return to PyAudio for output.
        with torch.no_grad():
            # Replace MetaVoice's soundfile write / librosa load for sample rate change with Librosa in-memory conversion.
            wav_src = librosa.resample(wav, orig_sr=self.sampling_rate, target_sr=self.p_sampling_rate, res_type="soxr_hq")
                        
            if not self.mac_silicon_device:
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
                c = self.pmodel(wav_src.squeeze(1))
            else:
                c = self.pmodel.predict({"input_values": wav_src[np.newaxis, :]})["var_3641"]
                c = torch.from_numpy(c).to(self.device)

            with lock:
                audio = self.model(c, self.target)
            audio = audio[0][0].data.cpu().float().numpy()
            
            # Replace MetaVoice's soundfile write / librosa load for sample rate change with Librosa in-memory conversion.
            out = librosa.resample(audio, orig_sr=self.pp_sampling_rate, target_sr=self.sampling_rate, res_type="soxr_hq")

        return out

    @abstractmethod
    def run(self, wav: np.ndarray):
        pass
        
# --------------------
#  Conversion pipeline
# --------------------
class ConversionPipeline(StudioModelConversionPipeline):
    def __init__(self, sample_rate: int):
        super().__init__()

        self._sample_rate = sample_rate
        
        fade_duration_ms = 20
        self._fade_samples = int(fade_duration_ms / 1000 * self._sample_rate)  # 20ms

        self._linear_fade_in = np.linspace(0, 1, self._fade_samples, dtype=np.float32)
        self._linear_fade_out = np.linspace(1, 0, self._fade_samples, dtype=np.float32)
        self._old_samples = np.zeros(self._fade_samples, dtype=np.float32)

    def run(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        return self.run_cross_fade(wav, HDW_FRAMES_PER_BUFFER)

    # Linear cross-fade
    def run_cross_fade(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        out = self.infer(wav)

        # cross-fade = fade_in + fade_out
        out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -HDW_FRAMES_PER_BUFFER] = (
            out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -HDW_FRAMES_PER_BUFFER] * self._linear_fade_in
        ) + (self._old_samples * self._linear_fade_out)
        # save
        self._old_samples = out[-self._fade_samples :]
        # send
        out = out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -self._fade_samples]
        return out        