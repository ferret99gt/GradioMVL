import abc
import math
import os
import threading
import time

from abc import abstractmethod

import librosa
import numpy as np
import torch

STUDIO_MODELS_ROOT = "studio_models"
lock = threading.Lock()

# Base class for Pipeline
class StudioModelConversionPipeline(abc.ABC):
    def __init__(self, input_sample_rate: int):
        self.p_sampling_rate = 16000
        #self.pp_sampling_rate = 24000
        self.input_sample_rate = input_sample_rate # 22050 by default

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            
        self.mac_silicon_device = torch.backends.mps.is_available()

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
        time_start = time.time()
        # Audio comes from PyAudio at 22050Hz, which we resample to 16khz for the preprocessor. The inference returns 24khz, which we resample back to 22050Hz to return to PyAudio for output.
        with torch.no_grad():
            # Replace MetaVoice's soundfile write / librosa load for sample rate change with Librosa in-memory conversion.
            # Increase resampling quality from soxr_hq to soxr_vhq
            # Resampling on a AMD Ryzen 7 3700X tested at 1-2ms.
            #time_start_res = time.time()
            wav_src = librosa.resample(wav, orig_sr=self.input_sample_rate, target_sr=self.p_sampling_rate, res_type="soxr_vhq")
            #print(f"Input down sample took: {time.time()-time_start_res}")
                        
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
            # Increase resampling quality from soxr_hq to soxr_vhq
            # Resampling on a AMD Ryzen 7 3700X tested at 1-3ms.            
            #time_start_res = time.time()
            #out = librosa.resample(audio, orig_sr=self.pp_sampling_rate, target_sr=self.sampling_rate, res_type="soxr_vhq")
            #print(f"Output down sample took: {time.time()-time_start_res}")
            out = audio

        # Average response time on RTX 2070S observed between 60-80ms.
        print(f"Conversion took: {time.time()-time_start}")
        return out

    @abstractmethod
    def run(self, wav: np.ndarray):
        pass
        
# --------------------
#  Conversion pipeline
# --------------------
class ConversionPipeline(StudioModelConversionPipeline):
    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)

        self._sample_rate = sample_rate
        
        fade_duration_ms = 20  # 20ms
        self._fade_samples = int(fade_duration_ms / 1000 * self._sample_rate) # sample count

        self._linear_fade_in = np.linspace(0, 1, self._fade_samples, dtype=np.float32)
        self._linear_fade_out = np.linspace(1, 0, self._fade_samples, dtype=np.float32)
        
        self._constant_power_fade_in = []
        self._constant_power_fade_out = []
        
        for t in self._linear_fade_in:
            self._constant_power_fade_in.append(math.sqrt(t))

        for t in self._linear_fade_out:
            self._constant_power_fade_out.append(math.sqrt(t))

        self._constant_power_fade_in = np.array(self._constant_power_fade_in)
        self._constant_power_fade_out = np.array(self._constant_power_fade_out)
        
        self._old_samples = np.zeros(self._fade_samples, dtype=np.float32)
        
        self.crossfade = "linear"

    def set_crossfade(self, crossfade: str):
        self.crossfade = crossfade

    def run(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        if self.crossfade == "linear":
            return self.run_linear_crossfade(wav, HDW_FRAMES_PER_BUFFER)
        elif self.crossfade == "constant power":
            return self.run_constant_power_crossfade(wav, HDW_FRAMES_PER_BUFFER)
        else:
            return self.run_unaltered(wav, HDW_FRAMES_PER_BUFFER)

    # Linear cross-fade
    def run_linear_crossfade(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        out = self.infer(wav)

        # cross-fade = fade_in + fade_out
        out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -HDW_FRAMES_PER_BUFFER] = (
            out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -HDW_FRAMES_PER_BUFFER] * self._linear_fade_in
        ) + (self._old_samples * self._linear_fade_out)

        # save old sample for next time.
        self._old_samples = out[-self._fade_samples :]

        # send
        return out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -self._fade_samples]
        
    # constant power cross-fade
    def run_constant_power_crossfade(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        out = self.infer(wav)

        # cross-fade = fade_in + fade_out
        out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -HDW_FRAMES_PER_BUFFER] = (
            out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -HDW_FRAMES_PER_BUFFER] * self._constant_power_fade_in
        ) + (self._old_samples * self._constant_power_fade_out)

        # save old sample for next time.
        self._old_samples = out[-self._fade_samples :]

        # send
        return out[-(HDW_FRAMES_PER_BUFFER + self._fade_samples) : -self._fade_samples]
        
    # Direct return. Observed to have a little bit of "poppy" and crackling.
    def run_unaltered(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        out = self.infer(wav)
        
        return out[-HDW_FRAMES_PER_BUFFER:]