import abc
import os
import threading
import time

from abc import abstractmethod

import librosa
import numpy as np
import torch

STUDIO_MODELS_ROOT = "studio_models"
model_lock = threading.Lock()


class StudioModelConversionPipeline(abc.ABC):
    def __init__(self, input_sample_rate: int, output_sample_rate: int):
        self.p_sampling_rate = 16000
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        if not torch.cuda.is_available():
            raise RuntimeError("The bundled MetaVoice models require a CUDA-capable GPU.")

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0")
        self.pmodel = torch.jit.load(
            os.path.join(STUDIO_MODELS_ROOT, "b_model.pt"), map_location=self.device
        ).eval()

        if self.output_sample_rate == 48000:
            print("Loading 48K model...")
            model_name = "model_48k.pt"
        elif self.output_sample_rate == 24000:
            print("Loading 24K model...")
            model_name = "model.pt"
        else:
            raise ValueError("Output sample rate must be 24000 or 48000 Hz.")

        self.model = torch.jit.load(
            os.path.join(STUDIO_MODELS_ROOT, model_name), map_location=self.device
        ).eval()
        self.target_set = False
        self.noise_scale = 0.0

    def isTargetSet(self):
        return self.target_set

    def set_target(self, speaker_id: str):
        target_path = os.path.join(STUDIO_MODELS_ROOT, "targets", f"{speaker_id}.npy")
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Target speaker {speaker_id} not found in {target_path}."
            )

        target = np.load(target_path)
        if target.shape != (256,) or not np.isfinite(target).all():
            raise ValueError(f"Target speaker {speaker_id} is not a valid 256-value embedding.")

        with model_lock:
            self.target = torch.from_numpy(target.astype(np.float32)).unsqueeze(0).to(self.device)
            self.target_set = True

    def set_noise_scale(self, noise_scale: float):
        with model_lock:
            self.noise_scale = min(max(float(noise_scale), 0.0), 1.0)

    def _run_voice_model(self, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        content_mask = torch.ones(
            content.shape[0], 1, content.shape[2], dtype=content.dtype, device=content.device
        )
        encoded = self.model.enc_p.pre(content) * content_mask
        encoded = self.model.enc_p.enc(encoded, content_mask)
        statistics = self.model.enc_p.proj(encoded) * content_mask
        mean, log_scale = torch.split(statistics, statistics.shape[1] // 2, dim=1)

        if self.noise_scale == 0.0:
            sampled = mean
        else:
            sampled = mean + torch.randn_like(mean) * torch.exp(log_scale) * self.noise_scale

        speaker = target.unsqueeze(-1)
        latent = self.model.flow(sampled * content_mask, content_mask, speaker) * content_mask
        return self.model.dec(latent, speaker)

    def infer(self, wav: np.ndarray, input_sample_rate: int | None = None) -> np.ndarray:
        time_start = time.perf_counter()
        source_sample_rate = input_sample_rate or self.input_sample_rate
        with torch.inference_mode():
            wav_src = librosa.resample(
                wav,
                orig_sr=source_sample_rate,
                target_sr=self.p_sampling_rate,
                res_type="soxr_vhq",
            )
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
            content = self.pmodel(wav_src.squeeze(1))

            with model_lock:
                audio = self._run_voice_model(content, self.target)

            out = audio[0][0].detach().cpu().float().numpy()

        print(f"Conversion took: {time.perf_counter() - time_start}")
        return out

    @abstractmethod
    def run(self, wav: np.ndarray, output_frames: int):
        pass


class ConversionPipeline(StudioModelConversionPipeline):
    def __init__(
        self,
        input_sample_rate: int,
        output_sample_rate: int,
        fade_duration_ms: int = 20,
    ):
        super().__init__(input_sample_rate, output_sample_rate)
        self._overlap_lock = threading.Lock()
        self.crossfade = "linear"
        self.set_fade_duration(fade_duration_ms)

    def set_crossfade(self, crossfade: str):
        with self._overlap_lock:
            if crossfade != self.crossfade:
                self.crossfade = crossfade
                self._old_samples = np.zeros(self._fade_samples, dtype=np.float32)

    def set_fade_duration(self, fade_duration_ms: int):
        fade_duration_ms = max(int(fade_duration_ms), 0)
        fade_samples = round(fade_duration_ms / 1000 * self.output_sample_rate)

        if fade_samples:
            linear_fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            linear_fade_out = 1.0 - linear_fade_in
            constant_power_fade_in = np.sqrt(linear_fade_in)
            constant_power_fade_out = np.sqrt(linear_fade_out)
            hann_fade_in = 0.5 - 0.5 * np.cos(np.pi * linear_fade_in)
            hann_fade_out = 1.0 - hann_fade_in
        else:
            linear_fade_in = np.empty(0, dtype=np.float32)
            linear_fade_out = np.empty(0, dtype=np.float32)
            constant_power_fade_in = np.empty(0, dtype=np.float32)
            constant_power_fade_out = np.empty(0, dtype=np.float32)
            hann_fade_in = np.empty(0, dtype=np.float32)
            hann_fade_out = np.empty(0, dtype=np.float32)

        with self._overlap_lock:
            self.fade_duration_ms = fade_duration_ms
            self._fade_samples = fade_samples
            self._linear_fade_in = linear_fade_in
            self._linear_fade_out = linear_fade_out
            self._constant_power_fade_in = constant_power_fade_in
            self._constant_power_fade_out = constant_power_fade_out
            self._hann_fade_in = hann_fade_in.astype(np.float32)
            self._hann_fade_out = hann_fade_out.astype(np.float32)
            self._old_samples = np.zeros(fade_samples, dtype=np.float32)

    def reset_overlap(self):
        with self._overlap_lock:
            self._old_samples = np.zeros(self._fade_samples, dtype=np.float32)

    def run(self, wav: np.ndarray, output_frames: int):
        if output_frames == 0:
            return self.run_unaltered(wav, output_frames)
        return self._run_crossfade(wav, output_frames)

    def runCustomSampleRate(self, wav: np.ndarray, inputSamplingRate: int):
        return self.infer(wav, inputSamplingRate)

    def _run_crossfade(
        self,
        wav: np.ndarray,
        output_frames: int,
    ) -> np.ndarray:
        out = self.infer(wav)
        if output_frames <= 0:
            return out

        with self._overlap_lock:
            if self.crossfade == "none":
                return out[-output_frames:].copy()

            effective_fade = min(
                self._fade_samples,
                output_frames,
                max(0, len(out) - output_frames),
            )
            if effective_fade == 0:
                self._save_overlap(out)
                return out[-output_frames:].copy()

            fade_windows = {
                "linear": (self._linear_fade_in, self._linear_fade_out),
                "constant power": (
                    self._constant_power_fade_in,
                    self._constant_power_fade_out,
                ),
                "hann": (self._hann_fade_in, self._hann_fade_out),
            }
            fade_in, fade_out = fade_windows.get(
                self.crossfade, (self._linear_fade_in, self._linear_fade_out)
            )
            if effective_fade != self._fade_samples:
                progress = np.linspace(0.0, 1.0, effective_fade, dtype=np.float32)
                if self.crossfade == "constant power":
                    fade_in = np.sqrt(progress)
                    fade_out = np.sqrt(1.0 - progress)
                elif self.crossfade == "hann":
                    fade_in = 0.5 - 0.5 * np.cos(np.pi * progress)
                    fade_out = 1.0 - fade_in
                else:
                    fade_in = progress
                    fade_out = 1.0 - progress

            current_overlap = out[-(output_frames + effective_fade) : -output_frames]
            previous_overlap = self._old_samples[-effective_fade:]
            result = out[-(output_frames + effective_fade) : -effective_fade].copy()
            result[:effective_fade] = (
                current_overlap * fade_in
                + previous_overlap * fade_out
            )
            self._save_overlap(out)
            return result

    def _save_overlap(self, out: np.ndarray):
        if self._fade_samples == 0:
            self._old_samples = np.empty(0, dtype=np.float32)
        elif len(out) >= self._fade_samples:
            self._old_samples = out[-self._fade_samples :].copy()
        else:
            self._old_samples = np.pad(
                out.copy(), (self._fade_samples - len(out), 0), mode="constant"
            )

    def run_unaltered(self, wav: np.ndarray, output_frames: int):
        out = self.infer(wav)
        if output_frames == 0:
            return out
        return out[-output_frames:].copy()
