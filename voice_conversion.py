import abc
import argparse
from abc import abstractmethod
from multiprocessing import Value
from struct import unpack

import librosa
import numpy as np
import torch
from util.torch_utils import get_device, is_mps_available

# electron prefers the roaming folder for user data
# TODO test for mac
STUDIO_MODELS_ROOT = "studio_models"

class StudioModelConversionPipeline(abc.ABC):
    def __init__(self, opt: argparse.Namespace):
        self._opt = opt
        self.p_sampling_rate = 16000
        self.pp_sampling_rate = 24000
        self.sampling_rate = 22050

        self.device = get_device()
        self.mac_silicon_device = is_mps_available()

        # TODO: add brancing logic for windows vs mac.
        self.model = torch.jit.load(os.path.join(STUDIO_MODELS_ROOT, "model.pt")).to(self.device)
        self._load_model_preprocessor()

        self._set_target(self._opt.target_speaker)
        #self.tmp_file = os.path.join(STUDIO_MODELS_ROOT, os.urandom(8).hex() + ".wav")

    def _set_target(self, speaker_id: str):
        path_studio = os.path.join(STUDIO_MODELS_ROOT, f"targets/{speaker_id}.npy")
        path = path_studio

        if not os.path.exists(path):
            raise FileNotFoundError(f"Target speaker {speaker_id} not found in {path_studio}.")

        self.target = np.load(path)
        self.target = torch.from_numpy(self.target).unsqueeze(0).to(self.device)

    def _load_model_preprocessor(self):
        if self.mac_silicon_device:
            import coremltools as ct

            self.pmodel = ct.models.MLModel(os.path.join(STUDIO_MODELS_ROOT, "model.mlpackage"))
        else:
            # TODO: add branching logic for windows vs mac
            self.pmodel = torch.jit.load(os.path.join(STUDIO_MODELS_ROOT, "b_model.pt")).to(self.device)

    def infer(self, wav: np.ndarray) -> np.ndarray:
        # TODO: fix hack. we write incoming 22050Hz audio into 16khz file, use preprocessor, then inference which
        # returns file at 24kHz, and write it back into 22050Hz file as expected for rest of the pipeline.
        with torch.no_grad():
            # Replace soundfile write / librosa load for sample rate change with Librosa in-memory conversion.
            wav_src = librosa.resample(wav, orig_sr=self.sampling_rate, target_sr=self.p_sampling_rate, res_type="soxr_hq")
            
            #soundfile.write(self.tmp_file, wav, self.sampling_rate)
            #wav_src, _ = librosa.load(self.tmp_file, sr=self.p_sampling_rate)
            
            if not self.mac_silicon_device:
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
                c = self.pmodel(wav_src.squeeze(1))
            else:
                c = self.pmodel.predict({"input_values": wav_src[np.newaxis, :]})["var_3641"]
                c = torch.from_numpy(c).to(self.device)

            audio = self.model(c, self.target)
            audio = audio[0][0].data.cpu().float().numpy()
            
            # Replace soundfile write / librosa load for sample rate change with Librosa in-memory conversion.
            out = librosa.resample(audio, orig_sr=self.pp_sampling_rate, target_sr=self.sampling_rate, res_type="soxr_hq")

            #soundfile.write(self.tmp_file, audio, self.pp_sampling_rate)
            #out, _ = librosa.load(self.tmp_file, sr=self.sampling_rate)
            #os.remove(self.tmp_file)

        return out

    @abstractmethod
    def run(self, wav: np.ndarray):
        pass