import argparse
import math
import multiprocessing
import os
import time
from collections import deque
from datetime import datetime
from multiprocessing import Process, Queue, Value
from queue import Empty
from typing import Callable, Optional
from enum import Enum

import numpy as np
import pyaudio

from util.torch_utils import get_device, set_seed
from util.perf_counter import DebugPerfCounter
from util.timedscope import TimedScope, get_logger
from util.multiprocessing_utils import SharedCounter
from voice_conversion import StudioModelConversionPipeline

_LOGGER = get_logger(os.path.basename(__file__))
SESSION_ID = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

num_samples = 128  # input spect shape num_mels * num_samples
hop_length = 256  # int(0.0125 * sample_rate)  # 12.5ms - in line with Tacotron 2 paper

# corresponds to 1.486s of audio, or 32768 samples in the time domain. This is the number of samples
# fed into the VC module
MAX_INFER_SAMPLES_VC = num_samples * hop_length

sample_rate = 22050
SEED = 1234  # numpy & torch PRNG seed

set_seed(SEED)

class InferencePipelineMode(Enum):
    offline_with_overlap = "offline_with_overlap"
    online_raw = "online_raw"
    online_with_past_future = "online_with_past_future"
    online_crossfade = "online_crossfade"

    def __str__(self):
        return self.value

# ----------------
#  PyAudio Setup
# ----------------

p = pyaudio.PyAudio()
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = sample_rate

PACKET_START_S = time.time()
WAV: Optional[np.ndarray] = None

# TODO sidroopdaska: remove numpy.ndarray allocation
# TODO sidroopdaska: create a lock-free, single producer and single consumer ring buffer
def get_io_stream_callback(
    q_in: Queue,
    q_in_counter: Value,
    data: list,
    q_out: Queue,
    q_out_counter: Value,
    MAX_RECORD_SEGMENTS: int,
) -> Callable:
    def callback(in_data, frame_count, time_info, status):
        global PACKET_START_S, WAV, BUFFER_OVERFLOW

        _LOGGER.debug(f"io_stream_callback duration={time.time() - PACKET_START_S}")
        _LOGGER.debug(f"io_stream_callback frame_count={frame_count}")
        if status:
            _LOGGER.warn(f"status: {status}")

        in_data_np = np.frombuffer(in_data, dtype=np.float32)

        data.append(in_data_np)
        q_in.put_nowait(
            (
                PACKET_START_S,
                # passing data as bytes in multiprocessing:Queue is quicker
                np.array(data).flatten().astype(np.float32)[-MAX_INFER_SAMPLES_VC:].tobytes(),
            )
        )
        q_in_counter.increment()

        # prepare output
        out_data = None
        p_start_s = None

        if q_out_counter.value == 0:
            _LOGGER.info("q_out: underflow")
            out_data = np.zeros(frame_count).astype(np.float32).tobytes()
        elif q_out_counter.value == 1:
            p_start_s, out_data = q_out.get_nowait()
            q_out_counter.increment(-1)

        else:
            _LOGGER.info("q_out: overflow")

            while not q_out.empty():
                try:
                    p_start_s, out_data = q_out.get_nowait()
                    q_out_counter.increment(-1)
                except Empty:
                    pass

        if p_start_s is not None:
            _LOGGER.info(f"roundtrip: {time.time() - p_start_s}")

        PACKET_START_S = time.time()
        return (out_data, pyaudio.paContinue)

    return callback

# --------------------
#  Conversion pipeline
# --------------------
class ConversionPipeline(StudioModelConversionPipeline):
    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)

        fade_duration_ms = 20
        self._fade_samples = int(fade_duration_ms / 1000 * sample_rate)  # 20ms

        self._linear_fade_in = np.linspace(0, 1, self._fade_samples, dtype=np.float32)
        self._linear_fade_out = np.linspace(1, 0, self._fade_samples, dtype=np.float32)
        self._old_samples = np.zeros(self._fade_samples, dtype=np.float32)

    def run(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        if self._opt.mode == InferencePipelineMode.online_crossfade:
            return self.run_cross_fade(wav, HDW_FRAMES_PER_BUFFER)
        elif self._opt.mode == InferencePipelineMode.online_with_past_future:
            raise NotImplementedError
        else:
            raise Exception(f"Mode: {self._opt.mode} unsupported")

    # Linear cross-fade
    def run_cross_fade(self, wav: np.ndarray, HDW_FRAMES_PER_BUFFER: int):
        with DebugPerfCounter("voice_conversion", _LOGGER):
            with DebugPerfCounter("studio_model", _LOGGER):
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


# -------------------
#  Main app processes
# -------------------
def conversion_process_target(
    stop: Value,
    q_in: Queue,
    q_out: Queue,
    q_in_counter: SharedCounter,
    q_out_counter: SharedCounter,
    model_warmup_complete: Value,
    opt: dict,
    HDW_FRAMES_PER_BUFFER: int,
):
    voice_conversion = ConversionPipeline(opt)

    # warmup models into the cache
    warmup_iterations = 20
    for _ in range(warmup_iterations):
        wav = np.random.rand(MAX_INFER_SAMPLES_VC).astype(np.float32)
        voice_conversion.run(wav, HDW_FRAMES_PER_BUFFER)
    model_warmup_complete.value = 1

    try:
        while not stop.value:
            p_start_s, wav_bytes = q_in.get()
            q_in_counter.increment(-1)

            wav = np.frombuffer(wav_bytes, dtype=np.float32)
            out = voice_conversion.run(wav, HDW_FRAMES_PER_BUFFER)

            q_out.put_nowait((p_start_s, out.tobytes()))
            q_out_counter.increment()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.info("conversion_process_target: stopped")


def run_inference_rt(
    opt: argparse.Namespace,
    stop_pipeline: Value,
    has_pipeline_started: Optional[Value] = None,
):
    global PACKET_START_S, WAV

    HDW_FRAMES_PER_BUFFER = math.ceil(sample_rate * opt.callback_latency_ms.value / 1000)
    NUM_CHUNKS = math.ceil(MAX_INFER_SAMPLES_VC / HDW_FRAMES_PER_BUFFER)
    MAX_RECORD_SEGMENTS = 5 * 60 * sample_rate // HDW_FRAMES_PER_BUFFER  # 5 mins in duration
    # make sure dependencies are updated before starting the pipeline
    _LOGGER.debug(f"MAX_RECORD_SEGMENTS: {MAX_RECORD_SEGMENTS}")
    _LOGGER.debug(f"HDW_FRAMES_PER_BUFFER: {HDW_FRAMES_PER_BUFFER}")
    _LOGGER.debug(f"NUM_CHUNKS: {NUM_CHUNKS}")

    # init
    stop_process = Value("i", 0)
    model_warmup_complete = Value("i", 0)
    q_in, q_out = Queue(), Queue()  # TODO sidroopdaska: create wrapper class for multiprocessing:Queue & shared counter
    q_in_counter, q_out_counter = SharedCounter(0), SharedCounter(0)

    # create rolling deque for io_stream data packets
    data = deque(maxlen=NUM_CHUNKS)
    for _ in range(NUM_CHUNKS):
        in_data = np.zeros(HDW_FRAMES_PER_BUFFER, dtype=np.float32)
        data.append(in_data)

    # run pipeline
    try:
        _LOGGER.info(f"backend={get_device()}")
        _LOGGER.info(f"opt={opt}")

        conversion_process = Process(
            target=conversion_process_target,
            args=(
                stop_process,
                q_in,
                q_out,
                q_in_counter,
                q_out_counter,
                model_warmup_complete,
                opt,
                HDW_FRAMES_PER_BUFFER,
            ),
        )
        conversion_process.start()

        with TimedScope("model_warmup", _LOGGER):
            while not model_warmup_complete.value:
                time.sleep(1)

        io_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            start=False,
            frames_per_buffer=HDW_FRAMES_PER_BUFFER,
            input_device_index=opt.input_device_idx,
            output_device_index=opt.output_device_idx,
            stream_callback=get_io_stream_callback(
                q_in,
                q_in_counter,
                data,
                q_out,
                q_out_counter,
                MAX_RECORD_SEGMENTS,
            ),
        )
        io_stream.start_stream()
        PACKET_START_S = time.time()

        # hook for calling process
        if has_pipeline_started is not None:
            with has_pipeline_started.get_lock():
                has_pipeline_started.value = 1

        while not stop_pipeline.value:
            time.sleep(0.2)

    finally:
        with stop_process.get_lock():
            stop_process.value = 1
        conversion_process.join()

        if io_stream:
            io_stream.close()
        p.terminate()

        # empty out the queues prior to deletion
        while not q_in.empty():
            try:
                q_in.get_nowait()
            except Empty:
                pass
        while not q_out.empty():
            try:
                q_out.get_nowait()
            except Empty:
                pass

        del q_in, q_out, q_in_counter, q_out_counter, stop_process, model_warmup_complete
        _LOGGER.info("Done cleaning, exiting.")