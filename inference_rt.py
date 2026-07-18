import math
import queue
import threading
import time

import numpy as np
import pyaudio

from collections import deque
from voice_conversion import ConversionPipeline
from conversion_thread import Conversion
from audio_input_thread import audio_input
from audio_output_thread import audio_output


class InferenceRt(threading.Thread):
    def __init__(
        self,
        input_device_idx: int,
        output_device_idx: int,
        input_latency: int,
        input_sample_rate: int,
        output_sample_rate: int,
        max_infer_samples_vc: int,
        capture_chunks_per_inference: int,
        voice_conversion: ConversionPipeline,
        start_queue: queue.Queue,
        status_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)

        self.__p = pyaudio.PyAudio()
        self.input_device_idx = input_device_idx
        self.output_device_idx = output_device_idx
        self.input_latency = input_latency
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.max_infer_samples_vc = max_infer_samples_vc
        self.capture_chunks_per_inference = capture_chunks_per_inference
        self.voice_conversion = voice_conversion
        self.start_queue = start_queue
        self.status_queue = status_queue

    def run(self):
        print(f"input_device_idx: {self.input_device_idx}")
        print(f"output_device_idx: {self.output_device_idx}")
        print(f"input_latency: {self.input_latency}")
        print(f"input_sample_rate: {self.input_sample_rate}")
        print(f"output_sample_rate: {self.output_sample_rate}")
        print(f"MAX_INFER_SAMPLES_VC: {self.max_infer_samples_vc}")

        input_frames_per_buffer = round(
            self.input_sample_rate
            * (self.input_latency / 1000)
            / self.capture_chunks_per_inference
        )
        print(f"INPUT_HDW_FRAMES_PER_BUFFER: {input_frames_per_buffer}")

        work_chunks = math.ceil(self.max_infer_samples_vc / input_frames_per_buffer)
        q_in = queue.Queue(maxsize=self.capture_chunks_per_inference * 2)
        q_work = deque(maxlen=work_chunks)
        for _ in range(work_chunks):
            q_work.append(np.zeros(input_frames_per_buffer, dtype=np.float32))
        q_out = queue.Queue(maxsize=2)

        audio_input_thread = None
        conversion_thread = None
        audio_output_thread = None
        startup_reported = False

        try:
            audio_input_thread = audio_input(
                self.__p,
                q_in,
                self.input_sample_rate,
                self.input_device_idx,
                input_frames_per_buffer,
            )
            conversion_thread = Conversion(
                q_in,
                q_work,
                q_out,
                self.voice_conversion,
                self.input_sample_rate,
                self.output_sample_rate,
                self.max_infer_samples_vc,
                self.capture_chunks_per_inference,
            )
            audio_output_thread = audio_output(
                self.__p, q_out, self.output_sample_rate, self.output_device_idx
            )

            audio_input_thread.start()
            startup_deadline = time.perf_counter() + max(
                5.0, self.input_latency / 1000 * 2
            )
            while q_in.qsize() < self.capture_chunks_per_inference:
                if not audio_input_thread.is_alive():
                    raise RuntimeError("Audio input stream stopped during startup.")
                if time.perf_counter() >= startup_deadline:
                    raise TimeoutError("Timed out waiting for microphone audio.")
                time.sleep(0.005)

            conversion_thread.start()
            audio_output_thread.start()
            self.start_queue.put_nowait("Started! Get to talking!")
            startup_reported = True

            while True:
                try:
                    status = self.status_queue.get(timeout=0.1)
                except queue.Empty:
                    if not audio_input_thread.is_alive():
                        raise RuntimeError("Audio input thread stopped unexpectedly.")
                    if not conversion_thread.is_alive():
                        raise RuntimeError("Conversion thread stopped unexpectedly.")
                    if not audio_output_thread.is_alive():
                        raise RuntimeError("Audio output thread stopped unexpectedly.")
                    continue

                if status == "pauseToggle":
                    conversion_thread.status_queue.put_nowait(status)
                else:
                    break
        except Exception as error:
            print(f"Inference pipeline failed: {error!r}")
            if not startup_reported:
                self.start_queue.put_nowait(f"Failed: {error}")
        finally:
            if audio_input_thread is not None and audio_input_thread.is_alive():
                audio_input_thread.stop_queue.put_nowait("stop")
                audio_input_thread.join()

            if conversion_thread is not None and conversion_thread.is_alive():
                conversion_thread.status_queue.put_nowait("stop")
                conversion_thread.join()

            if audio_output_thread is not None and audio_output_thread.is_alive():
                audio_output_thread.stop_queue.put_nowait("stop")
                audio_output_thread.join()

            self.__p.terminate()
            print("Done cleaning, exiting.")
