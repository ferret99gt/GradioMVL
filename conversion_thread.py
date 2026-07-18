import queue
import threading
import time

import librosa
import numpy as np

from collections import deque
from voice_conversion import ConversionPipeline


class Conversion(threading.Thread):
    def __init__(
        self,
        q_in: deque,
        q_work: deque,
        q_out: queue.Queue,
        voice_conversion: ConversionPipeline,
        latency: float,
        input_sample_rate: int,
        output_sample_rate: int,
        max_infer_samples_vc: int,
        capture_chunks_per_inference: int,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)

        self.q_in = q_in
        self.q_work = q_work
        self.q_out = q_out
        self.voice_conversion = voice_conversion
        self.latency = latency / 1000
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.max_infer_samples_vc = max_infer_samples_vc
        self.capture_chunks_per_inference = capture_chunks_per_inference
        self.status_queue = queue.Queue()
        self.paused = False

    def _handle_status(self, status: str) -> bool:
        if status == "pauseToggle":
            self.voice_conversion.reset_overlap()
            self.paused = not self.paused
            return True
        return False

    def _queue_latest(self, out: np.ndarray):
        out_bytes = np.ascontiguousarray(out, dtype=np.float32).tobytes()
        try:
            self.q_out.put_nowait(out_bytes)
        except queue.Full:
            try:
                self.q_out.get_nowait()
            except queue.Empty:
                pass
            self.q_out.put_nowait(out_bytes)

    def _passthrough(self, chunks: list[np.ndarray], output_frames: int) -> np.ndarray:
        out = np.concatenate(chunks)
        if self.input_sample_rate != self.output_sample_rate:
            out = librosa.resample(
                out,
                orig_sr=self.input_sample_rate,
                target_sr=self.output_sample_rate,
                res_type="soxr_vhq",
            )
        if len(out) < output_frames:
            out = np.pad(out, (output_frames - len(out), 0), mode="constant")
        return out[-output_frames:].astype(np.float32, copy=False)

    def run(self):
        next_run = time.perf_counter()
        try:
            while True:
                wait_time = max(0.0, next_run - time.perf_counter())
                try:
                    status = self.status_queue.get(timeout=wait_time)
                    if self._handle_status(status):
                        continue
                    break
                except queue.Empty:
                    pass

                new_chunks = []
                while True:
                    try:
                        new_chunks.append(self.q_in.popleft())
                    except IndexError:
                        break

                if len(new_chunks) > self.capture_chunks_per_inference:
                    dropped = len(new_chunks) - self.capture_chunks_per_inference
                    print(f"Input fell behind; dropped {dropped} stale capture chunks.")
                    new_chunks = new_chunks[-self.capture_chunks_per_inference :]

                if new_chunks:
                    for chunk in new_chunks:
                        self.q_work.append(chunk)

                    input_frames = sum(len(chunk) for chunk in new_chunks)
                    output_frames = round(
                        input_frames * self.output_sample_rate / self.input_sample_rate
                    )
                    wav = np.concatenate(tuple(self.q_work))[-self.max_infer_samples_vc :]

                    if self.paused:
                        out = self._passthrough(new_chunks, output_frames)
                    else:
                        out = self.voice_conversion.run(wav, output_frames)

                    self._queue_latest(out)

                next_run += self.latency
                finished_at = time.perf_counter()
                if finished_at > next_run:
                    print("Model generation logic took longer than specified latency.")
                    next_run = finished_at
        except KeyboardInterrupt:
            pass
        finally:
            print("conversion_process_target: stopped")
