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
        q_in: queue.Queue,
        q_work: deque,
        q_out: queue.Queue,
        voice_conversion: ConversionPipeline,
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

    def _handle_pending_status(self) -> bool:
        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                return False

            if not self._handle_status(status):
                return True

    def _wait_for_capture_batch(self) -> list[np.ndarray] | None:
        chunks = []
        while len(chunks) < self.capture_chunks_per_inference:
            if self._handle_pending_status():
                return None

            try:
                chunks.append(self.q_in.get(timeout=0.05))
            except queue.Empty:
                continue

        if self._handle_pending_status():
            return None

        return chunks

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
        try:
            while True:
                new_chunks = self._wait_for_capture_batch()
                if new_chunks is None:
                    break

                for chunk in new_chunks:
                    self.q_work.append(chunk)

                input_frames = sum(len(chunk) for chunk in new_chunks)
                output_frames = round(
                    input_frames * self.output_sample_rate / self.input_sample_rate
                )
                wav = np.concatenate(tuple(self.q_work))[-self.max_infer_samples_vc :]

                started_at = time.perf_counter()
                if self.paused:
                    out = self._passthrough(new_chunks, output_frames)
                else:
                    out = self.voice_conversion.run(wav, output_frames)
                elapsed = time.perf_counter() - started_at

                self._queue_latest(out)

                batch_duration = input_frames / self.input_sample_rate
                if elapsed > batch_duration:
                    print(
                        f"Model generation took {elapsed * 1000:.1f}ms for "
                        f"{batch_duration * 1000:.1f}ms of captured audio."
                    )
        except KeyboardInterrupt:
            pass
        finally:
            print("conversion_process_target: stopped")
