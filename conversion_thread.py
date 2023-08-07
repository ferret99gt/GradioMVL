import threading
import queue
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from voice_conversion import StudioModelConversionPipeline, ConversionPipeline

class Conversion(threading.Thread):
    def __init__(self,
        q_in: queue.Queue,
        q_out: queue.Queue,
        voice_conversion: ConversionPipeline,
        HDW_FRAMES_PER_BUFFER: int,
        stop_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.q_in = q_in
        self.q_out = q_out
        self.voice_conversion = voice_conversion
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.stop_queue = stop_queue
        
    def run(self):
        try:
            while self.stop_queue.empty():
                try:
                    # Wait on PyAudio input. We have a timeout so if the model never gives us anything, we'll break the block and loop around to check stop_queue
                    wav_bytes = self.q_in.get(timeout=5)

                    wav = np.frombuffer(wav_bytes, dtype=np.float32)
                    out = self.voice_conversion.run(wav, self.HDW_FRAMES_PER_BUFFER)

                    self.q_out.put_nowait(out.tobytes())
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            print("conversion_process_target: stopped")