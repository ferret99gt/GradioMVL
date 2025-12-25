import math
import threading
import queue

import numpy as np
import pyaudio
from pyaudio import PyAudio

from collections import deque

class audio_input(threading.Thread):
    def __init__(self,
        __p: PyAudio,
        q_in: deque,
        sample_rate: int,
        input_device_idx: int,
        HDW_FRAMES_PER_BUFFER: int,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.__p = __p
        self.q_in = q_in
        self.sample_rate = sample_rate
        self.input_device_idx = input_device_idx
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.stop_queue = queue.Queue()
        
    def run(self):
        # Get PyAudio input stream
        io_stream = self.__p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=False,
            start=False,
            input_device_index=self.input_device_idx,
            frames_per_buffer=self.HDW_FRAMES_PER_BUFFER,
        )
        io_stream.start_stream()    
    
        try:
            while self.stop_queue.empty():
                try:
                    # This is blocking until enough frames are received. We're silently ignoring overflows.
                    wav_bytes = io_stream.read(self.HDW_FRAMES_PER_BUFFER, False)
                 
                    # The model (and librosa) ultimately wants an ndarray. Convert it and ensure it's float32.
                    in_data_np = np.frombuffer(wav_bytes, dtype=np.float32)

                    # q_in is a rolling queue of chunks, totalling about 1.6s of audio. Bound it and drop oldest if we are behind to keep latency from ballooning.
                    if self.q_in.maxlen and len(self.q_in) >= self.q_in.maxlen:
                        self.q_in.popleft()
                    self.q_in.append(in_data_np)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            io_stream.close()
            print("audio_input_thread: stopped")
            
