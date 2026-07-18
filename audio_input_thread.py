import threading
import queue

import numpy as np
import pyaudio
from pyaudio import PyAudio

class audio_input(threading.Thread):
    def __init__(self,
        __p: PyAudio,
        q_in: queue.Queue,
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
        self.dropped_chunks = 0

    def _queue_latest(self, chunk: np.ndarray):
        try:
            self.q_in.put_nowait(chunk)
        except queue.Full:
            try:
                self.q_in.get_nowait()
            except queue.Empty:
                pass

            self.dropped_chunks += 1
            if self.dropped_chunks == 1:
                print("Input queue full; dropping stale capture audio.")
            self.q_in.put_nowait(chunk)
        
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

                    self._queue_latest(in_data_np)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            io_stream.close()
            if self.dropped_chunks:
                print(f"Input dropped {self.dropped_chunks} stale capture chunks total.")
            print("audio_input_thread: stopped")
            
