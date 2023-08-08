import math
import threading
import queue

import numpy as np
import pyaudio

from collections import deque

class audio_input(threading.Thread):
    def __init__(self,
        __p: pyaudio,
        q_in: queue.Queue,
        sample_rate: int,
        input_device_idx: int,
        MAX_INFER_SAMPLES_VC: int,
        HDW_FRAMES_PER_BUFFER: int,
        stop_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.__p = __p
        self.q_in = q_in
        self.sample_rate = sample_rate
        self.input_device_idx = input_device_idx
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.stop_queue = stop_queue
        
        self.NUM_CHUNKS = math.ceil(MAX_INFER_SAMPLES_VC / HDW_FRAMES_PER_BUFFER)
        print(f"NUM_CHUNKS: {self.NUM_CHUNKS}")
        
        # create rolling deque for io_stream data packets
        self.data = deque(maxlen=self.NUM_CHUNKS)
        for _ in range(self.NUM_CHUNKS):
            in_data = np.zeros(HDW_FRAMES_PER_BUFFER, dtype=np.float32)
            self.data.append(in_data)        
        
    def run(self):
        # Get PyAudio input stream
        io_stream = self.__p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=False,
            start=False,
            frames_per_buffer=self.HDW_FRAMES_PER_BUFFER,
            input_device_index=self.input_device_idx,
        )
        io_stream.start_stream()    
    
        try:
            while self.stop_queue.empty():
                try:
                    # This is blocking until enough frames are received. We're silently ignoring overflows.
                    wav_bytes = io_stream.read(self.HDW_FRAMES_PER_BUFFER, False)
                 
                    # The model (and librosa) ultimately wants an ndarray. Convert it and ensure it's float32.
                    in_data_np = np.frombuffer(wav_bytes, dtype=np.float32)

                    # Data is a rolling queue of chunks, totalling about 1.6s of audio. This is to give the model more to work with in terms of infering intonation, etc.
                    self.data.append(in_data_np)
                    
                    # Push into queue for the model. We splice because it's possible that deque has 1 more chunk than MAX_INFER_SAMPLES_VC wants.
                    self.q_in.put_nowait(np.array(self.data).flatten()[-self.MAX_INFER_SAMPLES_VC:])
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            io_stream.close()
            print("audio_input_thread: stopped")
            