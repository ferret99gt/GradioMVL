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
        NUM_CHUNKS: int,
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
        self.NUM_CHUNKS = NUM_CHUNKS
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.stop_queue = stop_queue
        
        # create rolling deque for io_stream data packets
        self.data = deque(maxlen=NUM_CHUNKS)
        for _ in range(NUM_CHUNKS):
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
                    # This is blocking until enough frames are received.
                    wav_bytes = io_stream.read()
                 
                    # Look into why we're pushing through NumPy. To convert to Float32? To allow for the following splice?
                    in_data_np = np.frombuffer(wav_bytes, dtype=np.float32)

                    # Data is a rolling queue of chunks, totalling about 1.48s of audio. This is to give the model more to work with in terms of infering tone, inflection, etc.
                    data.append(in_data_np)
                    
                    # Push into queue for the model.
                    self.q_in.put_nowait(np.array(data).flatten().astype(np.float32)[-MAX_INFER_SAMPLES_VC:].tobytes())
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            io_stream.close()
            print("audio_input_thread: stopped")
            