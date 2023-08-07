import threading
import queue

import numpy as np
import pyaudio

class audio_output(threading.Thread):
    def __init__(self,
        __p: pyaudio,
        q_out: queue.Queue,
        sample_rate: int,
        output_device_idx: int,
        MAX_INFER_SAMPLES_VC: int,
        HDW_FRAMES_PER_BUFFER: int,
        stop_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.__p = __p
        self.q_out = q_out
        self.sample_rate = sample_rate
        self.output_device_idx = output_device_idx
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.stop_queue = stop_queue
        
    def run(self):
        # Get PyAudio input stream
        io_stream = self.__p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=False,
            output=True,
            start=False,
            frames_per_buffer=self.HDW_FRAMES_PER_BUFFER,
            output_device_index=self.output_device_idx,
        )
        io_stream.start_stream()    
    
        try:
            while self.stop_queue.empty():
                try:
                    # Block on model output. We have a timeout so if the model never gives us anything, we'll break the block and loop around to check stop_queue
                    out_data = self.q_out.get(timeout=5)
                    if not self.q_out.empty():
                        print("queue is overflowing")
                    
                    # Write to PyAudio output. This blocks till completed. Then we'll loop and wait on the queue for more.
                    io_stream.write(out_data)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            io_stream.close()        
            print("audio_input_thread: stopped")
            