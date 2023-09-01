import queue
import threading
import time

import numpy as np

from collections import deque
from voice_conversion import ConversionPipeline

class Conversion(threading.Thread):
    def __init__(self,
        q_in: deque,
        q_out: queue.Queue,
        voice_conversion: ConversionPipeline,
        latency: float,
        MAX_INFER_SAMPLES_VC: int,
        HDW_FRAMES_PER_BUFFER: int,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
 
        self.q_in = q_in
        self.q_out = q_out
        self.voice_conversion = voice_conversion
        self.latency = latency / 1000 # milliseconds
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.status_queue = queue.Queue()
        self.paused = False
        
    def run(self):
        try:
            while True:
                # Time it.
                time_start = time.time()
                
                # Get the PyAudio input deque data.
                # q_in is a rolling queue of chunks, totalling about 1.6s of audio. This is to give the model more to work with in terms of infering intonation, etc.
                wav = np.array(self.q_in).flatten()[-self.MAX_INFER_SAMPLES_VC:]

                # Infer!
                if not self.paused:
                    out = self.voice_conversion.run(wav, self.HDW_FRAMES_PER_BUFFER)
                else:
                    out = wav[-self.HDW_FRAMES_PER_BUFFER:]

                # Queue the result up for the audio output thread.
                # This queue has a size of 1, and we're blocking here. This is to ensure the output thread has picked up the last segment first.
                self.q_out.put_nowait(out.tobytes())
                
                # Check the status queue for instructions.
                try:
                    status = self.status_queue.get_nowait()
                    
                    # Pause? Stop?
                    if status == "pauseToggle":
                        self.paused = not self.paused
                    else:
                        break
                        
                except queue.Empty:
                    pass
                    
                # We need to finish all work before self.latency or we're going to cut stuff off.
                duration = time.time() - time_start
                if duration < self.latency:
                    sleep_time = self.latency - duration
                    if(sleep_time < 0.001):
                        sleep_time = 0.001
                    
                    # Sleep while waiting for new audio input.
                    time.sleep(sleep_time)
                else:
                    # Report we overran, then loop back around.
                    print("Model generation logic took longer than specified latency.")
                    
        except KeyboardInterrupt:
            pass
        finally:
            print("conversion_process_target: stopped")