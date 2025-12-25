import math
import queue
import threading
import time

import numpy as np

from collections import deque
from voice_conversion import ConversionPipeline

class Conversion(threading.Thread):
    def __init__(self,
        q_in: deque,
        q_work: deque,
        q_out: queue.Queue,
        voice_conversion: ConversionPipeline,
        latency: float,
        output_sample_rate: int,
        MAX_INFER_SAMPLES_VC: int,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
 
        self.q_in = q_in
        self.q_work = q_work
        self.q_out = q_out
        self.voice_conversion = voice_conversion
        self.latency = latency / 1000 # milliseconds
        self.output_sample_rate = output_sample_rate        
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.status_queue = queue.Queue()
        self.paused = False
        
    def run(self):
        try:
            while True:
                # Time it.
                time_start = time.time()
                
                # Get the PyAudio input deque data.
                newChunks = 0
                try:
                    while True:
                        self.q_work.append(self.q_in.popleft())
                        newChunks += 1
                except IndexError:
                    pass
                
                if newChunks > 0:
                    # The frames per buffer is the sampling rate times the duration of audio. (Latency is in milliseconds)
                    # Input audio is gathering chunks 10 times faster than the selected latency. (That is, 500ms -> 50ms)
                    # So divide the number of new chunks we got by 10.
                    # This is in case we had too few or extras from the input deque and adjust accordingly.
                    HDW_FRAMES_PER_BUFFER = math.ceil(self.output_sample_rate * self.latency * (newChunks / 10)) 
                    #print(f"newChunks: {newChunks}, HDW_FRAMES_PER_BUFFER: {HDW_FRAMES_PER_BUFFER}")
                    
                    # q_work is a rolling queue of chunks, totalling about 1.6s of audio. This is to give the model more to work with in terms of infering intonation, etc.
                    wav = np.array(self.q_work).flatten()[-self.MAX_INFER_SAMPLES_VC:]

                    # Infer!
                    if not self.paused:
                        out = self.voice_conversion.run(wav, HDW_FRAMES_PER_BUFFER)
                    else:
                        out = wav[-HDW_FRAMES_PER_BUFFER:]

                    # Queue the result up for the audio output thread.
                    self.q_out.put_nowait(out.tobytes())
                
                # Check the status queue for instructions.
                try:
                    status = self.status_queue.get_nowait()
                    
                    # Pause? Stop?
                    if status == "pauseToggle":
                        self.voice_conversion.reset_overlap()
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
