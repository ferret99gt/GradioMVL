import math
import threading
import queue

import numpy as np
import pyaudio

from collections import deque
from voice_conversion import ConversionPipeline
from conversion_thread import Conversion
from audio_input_thread import audio_input
from audio_output_thread import audio_output

class InferenceRt(threading.Thread):
    def __init__(self,
        input_device_idx: int,
        output_device_idx: int,
        input_latency: int,
        input_sample_rate: int,
        output_sample_rate: int,
        MAX_INFER_SAMPLES_VC: int,
        voice_conversion: ConversionPipeline,
        start_queue: queue.Queue,
        status_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.__p = pyaudio.PyAudio()
        
        self.input_device_idx = input_device_idx
        self.output_device_idx = output_device_idx
        self.input_latency = input_latency
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.voice_conversion = voice_conversion
        self.start_queue = start_queue
        self.status_queue = status_queue

    def run(self):
        print(f"input_device_idx: {self.input_device_idx}")
        print(f"output_device_idx: {self.output_device_idx}")
        print(f"input_latency: {self.input_latency}")
        print(f"input_sample_rate: {self.input_sample_rate}")
        print(f"output_sample_rate: {self.output_sample_rate}")
        print(f"MAX_INFER_SAMPLES_VC: {self.MAX_INFER_SAMPLES_VC}")
    
        HDW_FRAMES_PER_BUFFER_IN = math.ceil(self.input_sample_rate * self.input_latency / 5000) # Gather input audio five times faster than selected input.
        HDW_FRAMES_PER_BUFFER_OUT = math.ceil(self.output_sample_rate * self.input_latency / 1000) # Output audio has to match the expected latency of the model conversion thread.
        
        print(f"HDW_FRAMES_PER_BUFFER_IN: {HDW_FRAMES_PER_BUFFER_IN}")
        print(f"HDW_FRAMES_PER_BUFFER_OUT: {HDW_FRAMES_PER_BUFFER_OUT}")
        
        NUM_CHUNKS = math.ceil(self.MAX_INFER_SAMPLES_VC / HDW_FRAMES_PER_BUFFER_IN)
        print(f"NUM_CHUNKS: {NUM_CHUNKS}")

        # create rolling deque for audio input data packets
        q_in = deque(maxlen=NUM_CHUNKS)
        for _ in range(NUM_CHUNKS):
            in_data = np.zeros(HDW_FRAMES_PER_BUFFER_IN, dtype=np.float32)
            q_in.append(in_data)            
        
        # create output deque for audio output packets.
        q_out = queue.Queue()

        # run pipeline
        try:
            audio_input_thread = audio_input(self.__p, q_in, self.input_sample_rate, self.input_device_idx, HDW_FRAMES_PER_BUFFER_IN)
            conversion_thread = Conversion(q_in, q_out, self.voice_conversion, self.input_latency, self.MAX_INFER_SAMPLES_VC, HDW_FRAMES_PER_BUFFER_OUT)
            audio_output_thread = audio_output(self.__p, q_out, self.output_sample_rate, self.output_device_idx, HDW_FRAMES_PER_BUFFER_OUT)
            
            audio_input_thread.start()
            conversion_thread.start()      
            audio_output_thread.start()

            # We're started. Let parent thread know, and wait for stop.
            self.start_queue.put_nowait("Started! Get to talking!")

            while True:
                status = self.status_queue.get()
                
                if status == "pauseToggle":
                    conversion_thread.status_queue.put_nowait(status)
                else:
                    break
        except Exception as inst:
            print(type(inst))    # the exception type
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
                                 # but may be overridden in exception subclasses
            x, y = inst.args     # unpack args
            print('x =', x)
            print('y =', y)
        finally:
            # Did we somehow get here without marking ourselves as started? LET MAIN THREAD KNOW!
            self.start_queue.put_nowait("Failed! UH... check the logs?")

            # Wait for input to stop.
            audio_input_thread.stop_queue.put_nowait("stop")
            audio_input_thread.join()

            # Wait for conversion to stop.
            conversion_thread.status_queue.put_nowait("stop")
            conversion_thread.join()
            
            # Wait for output to stop.
            audio_output_thread.stop_queue.put_nowait("stop")
            audio_output_thread.join()

            # Goodbye, PyAudio.
            self.__p.terminate()

            print("Done cleaning, exiting.")