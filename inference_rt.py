import math
import threading
import queue

import pyaudio

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
    
        HDW_FRAMES_PER_BUFFER_IN = math.ceil(self.input_sample_rate * self.input_latency / 1000)
        HDW_FRAMES_PER_BUFFER_OUT = math.ceil(self.output_sample_rate * self.input_latency / 1000)
        
        print(f"HDW_FRAMES_PER_BUFFER_IN: {HDW_FRAMES_PER_BUFFER_IN}")
        print(f"HDW_FRAMES_PER_BUFFER_OUT: {HDW_FRAMES_PER_BUFFER_OUT}")

        # init
        q_in, q_out = queue.Queue(), queue.Queue()

        # run pipeline
        try:
            audio_input_thread = audio_input(self.__p, q_in, self.input_sample_rate, self.input_device_idx, self.MAX_INFER_SAMPLES_VC, HDW_FRAMES_PER_BUFFER_IN, queue.Queue())
            conversion_thread = Conversion(q_in, q_out, self.voice_conversion, HDW_FRAMES_PER_BUFFER_OUT, queue.Queue(), args=())
            audio_output_thread = audio_output(self.__p, q_out, self.output_sample_rate, self.output_device_idx, HDW_FRAMES_PER_BUFFER_OUT, queue.Queue())
            
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