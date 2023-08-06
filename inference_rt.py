import logging
import math
import os
import threading
import time
import queue
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pyaudio

from voice_conversion import StudioModelConversionPipeline, ConversionPipeline
from conversion_thread import Conversion

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

FORMAT = "%(name)s: %(message)s"
level = logging.INFO
logging.basicConfig(level=level, format=FORMAT)
_LOGGER = logging.getLogger(os.path.basename(__file__))

PACKET_START_S = time.time()

class InferenceRt(threading.Thread):
    def __init__(self,
        input_device_idx: int,
        output_device_idx: int,
        callback_latency_ms: int,
        sample_rate: int,
        MAX_INFER_SAMPLES_VC: int,
        voice_conversion: ConversionPipeline,
        start_queue: queue.Queue,
        stop_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.__p = pyaudio.PyAudio()
        
        self.PACKET_START_S = time.time()
        self.input_device_idx = input_device_idx
        self.output_device_idx = output_device_idx
        self.callback_latency_ms = callback_latency_ms
        self.sample_rate = sample_rate
        self.MAX_INFER_SAMPLES_VC = MAX_INFER_SAMPLES_VC
        self.voice_conversion = voice_conversion
        self.start_queue = start_queue
        self.stop_queue = stop_queue

    def get_io_stream_callback(
        self,
        q_in: queue.Queue,
        q_out: queue.Queue,
        data: list,
        MAX_INFER_SAMPLES_VC: int,
    ) -> Callable:
        def callback(in_data, frame_count, time_info, status):
            _LOGGER.debug(f"io_stream_callback duration={time.time() - self.PACKET_START_S}")
            _LOGGER.debug(f"io_stream_callback frame_count={frame_count}")
            if status:
                _LOGGER.warn(f"status: {status}")

            in_data_np = np.frombuffer(in_data, dtype=np.float32)

            data.append(in_data_np)
            q_in.put_nowait(
                (
                    self.PACKET_START_S,
                    # passing data as bytes in multiprocessing:Queue is quicker
                    np.array(data).flatten().astype(np.float32)[-MAX_INFER_SAMPLES_VC:].tobytes(),
                )
            )

            # prepare output
            out_data = None
            p_start_s = None

            try:
                p_start_s, out_data = q_out.get_nowait()
                if not q_out.empty():
                    _LOGGER.info("q_out: overflow")
                    while not q_out.empty():
                        try:
                            p_start_s, out_data = q_out.get_nowait()
                        except Empty:
                            pass
            except queue.Empty:
                _LOGGER.info("q_out: underflow")
                out_data = np.zeros(frame_count).astype(np.float32).tobytes()            
            
            if p_start_s is not None:
                _LOGGER.info(f"roundtrip: {time.time() - p_start_s}")

            self.PACKET_START_S = time.time()
            return (out_data, pyaudio.paContinue)

        return callback

    def run(self):
        # being lazy, sue me.
        input_device_idx = self.input_device_idx
        output_device_idx = self.output_device_idx
        callback_latency_ms = self.callback_latency_ms
        sample_rate = self.sample_rate
        MAX_INFER_SAMPLES_VC = self.MAX_INFER_SAMPLES_VC
        voice_conversion = self.voice_conversion
        start_queue = self.start_queue
        stop_queue = self.stop_queue
        
        _LOGGER.info(f"input_device_idx: {input_device_idx}")
        _LOGGER.info(f"output_device_idx: {output_device_idx}")
        _LOGGER.info(f"callback_latency_ms: {callback_latency_ms}")
        _LOGGER.info(f"sample_rate: {sample_rate}")
        _LOGGER.info(f"MAX_INFER_SAMPLES_VC: {MAX_INFER_SAMPLES_VC}")
    
        HDW_FRAMES_PER_BUFFER = math.ceil(sample_rate * callback_latency_ms / 1000)
        NUM_CHUNKS = math.ceil(MAX_INFER_SAMPLES_VC / HDW_FRAMES_PER_BUFFER)
        _LOGGER.info(f"HDW_FRAMES_PER_BUFFER: {HDW_FRAMES_PER_BUFFER}")
        _LOGGER.info(f"NUM_CHUNKS: {NUM_CHUNKS}")

        # init
        conversion_stop_queue = queue.Queue()
        q_in, q_out = queue.Queue(), queue.Queue()

        # create rolling deque for io_stream data packets
        data = deque(maxlen=NUM_CHUNKS)
        for _ in range(NUM_CHUNKS):
            in_data = np.zeros(HDW_FRAMES_PER_BUFFER, dtype=np.float32)
            data.append(in_data)

        # run pipeline
        try:
            conversion_thread = Conversion(q_in, q_out, voice_conversion, HDW_FRAMES_PER_BUFFER, conversion_stop_queue, args=())
            conversion_thread.start()      
            
            io_stream = self.__p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                input=True,
                output=True,
                start=False,
                frames_per_buffer=HDW_FRAMES_PER_BUFFER,
                input_device_index=input_device_idx,
                output_device_index=output_device_idx,
                stream_callback=self.get_io_stream_callback(
                    q_in,
                    q_out,
                    data,
                    MAX_INFER_SAMPLES_VC,
                ),
            )
            io_stream.start_stream()
            self.PACKET_START_S = time.time()

            # We're started. Let parent thread know, and wait for stop.
            start_queue.put_nowait("Started! Get to talking!")

            stop_queue.get()
        except Exception as inst:
            _LOGGER.info(type(inst))    # the exception type
            _LOGGER.info(inst.args)     # arguments stored in .args
            _LOGGER.info(inst)          # __str__ allows args to be printed directly,
                                 # but may be overridden in exception subclasses
            x, y = inst.args     # unpack args
            _LOGGER.info('x =', x)
            _LOGGER.info('y =', y)
        finally:
            # Did we somehow get here without marking ourselves as started? LET MAIN THREAD KNOW!
            start_queue.put_nowait("Failed! UH... check the logs?")
            
            # Wait for conversion to stop.
            conversion_thread.stop_queue.put_nowait("stop")
            conversion_thread.join()

            if io_stream:
                io_stream.close()
            self.__p.terminate()

            # empty out the queues prior to deletion
            while not q_in.empty():
                try:
                    q_in.get_nowait()
                except Empty:
                    pass
            while not q_out.empty():
                try:
                    q_out.get_nowait()
                except Empty:
                    pass

            del q_in, q_out, start_queue, stop_queue
            _LOGGER.info("Done cleaning, exiting.")