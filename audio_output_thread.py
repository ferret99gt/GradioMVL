import threading
import queue

import pyaudio

class audio_output(threading.Thread):
    def __init__(self,
        __p: pyaudio,
        q_out: queue.Queue,
        sample_rate: int,
        output_device_idx: int,
        HDW_FRAMES_PER_BUFFER: int,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.__p = __p
        self.q_out = q_out
        self.sample_rate = sample_rate
        self.output_device_idx = output_device_idx
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.stop_queue = queue.Queue()
        
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
                    out_data = self.q_out.get(timeout=2)
                    if not self.q_out.empty():
                        print("output queue is overflowing")

                        # Clear the extras. It's going to cause an audio blip but nothing we can do to prevent that.
                        while not self.q_out.empty():
                            try:
                                self.q_out.get_nowait()
                            except Empty:
                                continue
                    
                    # Write to PyAudio output. This blocks till completed. Then we'll loop and wait on the queue for more.
                    io_stream.write(out_data)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            io_stream.close()        
            print("audio_output_thread: stopped")
            