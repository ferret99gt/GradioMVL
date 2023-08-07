import threading
import queue

from voice_conversion import ConversionPipeline

class Conversion(threading.Thread):
    def __init__(self,
        q_in: queue.Queue,
        q_out: queue.Queue,
        voice_conversion: ConversionPipeline,
        HDW_FRAMES_PER_BUFFER: int,
        status_queue: queue.Queue,
        args=(),
        kwargs=None,
    ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        
        self.q_in = q_in
        self.q_out = q_out
        self.voice_conversion = voice_conversion
        self.HDW_FRAMES_PER_BUFFER = HDW_FRAMES_PER_BUFFER
        self.status_queue = status_queue
        self.paused = False
        
    def run(self):
        try:
            while True:
                try:
                    # Wait on PyAudio input. We have a timeout so if the model never gives us anything, we'll break the block and loop around to check stop_queue
                    wav = self.q_in.get(timeout=5)

                    # Infer!
                    if not self.paused:
                        out = self.voice_conversion.run(wav, self.HDW_FRAMES_PER_BUFFER)
                    else:
                        out = wav[-self.HDW_FRAMES_PER_BUFFER:]

                    # Queue it up for the audio output thread.
                    self.q_out.put_nowait(out.tobytes())
                except queue.Empty:
                    pass
                
                try:
                    # Check the status queue for instructions.
                    status = self.status_queue.get_nowait()
                    
                    # Pause? Stop?
                    if status == "pauseToggle":
                        self.paused = not self.paused
                    else:
                        break
                        
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            print("conversion_process_target: stopped")