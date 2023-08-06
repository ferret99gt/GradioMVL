import gradio as gr
import math
import numpy as np
import os
import queue
import threading
import time

from inference_rt import InferenceRt
from typing import Optional
from util.data_types import DeviceMap
from util.portaudio_utils import get_devices
from util.torch_utils import set_seed
from voice_conversion import ConversionPipeline

# venv\Scripts\pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# venv\Scripts\pip install gradio sounddevice pyaudio librosa

if __name__ == '__main__':
    sample_rate = 22050 # sampling rate yeah!
    num_samples = 128  # input spect shape num_mels * num_samples
    hop_length = 256  # int(0.0125 * sample_rate)  # 12.5ms - in line with Tacotron 2 paper
    hop_length = int(0.0125 * sample_rate)  # Let's actually try that math, in line with Tacotron 2 paper!

    # corresponds to 1.486s of audio, or 32768 samples in the time domain. This is the number of samples
    # fed into the VC module
    MAX_INFER_SAMPLES_VC = num_samples * hop_length

    SEED = 1234  # numpy & torch PRNG seed
    set_seed(SEED)

    inference_rt_thread = None

    # Create the model.
    print("We're loading the model, please standby! Approximately 30-50 seconds!")
    voice_conversion = ConversionPipeline(sample_rate)
    voice_conversion.set_target("yara")
    
    # warmup models into the cache
    warmup_iterations = 20
    warmup_frames_per_buffer = math.ceil(sample_rate * 400 / 1000)
    for _ in range(warmup_iterations):
        wav = np.random.rand(MAX_INFER_SAMPLES_VC).astype(np.float32)
        voice_conversion.run(wav, warmup_frames_per_buffer)
    print("Model ready and warmed up!")

    # Retrieve available voices.
    voiceDirectory = "studio_models\\targets"
    voices = []
    for filename in os.listdir(voiceDirectory):
        f = os.path.join(voiceDirectory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            voices.append(filename[0:-4])

    # Retrieve available devices.
    devices = get_devices();
    inputNames = [p.name for p in devices['inputs']];
    outputNames = [p.name for p in devices['outputs']];

    def setVoice(target_speaker):
        global voice_conversion
        
        if target_speaker is None or len(target_speaker) == 0:
            return [gr.update(interactive=False), "Invalid Voice selected, please pick another."]

        voiceFile = os.path.join(voiceDirectory, f"{target_speaker}.npy")
        if not os.path.isfile(voiceFile):
            return [gr.update(interactive=False), "Selected Voice not found, please pick another."]
            
        voice_conversion.set_target(target_speaker)
            
        return [gr.update(interactive=True), "Voice prepared! You can start now!"]

    def startGenerateVoice(input, output, latency):
        global inference_rt_thread, voice_conversion
    
        inputDevice =  [p for p in devices['inputs'] if p.name == input];
        
        if inputDevice is None or len(inputDevice) != 1:
            return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
            
        outputDevice =  [p for p in devices['outputs'] if p.name == output];
        if outputDevice is None or len(outputDevice) != 1:
            return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
            
        if not voice_conversion.isTargetSet:
            return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), "A voice is not selected yet, conversion not started."]

        start_queue = queue.Queue()
        stop_queue = queue.Queue()
        
        inference_rt_thread = InferenceRt(inputDevice[0].index, outputDevice[0].index, latency, sample_rate, MAX_INFER_SAMPLES_VC, voice_conversion, start_queue, stop_queue, args=())
        inference_rt_thread.start()

        # Wait for start queue
        txt = inference_rt_thread.start_queue.get()
            
        return [gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), txt]
        
    def stopGenerateVoice():
        global inference_rt_thread

        if inference_rt_thread is not None and inference_rt_thread.is_alive():
            # Wait for end.        
            inference_rt_thread.stop_queue.put("stop")
            inference_rt_thread.join()
        
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), "Stopped!"]    

    with gr.Blocks() as demo:
        gr.Markdown("Select an input and output devices, then press Start.")
        with gr.Row():
            inputDrop = gr.Dropdown(choices=inputNames, label="Input Device");
            outputDrop = gr.Dropdown(choices=outputNames, label="Output Device");
            latencySlider = gr.Slider(100, 2000, label="Latency (one way)", step=50, value=300);
        with gr.Row():
            voiceDrop = gr.Dropdown(choices=voices, value="yara", label="Voice File");
            startButton = gr.Button(value="Start", interactive=True);
            stopButton = gr.Button(value="Stop", interactive=False);
        with gr.Row():
            text = gr.Textbox(label="Status");
            
        voiceDrop.input(fn=setVoice, inputs=[voiceDrop], outputs=[startButton, text])
        startButton.click(fn=startGenerateVoice, inputs=[inputDrop, outputDrop, latencySlider], outputs=[voiceDrop, startButton, stopButton, text])
        stopButton.click(fn=stopGenerateVoice, inputs=None, outputs=[voiceDrop, startButton, stopButton, text])

    demo.launch()