import gradio as gr
import math
import numpy as np
import os
import queue

from inference_rt import InferenceRt
from modules.portaudio_utils import get_devices
from modules.torch_utils import set_seed
from voice_conversion import ConversionPipeline

# venv\Scripts\pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# venv\Scripts\pip install gradio sounddevice pyaudio librosa

# Samples, rates, etc.
sample_rate = 22050 # sampling rate yeah!
num_samples = 128  # input spect shape num_mels * num_samples
hop_length = 256  # int(0.0125 * sample_rate)  # 12.5ms - in line with Tacotron 2 paper
hop_length = int(0.0125 * sample_rate)  # Let's actually try that math, in line with Tacotron 2 paper!

# MVL: corresponds to 1.486s of audio, or 32768 samples in the time domain. This is the number of samples fed into the VC module
# Us: With our tweaked hop_length above, this is 35280 samples, about 1.6s of audio. This may improve the model's tone/pitch/timbre/intonation? Just a little more audio to work from.
MAX_INFER_SAMPLES_VC = num_samples * hop_length

SEED = 1234  # numpy & torch PRNG seed
set_seed(SEED)

voiceDirectory = "studio_models\\targets"

# Hold the thread, model and devices.
inference_rt_thread = None
voice_conversion = None
devices = get_devices();

def setVoice(target_speaker, pauseLabel):
    global inference_rt_thread, voice_conversion, voiceDirectory

    if target_speaker is None or len(target_speaker) == 0:
        return [gr.update(interactive=False), "Invalid Voice selected, please pick another."]

    voiceFile = os.path.join(voiceDirectory, f"{target_speaker}.npy")
    if not os.path.isfile(voiceFile):
        return [gr.update(interactive=False), "Selected Voice not found, please pick another."]
        
    voice_conversion.set_target(target_speaker)
    
    if inference_rt_thread is not None and inference_rt_thread.is_alive():
        if pauseLabel == "Pause":
            return [gr.update(interactive=False), "Voice switched! Keep talking!"]
        else:
            return [gr.update(interactive=False), "Voice switched! Remember you are paused, your real voice is going through!"]
    else:
        return [gr.update(interactive=True), "Voice set! You can start now!"]        

def startGenerateVoice(input, output, latency):
    global inference_rt_thread, voice_conversion, devices, sample_rate, MAX_INFER_SAMPLES_VC
        
    inputDevice =  [p for p in devices['inputs'] if p.name == input];
    if inputDevice is None or len(inputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
        
    outputDevice =  [p for p in devices['outputs'] if p.name == output];
    if outputDevice is None or len(outputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
        
    if not voice_conversion.isTargetSet:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "A voice is not selected yet, conversion not started."]

    inference_rt_thread = InferenceRt(inputDevice[0].index, outputDevice[0].index, latency, sample_rate, MAX_INFER_SAMPLES_VC, voice_conversion, queue.Queue(), queue.Queue(), args=())
    inference_rt_thread.start()

    # Wait for start queue
    txt = inference_rt_thread.start_queue.get()
        
    return [gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True), txt]
    
def pauseGenerateVoice(pauseLabel):
    global inference_rt_thread
        
    label = "Unpause"
    text = "AI paused! Real voice is going through!"
    if pauseLabel != "Pause":
        label = "Pause"
        text = "AI unpaused! Keep talking!"
        
    if inference_rt_thread is not None and inference_rt_thread.is_alive():
        inference_rt_thread.status_queue.put("pauseToggle")        
    
    return label, text
    
def stopGenerateVoice():
    global inference_rt_thread
    
    if inference_rt_thread is not None and inference_rt_thread.is_alive():
        # Wait for end.        
        inference_rt_thread.status_queue.put("stop")
        inference_rt_thread.join()
    
    return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(value="Pause", interactive=False), gr.update(interactive=False), "Stopped!"]

def start():
    global inference_rt_thread, voice_conversion, devices, sample_rate, MAX_INFER_SAMPLES_VC, voiceDirectory

    # Create the model.
    print("We're loading the model and warming it up, please standby! Approximately 30-50 seconds!")
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
    voices = []
    for filename in os.listdir(voiceDirectory):
        f = os.path.join(voiceDirectory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            voices.append(filename[0:-4])

    # Retrieve available devices.
    inputNames = [p.name for p in devices['inputs']];
    outputNames = [p.name for p in devices['outputs']];

    with gr.Blocks() as demo:
        gr.Markdown("Select an input device, output device, adjust your input latency, and select a voice. Then press Start.  \nInput latency is how frequently audio will be gathered to send to the model. 400ms is the default. Below 200ms may produce a lot of stuttering.  \nOutput latency is determined by your GPU's performance. As soon as the model produces audio, it will be output to you.  \nTotal round trip will be the input latency + how long your GPU needs to convert audio.  \n  \nThe Voice drop down can be changed at any time without stopping!  \nThe Pause button will allow your normal voice through!  \nThe Stop button will stop audio entirely, and may take up to 5 seconds to complete.")
        with gr.Row():
            inputDrop = gr.Dropdown(choices=inputNames, label="Input Device");
            outputDrop = gr.Dropdown(choices=outputNames, label="Output Device");
            latencySlider = gr.Slider(50, 2000, label="Input latency (milliseconds)", step=50, value=400);
        with gr.Row():
            voiceDrop = gr.Dropdown(choices=voices, value="yara", label="Voice File");
            startButton = gr.Button(value="Start", interactive=True);
            pauseButton = gr.Button(value="Pause", interactive=False);
            stopButton = gr.Button(value="Stop", interactive=False);
        with gr.Row():
            text = gr.Textbox(label="Status");
            
        voiceDrop.input(fn=setVoice, inputs=[voiceDrop, pauseButton], outputs=[startButton, text])
        startButton.click(fn=startGenerateVoice, inputs=[inputDrop, outputDrop, latencySlider], outputs=[inputDrop, outputDrop, latencySlider, startButton, pauseButton, stopButton, text])
        pauseButton.click(fn=pauseGenerateVoice, inputs=pauseButton, outputs=[pauseButton, text])
        stopButton.click(fn=stopGenerateVoice, inputs=None, outputs=[inputDrop, outputDrop, latencySlider, startButton, pauseButton, stopButton, text])

    demo.launch()
    
if __name__ == '__main__':
    start()
