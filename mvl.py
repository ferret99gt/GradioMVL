import gradio as gr
import math
import numpy as np
import os
import queue
import random
import torch 
import pyaudio

from inference_rt import InferenceRt
from voice_conversion import ConversionPipeline

# venv\Scripts\pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# venv\Scripts\pip install gradio pyaudio librosa

# Samples, rates, etc.
input_sample_rate = 22050 # sampling rate yeah!
output_sample_rate = 24000 # the sampling rate for output. Do not change, matches what the model outputs.
num_samples = 128  # input spect shape num_mels * num_samples
hop_length = 256  # int(0.0125 * input_sample_rate)  # 12.5ms - in line with Tacotron 2 paper
hop_length = int(0.0125 * input_sample_rate)  # Let's actually try that math, in line with Tacotron 2 paper!

# MVL: corresponds to 1.486s of audio, or 32768 samples in the time domain. This is the number of samples fed into the VC module
# Us: With our tweaked hop_length above, this is 35280 samples, about 1.6s of audio. This may improve the model's tone/pitch/timbre/intonation? Just a little more audio to work from.
"""
IMPORTANT:

The model is ALWAYS given the MAX_INFER_SAMPLES_VC amount of audio. A lower latency doesn't reduce the data the model receives. A higher latency doesn't increase it. The input latency
controls how frequently the input stream is read. Each time it reads data, it puts it in a queue with the last x chunks worth of audio. The number of chunks depends on MAX_INFER_SAMPLES_VC
divided by the input latency. This entire queue of chunks, sized to MAX_INFER_SAMPLES_VC, is what gets sent to the model. The model infers based on thae last 1.6 seconds of audio, returning
a 1.6 second output wav. That output wav is then trimmed to the length of the input latency so only the "new" audio is output to you.

A lower latency does not reduce the model's quality. It increases how frequently the model is called (impacting GPU usage) as well as reducing the size of the output audio. More frequent
smaller chunks come back, leading to potential audio artifacts as the smaller chunks are streamed one by one. Each chunk is from a different model generation, so the output of the model
subtly shifts each time. Let's assume a 200ms input latency. When you speak for 2 seconds, what comes back is 10 individual outputs from the model, each generated separate from each other.
So there are 10 pieces of audio that are blended together, with 9 "seams" where the audio's character may change.

A high latency doesn't increase the quality of the model's output. What it does is reduce how often the model is called, lowering GPU usage, and producing longer output chunks. Because
the outputs are less frequent and longer, they will tend to have a more consistent sound. Let's assume a 1000ms input latency. When you speak for 2 seconds, what comes back is only 2 individual
outputs from the model. Now there's a single seam, a single place where the audio's character might change.
"""           
MAX_INFER_SAMPLES_VC = num_samples * hop_length
max_input_latency = int(num_samples * 0.0125 * 1000 - 1) # Max latency in ms allowed by MAX_INFER_SAMPLES_VC. Minus 1, so we're below the limit.
max_input_latency = max_input_latency - (max_input_latency%50) # Round down to nearly 50 for latency slider stepping. This should end up being 1550ms.

# Hardcode the seed.
SEED = 1234  # numpy & torch PRNG seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Where are the voice targets?
voiceDirectory = "studio_models\\targets"

# Hold the thread, model and devices.
inference_rt_thread = None
voice_conversion = None

# Collect audio devices from hostApi 0
# MVL used only hostApi 0 and it seems fine, but should look into this again later.
inputs = []
outputs = []

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device = p.get_device_info_by_index(i)
    #print(device)
    
    #defaultSampleRate
    if device["hostApi"] == 0:
        if device["maxInputChannels"]:
            inputs.append(device)
        if device["maxOutputChannels"]:
            outputs.append(device)

p.terminate()

devices = dict()
devices["inputs"] = inputs
devices["outputs"] = outputs

def setCrossfade(crossfade):
    global voice_conversion
    
    voice_conversion.set_crossfade(crossfade)

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

def startGenerateVoice(input, output, latency, crossfade):
    global inference_rt_thread, voice_conversion, devices, input_sample_rate, output_sample_rate, MAX_INFER_SAMPLES_VC
        
    inputDevice =  [p for p in devices['inputs'] if p["name"] == input];
    if inputDevice is None or len(inputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
        
    outputDevice =  [p for p in devices['outputs'] if p["name"] == output];
    if outputDevice is None or len(outputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid output device selected, conversion not started."]
        
    if not voice_conversion.isTargetSet:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "A voice is not selected yet, conversion not started."]

    voice_conversion.set_crossfade(crossfade)

    inference_rt_thread = InferenceRt(inputDevice[0]["index"], outputDevice[0]["index"], latency, input_sample_rate, output_sample_rate, MAX_INFER_SAMPLES_VC, voice_conversion, queue.Queue(), queue.Queue(), args=())
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
        inference_rt_thread.status_queue.put("stop")
        # Wait for end.
        inference_rt_thread.join()
    
    return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(value="Pause", interactive=False), gr.update(interactive=False), "Stopped!"]

def start():
    global inference_rt_thread, voice_conversion, devices, input_sample_rate, MAX_INFER_SAMPLES_VC, voiceDirectory, max_input_latency

    # Create the model.
    print("We're loading the model and warming it up, please standby! Approximately 30-50 seconds!")
    voice_conversion = ConversionPipeline(input_sample_rate)
    voice_conversion.set_target("yara")
    
    # warmup models into the cache
    warmup_iterations = 20
    warmup_frames_per_buffer = math.ceil(input_sample_rate * 400 / 1000) # Warm up with 400ms mock latency
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
    inputNames = [p["name"] for p in devices['inputs']];
    outputNames = [p["name"] for p in devices['outputs']];
    
    # cross faders
    crossFadeNames = ["linear", "constant power", "none"]

    with gr.Blocks() as demo:
        gr.Markdown("Select an input device, output device, adjust your input latency, and select a voice. Then press Start.  \nInput latency is how frequently audio will be gathered to send to the model. Below 200ms may produce a lot of stuttering.  \nOutput latency is determined by your GPU's performance. As soon as the model produces audio, it will be output to you.  \nTotal round trip will be the input latency + how long your GPU needs to convert audio.  \n  \nCrossfade can be switched at any time. MVL uses linear. Disabling entirely will introduce some \"popping\".  \nThe Voice drop down can be changed at any time without stopping!  \nThe Pause button will allow your normal voice through!  \nThe Stop button will stop audio entirely, and may take up to 5 seconds to complete.")
        with gr.Row():
            inputDrop = gr.Dropdown(choices=inputNames, label="Input Device");
            outputDrop = gr.Dropdown(choices=outputNames, label="Output Device");
            latencySlider = gr.Slider(50, max_input_latency, label="Input latency (milliseconds)", step=50, value=300);
            crossfadeDrop = gr.Dropdown(choices=crossFadeNames, label="Crossfade Method");
        with gr.Row():
            voiceDrop = gr.Dropdown(choices=voices, value="yara", label="Voice File");
            startButton = gr.Button(value="Start", interactive=True);
            pauseButton = gr.Button(value="Pause", interactive=False);
            stopButton = gr.Button(value="Stop", interactive=False);
        with gr.Row():
            text = gr.Textbox(label="Status");
            
        crossfadeDrop.input(fn=setCrossfade, inputs=crossfadeDrop, outputs=None)
        voiceDrop.input(fn=setVoice, inputs=[voiceDrop, pauseButton], outputs=[startButton, text])
        startButton.click(fn=startGenerateVoice, inputs=[inputDrop, outputDrop, latencySlider, crossfadeDrop], outputs=[inputDrop, outputDrop, latencySlider, startButton, pauseButton, stopButton, text])
        pauseButton.click(fn=pauseGenerateVoice, inputs=pauseButton, outputs=[pauseButton, text])
        stopButton.click(fn=stopGenerateVoice, inputs=None, outputs=[inputDrop, outputDrop, latencySlider, startButton, pauseButton, stopButton, text])

    demo.launch()
    
if __name__ == '__main__':
    start()
