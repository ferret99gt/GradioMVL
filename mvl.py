import gradio as gr
import math
import numpy as np
import os
import queue
import random
import requests
import torch 
import pyaudio
import librosa

from inference_rt import InferenceRt
from voice_conversion import ConversionPipeline

# venv\Scripts\pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# venv\Scripts\pip install gradio pyaudio librosa

# Samples, rates, etc.
model_48K = False
if(model_48K):
    input_sample_rate = 48000 # sampling rate yeah!
    output_sample_rate = 48000 # the sampling rate for output. Do not change, matches what the model outputs.
else:
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
controls how frequently the model will attempt to run, with whatever audio is available from the input stream is read. Each time it reads data, it tracks how many new chunks of audio it
received. This entire queue of chunks, sized to MAX_INFER_SAMPLES_VC, is what gets sent to the model. The model infers based on the last 1.6 seconds of audio, returning a 1.6 second output
wav. That output wav is then trimmed based on the number of new input chunks that were received, so only the "new" audio is output to you.

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
max_input_latency = max_input_latency - (max_input_latency%50) # Round down to nearest 50 for latency slider stepping. This should end up being 1550ms.

# Hardcode the seed.
SEED = 1234  # numpy & torch PRNG seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Borrowed download method.
def downloadFile(file_path: str, destination_path: str):

    url = f"https://github.com/metavoicexyz/MetaVoiceLive/raw/main/ai/models/{file_path}?download="
    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the file content to the local destination path
        with open(destination_path, "wb") as file:
            file.write(response.content)

        print("File downloaded successfully.")
    else:
        print("File download failed.")

# Where are the voice targets?
modelDirectory = "studio_models"
voiceDirectory = modelDirectory + "\\targets"

# Do we have the models?
if not os.path.exists(modelDirectory):
    os.mkdir(modelDirectory)
    
if not os.path.exists(voiceDirectory):
    os.mkdir(voiceDirectory)
    
# main model?
if not os.path.exists(modelDirectory + "\\model.pt"):
    print("Downloading model.pt from MetaVoiceLive Github, about 154megs. Please wait a moment.")
    downloadFile("model.pt", modelDirectory + "\\model.pt")
    
# preprocess model?
if not os.path.exists(modelDirectory + "\\b_model.pt"):
    print("Downloading b_model.pt from MetaVoiceLive Github, about 1.2gigs. This will take a while but will only happen once.")
    downloadFile("b_model.pt", modelDirectory + "\\b_model.pt")

# default voices
if not os.path.exists(voiceDirectory + "\\blake.npy"):
    print("Downloading blake voice from MetaVoiceLive Github.")
    downloadFile("/targets/blake.npy", voiceDirectory + "\\blake.npy")
    
if not os.path.exists(voiceDirectory + "\\eva.npy"):
    print("Downloading eva voice from MetaVoiceLive Github.")
    downloadFile("/targets/eva.npy", voiceDirectory + "\\eva.npy")

if not os.path.exists(voiceDirectory + "\\scarlett.npy"):
    print("Downloading scarlett voice from MetaVoiceLive Github.")
    downloadFile("/targets/scarlett.npy", voiceDirectory + "\\scarlett.npy")
    
if not os.path.exists(voiceDirectory + "\\yara.npy"):
    print("Downloading yara voice from MetaVoiceLive Github.")
    downloadFile("/targets/yara.npy", voiceDirectory + "\\yara.npy")

if not os.path.exists(voiceDirectory + "\\zeus.npy"):
    print("Downloading zeus voice from MetaVoiceLive Github.")
    downloadFile("/targets/zeus.npy", voiceDirectory + "\\zeus.npy")

# Retrieve available voices.
voices = []
for filename in os.listdir(voiceDirectory):
    f = os.path.join(voiceDirectory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        voices.append(filename[0:-4])

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

# Retrieve available devices.
inputNames = [p["name"] for p in devices['inputs']];
outputNames = [p["name"] for p in devices['outputs']];

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

def startGenerateVoice(input, output, latency, bufferSize, crossfade):
    global inference_rt_thread, voice_conversion, devices, input_sample_rate, output_sample_rate, MAX_INFER_SAMPLES_VC
        
    inputDevice = [p for p in devices['inputs'] if p["name"] == input];
    if inputDevice is None or len(inputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
        
    outputDevice = [p for p in devices['outputs'] if p["name"] == output];
    if outputDevice is None or len(outputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid output device selected, conversion not started."]
        
    if not voice_conversion.isTargetSet:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "A voice is not selected yet, conversion not started."]

    voice_conversion.set_crossfade(crossfade)

    inference_rt_thread = InferenceRt(inputDevice[0]["index"], outputDevice[0]["index"], latency, input_sample_rate, output_sample_rate, MAX_INFER_SAMPLES_VC * bufferSize, voice_conversion, queue.Queue(), queue.Queue(), args=())
    inference_rt_thread.start()

    # Wait for start queue
    txt = inference_rt_thread.start_queue.get()
        
    return [gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True), txt]
    
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
    
    return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(value="Pause", interactive=False), gr.update(interactive=False), "Stopped!"]

def startStudioVoice(target_speaker, audioIn):
    global voice_conversion, input_sample_rate, output_sample_rate

    if target_speaker is None or len(target_speaker) == 0:
        return ["none", "Invalid Voice selected, please pick another."]

    voiceFile = os.path.join(voiceDirectory, f"{target_speaker}.npy")
    if not os.path.isfile(voiceFile):
        return ["none", "Selected Voice not found, please pick another."]
        
    voice_conversion.set_target(target_speaker)
    voice_conversion.set_crossfade("none")
    
    # Direct numpy from Gradio. It's int16, so we need to convert to float32, resample if needed, then run.
    wavSampleRate = audioIn[0]
    wav_src = audioIn[1]
    wav_src = wav_src.astype(np.float32)
    if wavSampleRate != input_sample_rate:
        wav_src = librosa.resample(wav_src, orig_sr=wavSampleRate, target_sr=input_sample_rate, res_type="soxr_vhq")
        
    wav_bytes_out = voice_conversion.run(wav_src, 0)

    # Let Gradio handle it. Float32 will be converted to Int16 automatically.
    output = (output_sample_rate, wav_bytes_out)
    
    return [output, f"Your file is ready! You can play or download it above."]

def start():
    global inference_rt_thread, voice_conversion, devices, input_sample_rate, output_sample_rate, MAX_INFER_SAMPLES_VC, voiceDirectory, max_input_latency

    # Create the model.
    print("We're loading the model and warming it up, please standby! Approximately 30-50 seconds!")
    voice_conversion = ConversionPipeline(output_sample_rate)
    voice_conversion.set_target("yara")
    
    # warmup models into the cache
    warmup_iterations = 20
    warmup_frames_per_buffer = math.ceil(input_sample_rate * 400 / 1000) # Warm up with 1000ms mock latency
    for _ in range(warmup_iterations):
        wav = np.random.rand(MAX_INFER_SAMPLES_VC * 5).astype(np.float32)
        voice_conversion.run(wav, warmup_frames_per_buffer)
    print("Model ready and warmed up!")
    
    # cross faders
    crossFadeNames = ["linear", "constant power", "none"]

    with gr.Blocks() as demo:
        live = gr.Tab("Live")
        studio = gr.Tab("Studio")
        
        with live:
            gr.Markdown("Select an input device, output device, adjust your input latency, buffer size, and select a voice. Then press Start.  \n\nInput latency is how frequently the model will run. Below 100ms may produce a lot of stuttering if your GPU cannot convert audio fast enough. Watch the console and raise the latency if model is taking too long. Even if your GPU can keep up, higher latencies may be \"smoother\".  \n\nBuffer size is how much audio will be used as input. The minimum buffer size is about 1.6 seconds. Increasing this can improve smoothness, but will also use more GPU. As with latency, adjust to your GPU's performance. You may need to start, stop, and start again after changing it.  \n\nCrossfade can be switched at any time. MVL uses linear. Disabling entirely will introduce some \"popping\".  \nThe Voice drop down can be changed at any time without stopping!  \nThe Pause button will allow your normal voice through!  \nThe Stop button will stop audio entirely, and may take up to 3 seconds to complete.")
            with gr.Row():
                inputDrop = gr.Dropdown(choices=inputNames, label="Input Device");
                outputDrop = gr.Dropdown(choices=outputNames, label="Output Device");
                latencySlider = gr.Slider(50, max_input_latency, label="Input latency (milliseconds)", step=25, value=500);
            with gr.Row():
                crossfadeDrop = gr.Dropdown(choices=crossFadeNames, value="linear", label="Crossfade Method");
                voiceDrop = gr.Dropdown(choices=voices, value="yara", label="Voice File");
                bufferSizeSlider = gr.Slider(1, 10, label="Buffer Size", step=1, value=5);
            with gr.Row():                
                startButton = gr.Button(value="Start", interactive=True);
                pauseButton = gr.Button(value="Pause", interactive=False);
                stopButton = gr.Button(value="Stop", interactive=False);
            with gr.Row():
                text = gr.Textbox(label="Status");
                
            crossfadeDrop.input(fn=setCrossfade, inputs=crossfadeDrop, outputs=None)
            voiceDrop.input(fn=setVoice, inputs=[voiceDrop, pauseButton], outputs=[startButton, text])
            startButton.click(fn=startGenerateVoice, inputs=[inputDrop, outputDrop, latencySlider, bufferSizeSlider, crossfadeDrop], outputs=[inputDrop, outputDrop, latencySlider, bufferSizeSlider, startButton, pauseButton, stopButton, text])
            pauseButton.click(fn=pauseGenerateVoice, inputs=pauseButton, outputs=[pauseButton, text])
            stopButton.click(fn=stopGenerateVoice, inputs=[], outputs=[inputDrop, outputDrop, latencySlider, bufferSizeSlider, startButton, pauseButton, stopButton, text])
        with studio:
            gr.Markdown("Upload some audio! Pick a voice! Convert!")
            with gr.Row():
                audioIn = gr.Audio(type="numpy", format="wav", interactive=True, label="Upload audio!")
                audioOut = gr.Audio(type="numpy", format="wav", interactive=False, show_download_button=True, label="Download result!")
            with gr.Row():
                studioVoiceDrop = gr.Dropdown(choices=voices, interactive=True, value="yara", label="Voice File");
                studioStartButton = gr.Button(value="Convert", interactive=True);
            with gr.Row():
                text = gr.Textbox(label="Status");

            studioStartButton.click(fn=startStudioVoice, inputs=[studioVoiceDrop, audioIn], outputs=[audioOut, text])

    demo.launch()
    
if __name__ == '__main__':
    start()
