import math
import numpy as np
import os
import queue
import random
import re
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import requests
import torch 
import pyaudio
import librosa

from inference_rt import InferenceRt
from voice_conversion import ConversionPipeline

# venv\Scripts\pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# venv\Scripts\pip install gradio==6.2.0 pyaudio librosa

# Samples, rates, etc.
model_48K = False
if(model_48K):
    input_sample_rate = 48000
    output_sample_rate = 48000
else:
    input_sample_rate = 24000
    output_sample_rate = 24000

MODEL_FRAME_MS = 20
BASE_CONTEXT_MS = 1600
FADE_DURATION_MS = 20
CAPTURE_CHUNKS_PER_INFERENCE = 10

# Each content-model frame represents 20ms. The rolling context defaults to 1.6 seconds,
# while input latency controls how often a newly generated fragment is emitted.
"""
IMPORTANT:

The model is ALWAYS given the MAX_INFER_SAMPLES_VC amount of audio. A lower latency doesn't reduce the data the model receives. A higher latency doesn't increase it. The input latency
controls how frequently the model will attempt to run, with whatever audio is available from the input stream is read. Each time it reads data, it tracks how many new chunks of audio it
received. This entire queue of chunks, sized to MAX_INFER_SAMPLES_VC, is what gets sent to the model. The model infers based on the last 1.6 seconds of audio, returning a 1.6 second output
wav. That output wav is then trimmed based on the number of new input chunks that were received, so only the "new" audio is output to you.

A lower latency increases how frequently the model is called (impacting GPU usage) and reduces the size of each output fragment. Below roughly 200ms, this model can become unstable even
when the GPU keeps up. More frequent, smaller chunks can also produce more stitching artifacts. Each chunk is from a different model generation, so the output of the model
subtly shifts each time. Let's assume a 200ms input latency. When you speak for 2 seconds, what comes back is 10 individual outputs from the model, each generated separate from each other.
So there are 10 pieces of audio that are blended together, with 9 "seams" where the audio's character may change.

A high latency doesn't increase the quality of the model's output. What it does is reduce how often the model is called, lowering GPU usage, and producing longer output chunks. Because
the outputs are less frequent and longer, they will tend to have a more consistent sound. Let's assume a 1000ms input latency. When you speak for 2 seconds, what comes back is only 2 individual
outputs from the model. Now there's a single seam, a single place where the audio's character might change.
"""           
MAX_INFER_SAMPLES_VC = input_sample_rate * BASE_CONTEXT_MS // 1000
max_input_latency = BASE_CONTEXT_MS - 100

# Hardcode the seed.
SEED = 1234  # numpy & torch PRNG seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Borrowed download method.
def downloadFile(file_path: str, destination_path: str):

    url = f"https://github.com/metavoicexyz/MetaVoiceLive/raw/main/ai/models/{file_path}?download="
    # Stream large model files to avoid holding gigabytes in memory.
    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            print("File download failed.")
            return

        with open(destination_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)

        print("File downloaded successfully.")

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
    if os.path.isfile(f) and filename.lower().endswith(".npy"):
        voices.append(filename[0:-4])
voices.sort()

speaker_encoder = None
speaker_encoder_lock = threading.Lock()
inference_rt_thread = None

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

def setFadeDuration(fade_duration_ms):
    global voice_conversion

    voice_conversion.set_fade_duration(fade_duration_ms)

def setNoiseScale(noise_scale):
    global voice_conversion

    voice_conversion.set_noise_scale(noise_scale)

def normalizeGradioAudio(audio):
    if audio is None:
        raise ValueError("No audio was supplied.")

    sample_rate, samples = audio
    samples = np.asarray(samples)
    if np.issubdtype(samples.dtype, np.integer):
        dtype_info = np.iinfo(samples.dtype)
        scale = max(abs(dtype_info.min), dtype_info.max)
        samples = samples.astype(np.float32) / scale
    else:
        samples = samples.astype(np.float32)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError("Audio must be mono or stereo.")

    if not np.isfinite(samples).all() or samples.size == 0:
        raise ValueError("Audio is empty or contains invalid samples.")
    return int(sample_rate), np.ascontiguousarray(samples)

def createTargetVoice(voice_name, audio_files, recorded_audio):
    global speaker_encoder, voices

    voice_name = (voice_name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", voice_name):
        return [gr.update(), gr.update(), "Use only letters, numbers, underscores, or hyphens in the voice name."]

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        prepared_clips = []
        skipped_clips = 0

        if audio_files is None:
            audio_files = []
        elif isinstance(audio_files, (str, os.PathLike)):
            audio_files = [audio_files]

        for audio_path in audio_files:
            prepared_clip = np.asarray(preprocess_wav(audio_path), dtype=np.float32)
            if (
                prepared_clip.size == 0
                or not np.isfinite(prepared_clip).all()
                or np.sqrt(np.mean(np.square(prepared_clip, dtype=np.float64))) < 1e-5
            ):
                skipped_clips += 1
                continue
            prepared_clips.append(prepared_clip)

        if recorded_audio is not None:
            sample_rate, samples = normalizeGradioAudio(recorded_audio)
            prepared_clip = np.asarray(
                preprocess_wav(samples, source_sr=sample_rate), dtype=np.float32
            )
            if (
                prepared_clip.size == 0
                or not np.isfinite(prepared_clip).all()
                or np.sqrt(np.mean(np.square(prepared_clip, dtype=np.float64))) < 1e-5
            ):
                skipped_clips += 1
            else:
                prepared_clips.append(prepared_clip)

        if not prepared_clips:
            raise ValueError("No usable speech was supplied.")

        prepared = np.concatenate(prepared_clips)
        if prepared.size < 8000:
            raise ValueError(
                "Reference clips need at least 0.5 seconds of detected speech in total."
            )

        with speaker_encoder_lock:
            if speaker_encoder is None:
                speaker_encoder = VoiceEncoder(device=str(voice_conversion.device))
            embedding = speaker_encoder.embed_utterance(prepared)

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (256,) or not np.isfinite(embedding).all():
            raise ValueError("Speaker encoder did not produce a valid 256-value target.")

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Speaker encoder produced an empty target.")
        embedding /= norm
        np.save(os.path.join(voiceDirectory, f"{voice_name}.npy"), embedding)

        voices = sorted(set(voices + [voice_name]))
        duration_seconds = prepared.size / 16000
        status = (
            f"Created local voice target '{voice_name}' from {len(prepared_clips)} "
            f"clip(s), totaling {duration_seconds:.1f} seconds of speech."
        )
        if skipped_clips:
            status += f" Skipped {skipped_clips} empty or silent clip(s)."
        return [
            gr.update(choices=voices, value=voice_name),
            gr.update(choices=voices, value=voice_name),
            status,
        ]
    except Exception as error:
        return [gr.update(), gr.update(), f"Voice target creation failed: {error}"]

def setVoice(target_speaker, pauseLabel):
    global inference_rt_thread, voice_conversion, voiceDirectory

    if target_speaker is None or len(target_speaker) == 0:
        return [gr.update(interactive=False), "Invalid Voice selected, please pick another."]

    voiceFile = os.path.join(voiceDirectory, f"{target_speaker}.npy")
    if not os.path.isfile(voiceFile):
        return [gr.update(interactive=False), "Selected Voice not found, please pick another."]
        
    voice_conversion.set_target(target_speaker)
    voice_conversion.reset_overlap()
    
    if inference_rt_thread is not None and inference_rt_thread.is_alive():
        if pauseLabel == "Pause":
            return [gr.update(interactive=False), "Voice switched! Keep talking!"]
        else:
            return [gr.update(interactive=False), "Voice switched! Remember you are paused, your real voice is going through!"]

    return [gr.update(interactive=True), "Voice set! You can start now!"]

def startGenerateVoice(input, output, latency, bufferSize, crossfade, fadeDuration, noiseScale):
    global inference_rt_thread, voice_conversion, devices, input_sample_rate, output_sample_rate, MAX_INFER_SAMPLES_VC
        
    inputDevice = [p for p in devices['inputs'] if p["name"] == input];
    if inputDevice is None or len(inputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
        
    outputDevice = [p for p in devices['outputs'] if p["name"] == output];
    if outputDevice is None or len(outputDevice) != 1:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Invalid output device selected, conversion not started."]
        
    if not voice_conversion.isTargetSet():
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "A voice is not selected yet, conversion not started."]

    latency = int(latency)
    if latency % MODEL_FRAME_MS != 0:
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "Input latency must be a multiple of 20ms."]

    voice_conversion.set_crossfade(crossfade)
    voice_conversion.set_fade_duration(fadeDuration)
    voice_conversion.set_noise_scale(noiseScale)

    context_samples = int(MAX_INFER_SAMPLES_VC * bufferSize)
    inference_rt_thread = InferenceRt(inputDevice[0]["index"], outputDevice[0]["index"], latency, input_sample_rate, output_sample_rate, context_samples, CAPTURE_CHUNKS_PER_INFERENCE, voice_conversion, queue.Queue(), queue.Queue(), args=())
    inference_rt_thread.start()

    # Wait for start queue
    txt = inference_rt_thread.start_queue.get()
    if txt.startswith("Failed:"):
        inference_rt_thread.join()
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), txt]
        
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
        return [None, "Invalid Voice selected, please pick another."]

    voiceFile = os.path.join(voiceDirectory, f"{target_speaker}.npy")
    if not os.path.isfile(voiceFile):
        return [None, "Selected Voice not found, please pick another."]
        
    voice_conversion.set_target(target_speaker)
    voice_conversion.reset_overlap()
    
    try:
        wavSampleRate, wav_src = normalizeGradioAudio(audioIn)
    except ValueError as error:
        return [None, str(error)]

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
    voice_conversion = ConversionPipeline(input_sample_rate, output_sample_rate, FADE_DURATION_MS)
    voice_conversion.set_target("yara")
    
    # warmup models into the cache
    warmup_iterations = 20
    warmup_frames_per_buffer = math.ceil(output_sample_rate * 400 / 1000)
    for _ in range(warmup_iterations):
        wav = np.random.rand(MAX_INFER_SAMPLES_VC * 5).astype(np.float32)
        voice_conversion.run(wav, warmup_frames_per_buffer)
    voice_conversion.reset_overlap()
    print("Model ready and warmed up!")
    
    # cross faders
    crossFadeNames = ["hann", "linear", "constant power", "none"]

    with gr.Blocks() as demo:
        live = gr.Tab("Live")
        studio = gr.Tab("Studio")
        create_voice = gr.Tab("Create Voice")
        
        with live:
            gr.Markdown("Select input and output devices, tune the conversion interval and context buffer, select a voice, then press Start.  \n\nInput latency controls how often inference runs. Values are aligned to the model's 20ms audio frames. Very short intervals can degrade this model even when the GPU is fast enough.  \n\nBuffer size controls how much rolling context is sent to the model, starting at 1.6 seconds. Larger buffers consume more GPU memory and compute.  \n\nCrossfade method, fade duration, target voice, and latent variation can be changed while running. Longer fades hold the corresponding amount of overlap audio. A latent variation of zero is deterministic after warm-up.  \n\nPause passes your unmodified voice through. Stop ends audio processing.")
            with gr.Row():
                inputDrop = gr.Dropdown(choices=inputNames, label="Input Device");
                outputDrop = gr.Dropdown(choices=outputNames, label="Output Device");
                latencySlider = gr.Slider(40, max_input_latency, label="Input latency (milliseconds)", step=MODEL_FRAME_MS, value=200);
            with gr.Row():
                crossfadeDrop = gr.Dropdown(choices=crossFadeNames, value="hann", label="Crossfade Method");
                voiceDrop = gr.Dropdown(choices=voices, value="yara", label="Voice File");
                bufferSizeSlider = gr.Slider(1, 10, label="Buffer Size", step=1, value=5);
            with gr.Row():
                fadeDurationSlider = gr.Slider(20, 60, label="Crossfade duration (milliseconds)", step=MODEL_FRAME_MS, value=FADE_DURATION_MS);
                noiseScaleSlider = gr.Slider(0, 1, label="Latent variation", step=0.05, value=0);
            with gr.Row():                
                startButton = gr.Button(value="Start", interactive=True);
                pauseButton = gr.Button(value="Pause", interactive=False);
                stopButton = gr.Button(value="Stop", interactive=False);
            with gr.Row():
                text = gr.Textbox(label="Status");
                
            crossfadeDrop.input(fn=setCrossfade, inputs=crossfadeDrop, outputs=None)
            fadeDurationSlider.input(fn=setFadeDuration, inputs=fadeDurationSlider, outputs=None)
            noiseScaleSlider.input(fn=setNoiseScale, inputs=noiseScaleSlider, outputs=None)
            voiceDrop.change(fn=setVoice, inputs=[voiceDrop, pauseButton], outputs=[startButton, text])
            startButton.click(fn=startGenerateVoice, inputs=[inputDrop, outputDrop, latencySlider, bufferSizeSlider, crossfadeDrop, fadeDurationSlider, noiseScaleSlider], outputs=[inputDrop, outputDrop, latencySlider, bufferSizeSlider, startButton, pauseButton, stopButton, text])
            pauseButton.click(fn=pauseGenerateVoice, inputs=pauseButton, outputs=[pauseButton, text])
            stopButton.click(fn=stopGenerateVoice, inputs=[], outputs=[inputDrop, outputDrop, latencySlider, bufferSizeSlider, startButton, pauseButton, stopButton, text])
        with studio:
            gr.Markdown("Upload some audio! Pick a voice! Convert!")
            with gr.Row():
                audioIn = gr.Audio(type="numpy", format="wav", interactive=True, label="Upload audio!")
                audioOut = gr.Audio(type="numpy", format="wav", interactive=False, buttons=["download"], label="Download result!")
            with gr.Row():
                studioVoiceDrop = gr.Dropdown(choices=voices, interactive=True, value="yara", label="Voice File");
                studioStartButton = gr.Button(value="Convert", interactive=True);
            with gr.Row():
                text = gr.Textbox(label="Status");

            studioStartButton.click(fn=startStudioVoice, inputs=[studioVoiceDrop, audioIn], outputs=[audioOut, text])

        with create_voice:
            gr.Markdown("Create a local voice target from clean reference speech. Drag in one or more short clips, optionally record another sample, then create the target. Longer, varied speech generally gives the speaker encoder more useful material. The recordings and resulting target remain on this computer.")
            with gr.Row():
                targetName = gr.Textbox(label="Voice name", placeholder="my_voice")
                targetFiles = gr.File(file_count="multiple", file_types=["audio"], type="filepath", interactive=True, label="Reference audio clips")
            with gr.Row():
                targetRecording = gr.Audio(type="numpy", format="wav", sources=["microphone"], interactive=True, label="Optional microphone recording")
            with gr.Row():
                createTargetButton = gr.Button(value="Create Voice Target", interactive=True)
            with gr.Row():
                targetStatus = gr.Textbox(label="Status")

            createTargetButton.click(fn=createTargetVoice, inputs=[targetName, targetFiles, targetRecording], outputs=[voiceDrop, studioVoiceDrop, targetStatus])

    demo.launch()
    
if __name__ == '__main__':
    start()
