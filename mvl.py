import argparse
import gradio as gr
import multiprocessing
import os
import time

from typing import Optional
from util.data_types import DeviceMap
from util.portaudio_utils import get_devices
from inference_rt import run_inference_rt, InferencePipelineMode

# venv\Scripts\pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# venv\Scripts\pip install gradio sounddevice pyaudio librosa

convert_process: Optional[multiprocessing.Process] = None
stop_pipeline: Optional[multiprocessing.Value] = None
has_pipeline_started: Optional[multiprocessing.Value] = None
callback_latency_ms: Optional[multiprocessing.Value] = None

def convert_process_target(
        stop_pipeline: multiprocessing.Value,
        has_pipeline_started: multiprocessing.Value,
        input_device_idx: int,
        output_device_idx: int,
        callback_latency_ms: multiprocessing.Value,
        target_speaker: str,
    ):

    opt = argparse.Namespace(
        mode=InferencePipelineMode.online_crossfade,
        input_device_idx=input_device_idx,
        output_device_idx=output_device_idx,
        callback_latency_ms=callback_latency_ms,
        target_speaker=target_speaker,
    )

    run_inference_rt(
        opt,
        stop_pipeline=stop_pipeline,
        has_pipeline_started=has_pipeline_started,
    )

if __name__ == '__main__':
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

    def generateVoice(input, output, target_speaker, latency):
        global convert_process, stop_pipeline
    
        inputDevice =  [p for p in devices['inputs'] if p.name == input];
        
        if inputDevice is None or len(inputDevice) != 1:
            return [gr.update(interactive=True), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
            
        outputDevice =  [p for p in devices['outputs'] if p.name == output];
        if outputDevice is None or len(outputDevice) != 1:
            return [gr.update(interactive=True), gr.update(interactive=False), "Invalid input device selected, conversion not started."]
            
        if target_speaker is None or len(target_speaker) == 0:
            return [gr.update(interactive=True), gr.update(interactive=False), "Invalid Voice selected, conversion not started."]

        voiceFile = os.path.join(voiceDirectory, f"{target_speaker}.npy")
        if not os.path.isfile(voiceFile):
            return [gr.update(interactive=True), gr.update(interactive=False), "Selected Voice not found, conversion not started."]
            
        stop_pipeline = multiprocessing.Value("i", 0)
        has_pipeline_started = multiprocessing.Value("i", 0)
        
        input_device_idx = inputDevice[0].index
        output_device_idx = outputDevice[0].index
        callback_latency_ms = multiprocessing.Value("I", latency)
        
        convert_process = multiprocessing.Process(
            target=convert_process_target,
            args=(
                stop_pipeline,
                has_pipeline_started,
                input_device_idx,
                output_device_idx,
                callback_latency_ms,
                target_speaker,
            ),
        )
        convert_process.start()      

        while not has_pipeline_started.value:
            time.sleep(0.2)
            
        return [gr.update(interactive=False), gr.update(interactive=True), "Started! Get to talking!"]
        
    def stopGenerateVoice():
        global convert_process, stop_pipeline
        
        with stop_pipeline.get_lock():
            stop_pipeline.value = 1

        convert_process.join(5)
        if convert_process.is_alive():
            convert_process.terminate()
        
        return [gr.update(interactive=True), gr.update(interactive=False), "Stopped!"]    

    with gr.Blocks() as demo:
        gr.Markdown("Select an input and output devices, then press Start.")
        with gr.Row():
            inputDrop = gr.Dropdown(choices=inputNames, label="Input Device");
            outputDrop = gr.Dropdown(choices=outputNames, label="Output Device");
        with gr.Row():
            voiceDrop = gr.Dropdown(choices=voices, label="Voice File");
            latencySlider = gr.Slider(100, 2000, label="Latency (one way)", step=50, value=300);
        with gr.Row():
            startButton = gr.Button("Start")
            stopButton = gr.Button(value="Stop", interactive=False)
        with gr.Row():
            text = gr.Textbox(label="Status")
            
        startButton.click(fn=generateVoice, inputs=[inputDrop, outputDrop, voiceDrop, latencySlider], outputs=[startButton, stopButton, text])
        stopButton.click(fn=stopGenerateVoice, inputs=None, outputs=[startButton, stopButton, text])

    demo.launch()