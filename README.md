# Gradio MetaVoiceLive
This is a reworking of MetaVoiceLive to completely strip out the Electron layer and JavaScript reliance. It's a pure Python implementation. All unnecessary code removed.

## Changelog

 - Initial release on 8/5/2023

## Setup

 - Install Python 3.10.x and make use it is added to PATH.
 - Clone this repo.
 - Get the MVL models. Retrieve a copy of the MVML 1.4 download or install the normal MVL to get it. Navigate to "resources\app\dist\metavoice\ai" and copy the entire studio_models folder to your new repo. Such as you have "GradioMVL\studio_models" with the two .pt files, and "GradioMVL\studio_models\targets" with the .npy files.
 - Open a Command window, navigate to the repo, and run launch.bat once to create the venv. It will fail.
 - Run: venv\Scripts\pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 - Run: venv\Scripts\pip install gradio sounddevice pyaudio librosa
 - Run launch.bat again. It will tell you when Gradio is ready to open in your web browser. At this point you can double click launch.bat whenever you need to Gradio.
 
## Use

 - Pick your input and output devices.
 - Adjust the latency slider according to your system hardware. I have been able to test 200ms on a 2070S. Note this is the one way latency. So it's 200ms in, 200ms back, for 400ms total.
 - Pick your voice!
 - Press Start and wait. The model has a built in warm-up process that takes approximately 30-45 seconds to complete.