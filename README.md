# Gradio MetaVoiceLive
This is a reworking of MetaVoiceLive to completely strip out the Electron layer and JavaScript reliance. It's a pure Python implementation. All unnecessary code removed.

## Changelog

 - 2023-08-05: Initial release
 - 2023-08-06: Automatic install of dependencies
 - 2023-08-07: Implement split audio input/output to improve latency. Add hot-swappable voice function. Add real voice passthrough/pause function. Minor performance cleanups.

## Setup

 - Install Python 3.10.x and make use it is added to PATH.
 - Clone this repo.
 - Get the MVL models. Retrieve a copy of the MVML 1.4 download or install the normal MVL to get it. Navigate to "resources\app\dist\metavoice\ai" and copy the entire studio_models folder to your new repo.
   - You should have "GradioMVL\studio_models" with the two .pt files, and "GradioMVL\studio_models\targets" with the .npy files.
   - This is not optional, you must have these files.
 - Either open a command window and navgiate to the repo, or find it in Explorer. Run launch.bat! It will install all dependencies and then start!
   - The first time launch will take a bit as it has to download and install Torch, a 2+ gig install.
   - GradioMVL makes a Python VENV, so everything is self-contained. You can delete the repo entirely and it'll all be cleaned up.
 - GradioMVL will load the model. There is a warm-up process that takes approximately 30-45 seconds to complete.
 - When you're told to open localhost:7860 you're ready to go! Just open it in your web browser.
 
## Use

 - Pick your input and output devices.
 - Adjust the latency slider according to your system hardware.
   - I have been able to test very low even on a 2070S. But I generally don't recommend lower than 300, and find 350-400 to be the sweet spot.
   - Note this is the one way latency. So it's 400ms in, 400ms back, for 800ms total.
 - Pick your voice! yara is used as a default.
 - Press Start and go!
 
## Advanced

Do you know how to make your own npy target voices? If so, just drop them in "GradioMVL\studio_models\targets" and restart!

### Todo

 - Need to add a launch.sh for Linux.