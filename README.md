# Gradio MetaVoiceLive
This is a reworking of MetaVoiceLive to completely strip out the Electron layer and JavaScript reliance. It's a pure Python implementation. All unnecessary code removed.

MVL: https://github.com/metavoicexyz/MetaVoiceLive
MetaVoice Studio: https://studio.themetavoice.xyz/

MVL is based in part on the following projects, or related projects/papers:
 - https://github.com/ebadawy/voice_conversion/
 - https://github.com/CorentiJ/Real-Time-Voice-Conversion/
 - https://github.com/leimao/Voice-Converter-CycleGAN/ 

I also borrowed the launch/setup code from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/

## Changelog

 - 2023-08-05:
   - Initial release
 - 2023-08-06:
   - Automatic install of dependencies
 - 2023-08-07:
   - Implement split audio input/output to improve latency. ~40% latency reduction.
   - Add hot-swappable voice function.
   - Add real voice passthrough/pause function.
   - Remove torchvision dependency, not needed.
   - Remove sounddevice dependency, not needed, as PyAudio can do the same thing.
   - Minor performance cleanups.
 - 2023-08-08:
   - Match output sampling rate to model sampling rate. No more down sampling, slight performance saving, potentially slight quality improvement.
   - Add ability to disable crossfade if desired. May introduce some "popping".
   - Output model performance to console.
   - Adjust default input latency to 300ms for now.
   - Set max input latency based on size of input buffer to model. Can't be bigger!
   - Added constant power crossfade option.
 - 2023-09-25:
   - Reworked input/conversion/output threads again to further improve latency. A tad unsure on this, may revert in future.
   - Added "zeroed" default voice, provided on Discord by Quack.
   - Added automatic download of MetaVoiceLive models. No more getting MVML 1.4 manually and copying it over.
 - 2023-12-20:
   - Due to the second week long outage of the Studio web site, I added a new tab allowing you to choose and convert an audio file, similar to Studio.
   - MVL's model is not nearly as good as Studio's, so don't expect miracles.
 - 2024-07-08:
   - Added optional support for the 48kHz MVL decoder while keeping the 24kHz model as the default.
   - Added the Buffer Size control, allowing longer rolling context than the model's original 1.6-second minimum at the cost of additional inference work.
   - Updated real-time chunk sizing and crossfade handling for variable context lengths.
 - 2024-07-11:
   - Pinned NumPy to 1.26.4 because NumPy 2.x was incompatible with the Torch build in use.
 - 2025-01-20:
   - Fixed voice selection before the live inference thread had been created.
 - 2025-12-24:
   - Added the Hann crossfade option and reset overlap state when switching voices.
   - Restored the 20ms default fade duration and updated Gradio's audio download-button API.
   - Streamed large model downloads to disk and added early input-buffer overrun protection.
   - Confirmed Python 3.12.10 support and updated the launcher from CUDA 11.8 to the CUDA 12.8 Torch package index.
   - Cleaned up PyAudio type annotations and several startup/runtime checks.
 - 2026-07-18:
   - Corrected the live input rate to 24kHz so capture time matches the 24kHz decoder output.
   - Aligned inference intervals to the model's 20ms content frames.
   - Added deterministic latent inference by default, with adjustable latent variation.
   - Bounded all real-time queues and made overload handling discard stale audio instead of increasing latency.
   - Added local target-voice creation from reference speech using Resemblyzer embeddings.
   - Added adjustable 20/40/60ms crossfades and safe overlap resets when settings change.
   - Upgraded to Gradio 6.20 and added multi-file reference uploads for local target creation.
   - Made Hann the default crossfade and fixed programmatic voice selections so newly created targets load immediately.
   - Added automatic Studio stereo-to-mono conversion and direct source-rate conversion to the model's 16kHz content encoder.
   - Explicitly exported Studio results as normalized 16-bit PCM at the selected decoder's 24kHz or 48kHz output rate.
   - Paced live inference from complete microphone-frame batches instead of an independent wall-clock timer.

## Setup

 - Install Python 3.10 or newer and make sure it is added to PATH.
 - Clone this repo.
 - Either open a command window and navgiate to the repo, or find it in Explorer. Run launch.bat! It will install all dependencies and then start!
   - The first time launch will take a bit as it has to download and install Torch, a 2+ gig install.
   - It will also download the models from the MetaVoiceLive github repo. The preprocessor model is about 1.2 gig so may take a moment as well.
   - GradioMVL makes a Python VENV, so everything is self-contained. You can delete the repo entirely and it'll all be cleaned up.
 - GradioMVL will load the model. There is a warm-up process that takes approximately 30-45 seconds to complete.
 - When you're told to open localhost:7860 you're ready to go! Just open it in your web browser.
 
## Use

 - Live:
   - Pick your input and output devices.
   - Adjust the input latency slider according to your system hardware.
     - Input latency is how much new microphone audio is gathered before each model run. 200ms is the default, and values use 20ms steps to match the model's frame size.
     - Output latency is determined by your GPU's performance. As soon as the model produces audio, it will be output to you.
     - Total round trip will be the input latency + how long your GPU needs to convert audio. The conversion timing is output to the console.
     - Testing on a RTX 2070S showed an average model response time of 60-80ms. There may be periodic spikes.
     - Generally recommend that you set the input latency to be double your GPU's response time or more. This is to ensure the input never overruns the model's conversion.
     - The lower the input latency, the more frequently the model is called and the more seams it must stitch. This model can become unstable below roughly 200ms even when the GPU keeps up.
   - Select a crossfade type and a 20, 40, or 60ms fade duration. Hann is the default. These can be changed while converting; longer fades hold the corresponding amount of overlap audio.
   - Leave latent variation at zero for repeatable inference, or increase it to restore the model's original random latent sampling.
   - Pick your voice! yara is used as a default. The voice can be changed at any time, including while already converting.
   - Press Start and go!
   - Press Pause to pause the AI conversion and send your real voice.
   - Press Stop to shutdown the audio completely. This is required to change the input/output devices or the input latency.
 - Studio:
   - Choose an audio file!
   - Stereo audio is automatically mixed down to mono, and source sample rates are converted internally for the model.
   - Choose a voice!
   - Press convert!
   - Wait! The first time a file is processed will take longer. After the first time, changing voices will convert again much faster.
   - Listen!
 - Create Voice:
   - Drag in one or more clean reference clips, optionally record another sample, and enter a local voice name.
   - Press Create Voice Target. The resulting 256-value target is saved under `studio_models/targets` and immediately appears in both voice lists.
   - Usable speech from all supplied clips is combined before the target embedding is generated.
   - Reference audio and generated targets stay local.
   - Experimental, results may not be great.
 
## Advanced

Target files are 256-value Resemblyzer speaker embeddings saved as NumPy arrays. Use the Create Voice tab, or place a compatible `.npy` file in `studio_models/targets` and restart.

## Todo

 - Need to add a launch.sh for Linux.
