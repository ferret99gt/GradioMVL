# Gradio MetaVoiceLive
This is a reworking of [MetaVoice Live](https://github.com/metavoiceio/MetaVoiceLive) that removes the Electron layer and JavaScript reliance. It is a local, pure-Python implementation with no MetaVoice sign-in, voice-data upload, or telemetry path. The application itself uses network access only to download missing model files from the original public repository.

> [!IMPORTANT]
> **Project status:** GradioMVL is effectively frozen. Basic bug fixes and dependency maintenance may continue, but no further model or conversion-quality advancement is expected.

## Project Research
GPT-5.6 Sol assisted in recovering the provenance and ancestry of MetaVoice Live. The following sections document those findings, cross-checked against the original source, serialized models, file hashes, and identified upstream projects:

MetaVoice published the inference wrapper and TorchScript model files, but did not document the model architecture, training history, or third-party model lineage. The sections below preserve what can now be established from the original `/ai/` source, the serialized model graphs, public upstream code, and file hashes. They deliberately distinguish direct evidence from inference.

### Recovered Architecture

The distributed model is a chunked, zero-shot voice-conversion pipeline:

`mono audio -> 16kHz -> b_model.pt -> 1024-channel content frames -> model.pt/model_48k.pt + 256-value speaker target -> 24kHz/48kHz audio`

 - **Content encoder:** `b_model.pt` is serialized as `ai_studio_tracing.b_models.Preprocessor`. Its seven convolutional feature-extractor layers, 1024-wide projection, 24 transformer layers, 16 attention heads, 320 relative-position buckets, and WavLM-specific relative-attention modules match [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm). It emits one 1024-value content frame per 20ms of 16kHz input. The graph establishes the architecture, but not whether MetaVoice used untouched or fine-tuned WavLM weights.
 - **Voice decoder:** `model.pt` and `model_48k.pt` are serialized `SynthesizerTrn` models with `enc_p`, `enc_q`, `flow`, and `dec` modules. Their class names, graph structure, 1024-channel content input, 192-channel latent space, and 256-channel speaker conditioning closely match [FreeVC](https://github.com/OlaWod/FreeVC), whose decoder is derived from [VITS](https://github.com/jaywalnut310/vits) and [HiFi-GAN](https://github.com/jik876/hifi-gan). The 24kHz decoder upsamples each 20ms content frame by `10 * 6 * 4 * 2 = 480` samples, consistent with FreeVC's [24kHz conversion guidance](https://github.com/OlaWod/FreeVC/tree/main/tips-for-synthesizing-24KHz-wavs-from-16kHz-wavs). The 48kHz variant changes the final factor to four for `960` samples per frame.
 - **Speaker targets:** the supplied nonzero `.npy` files are 256-value `float32`, nonnegative, L2-normalized vectors accepted directly by the FreeVC-style speaker-conditioning input. A speaker-encoder checkpoint later recommended by MetaVoice developers is byte-for-byte identical to FreeVC's published [`speaker_encoder/ckpt/pretrained_bak_5805000.pt`](https://github.com/OlaWod/FreeVC/blob/main/speaker_encoder/ckpt/pretrained_bak_5805000.pt) (SHA-256 `bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca`). This strongly establishes FreeVC speaker-embedding compatibility, but does not prove how MetaVoice created each supplied target. GradioMVL's Create Voice feature uses Resemblyzer's same-sized but different default checkpoint, so its output is structurally compatible rather than a proven recreation of MetaVoice's embedding process.

No training source, training configuration, dataset record, or pre-export checkpoints were published with MetaVoice Live. The artifacts therefore do **not** establish whether MetaVoice retrained or fine-tuned these models, changed only the decoder configuration, or repackaged existing weights. No distinct MetaVoice-authored model architecture was found in the serialized graphs; the defensible description is a WavLM-Large-to-FreeVC/VITS-derived system packaged for MetaVoice Live.

This also defines the technical limit of the application: WavLM and the exported decoder are noncausal, offline models with no recurrent state or streaming cache exposed between calls. GradioMVL supplies rolling history, repeatedly runs the complete model, keeps the newest output segment, and crossfades adjacent generations. Queueing, pacing, and overlap improvements can make that wrapper more reliable, but cannot turn the checkpoint into true stateful streaming inference.

### Recovered Code Provenance

Model lineage and Python-code lineage are separate questions. A direct comparison of the unpacked MetaVoice Live `/ai/` tree found:

 - Original MVL's `ai/spectrogram_conversion/params.py` directly preserves distinctive values and comments from [Ehab A. AlBadawy and Siwei Lyu's `voice_conversion`](https://github.com/ebadawy/voice_conversion), including `num_samples = 128`, the `num_mels * num_samples` shape comment, and the Tacotron 2 timing comments. `ai/common/utils.py` also preserves its `db_to_amp` helper. These are direct code-level matches, although they are remnants of a mel-spectrogram pipeline rather than evidence for the shipped TorchScript architecture.
 - AlBadawy's utility source itself credits [CorentinJ's Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) and `r9y9/wavenet_vocoder`. Combined with the 256-value speaker encoder, this supports speaker-embedding ancestry, but not direct ancestry for MVL's real-time queueing or conversion wrapper.
 - Original MVL's reduced `SharedCounter` class matches the widely circulated multiprocessing queue recipe that credits [Eli Bendersky's shared-counter article](https://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/).
 - The callback, queue, rolling-buffer, and crossfade code in `inference_rt.py` carries MetaVoice developer TODO markers and no external attribution. This audit found no matching public source for its distinctive identifiers, so it is treated here as MetaVoice's integration code.
 - Earlier GradioMVL README versions also listed Lei Mao's `Voice-Converter-CycleGAN` and a broken `CorentiJ/Real-Time-Voice-Conversion` URL as possible lineage. Direct comparison found no CycleGAN code or model architecture in MVL. Those links are no longer presented as direct ancestry; the supported CorentinJ connection is the speaker-encoder line described above.
 - GradioMVL's launcher/setup helper was intentionally adapted from [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), as recorded in this repository's commit history.

Most of the original `/ai/` runtime was subsequently rewritten for GradioMVL. This section records the provenance of the recovered project and artifacts; it does not claim that the current runtime files remain verbatim copies of every source named above.

### Credits and License Record

| Project | Relevance established by this audit | Upstream repository license |
| --- | --- | --- |
| [MetaVoice Live](https://github.com/metavoiceio/MetaVoiceLive) | Original application, inference wrapper, distributed models, and built-in targets | [GPL-3.0](https://github.com/metavoiceio/MetaVoiceLive/blob/main/LICENSE.txt) |
| [FreeVC](https://github.com/OlaWod/FreeVC) | Decoder architecture, 24kHz recipe, speaker-conditioning format, and exact speaker-encoder checkpoint match | [MIT](https://github.com/OlaWod/FreeVC/blob/main/LICENSE) |
| [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) | Content-encoder architecture in `b_model.pt` | [MIT](https://github.com/microsoft/unilm/blob/master/LICENSE) |
| [VITS](https://github.com/jaywalnut310/vits) and [HiFi-GAN](https://github.com/jik876/hifi-gan) | Decoder foundations inherited through FreeVC | [MIT](https://github.com/jaywalnut310/vits/blob/main/LICENSE) / [MIT](https://github.com/jik876/hifi-gan/blob/master/LICENSE) |
| [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) and [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | 256-value speaker-encoder family; Resemblyzer powers GradioMVL's Create Voice feature | [MIT](https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/master/LICENSE) / [Apache-2.0](https://github.com/resemble-ai/Resemblyzer/blob/master/LICENSE) |
| [`ebadawy/voice_conversion`](https://github.com/ebadawy/voice_conversion) | Direct source of parameter and helper fragments in original MVL Python | [MIT](https://github.com/ebadawy/voice_conversion/blob/master/LICENSE) |
| [AUTOMATIC1111 Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | Source of GradioMVL's simplified launcher/setup helper | [AGPL-3.0](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt) |

The upstream copyright notices identify Jingyi Li (FreeVC), Microsoft Corporation (WavLM), Jaehyeon Kim (VITS), Jungil Kong (HiFi-GAN), Ehab A. AlBadawy and Siwei Lyu (`voice_conversion`), Corentin Jemine and the predecessors named in his license (Real-Time Voice Cloning), and AUTOMATIC1111. Their repositories and license texts are linked above so those notices are not lost behind MetaVoice's opaque exports.

MetaVoice Live applies GPL-3.0 to its root repository, but does not separately identify the origin or license of the opaque TorchScript weights and built-in target vectors. The unpacked repository also contains a separate proprietary end-user agreement under `services/desktop_app/LICENCE.txt` for a distributed Apple application; GradioMVL does not include that Electron application. Likewise, architectural identification cannot determine ownership or training history. This is a provenance and credit record, not an attempt to assign rights that the released artifacts do not document.

### Preservation Record

The original Git LFS pointers for `model.pt` and `b_model.pt` match the files currently used by GradioMVL exactly.

| Artifact | Bytes | SHA-256 | Recovered role |
| --- | ---: | --- | --- |
| `studio_models/b_model.pt` | 1,262,533,109 | `b4b24ab4c5212c98e2b88a5fc5c12b0bb088789b280a5dbf6b9aa8752f013462` | WavLM-Large-style 16kHz content encoder |
| `studio_models/model.pt` | 157,848,726 | `9abee57727248606e86ab86b5589ce9c9cca6c11aef5eb79181eb50c29286130` | FreeVC/VITS-derived 24kHz decoder |
| `studio_models/model_48k.pt` | 157,846,566 | `4eae94eadd14d266bf693e05baa17c795361fac445bad4fce444ef4fc8bcfec0` | Same decoder family with 48kHz upsampling |
| FreeVC speaker encoder | 17,090,379 | `bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca` | FreeVC-compatible 256-value target encoder recommended by MetaVoice developers |
| Resemblyzer 0.1.4 speaker encoder | 17,090,379 | `39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e` | Different compatible checkpoint currently used by Create Voice |

Known-good environment recorded on 2026-07-18: Python 3.12.10, Gradio 6.20.0, PyTorch/Torchaudio 2.9.1+cu128, NumPy 1.26.4, Librosa 0.11.0, Resemblyzer 0.1.4, PyAudio 0.2.14, SoundFile 0.13.1, and SciPy 1.16.3.

Current operational diagnostics report inference runs that exceed their captured-audio duration, the total number of stale input chunks discarded, and stale output chunks discarded when playback falls behind. No regression WAV is committed because this audit did not identify a clean reference recording with clear redistribution rights.

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
   - Documented recovered model and code provenance, preserved model fingerprints, and marked the project as effectively frozen.

## Setup

 - Install Python 3.12 and make sure it is added to PATH. Python 3.12.10 is the recorded known-good version.
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

Target files are 256-value `float32` speaker embeddings compatible with the FreeVC conditioning input and saved as NumPy arrays. MetaVoice's supplied targets are nonnegative and L2-normalized; `zero.npy` is the intentional all-zero exception. The Create Voice tab produces structurally compatible vectors with Resemblyzer, or you can place a compatible `.npy` file in `studio_models/targets` and restart. The exact encoder used to create MetaVoice's supplied targets was not published.

## Todo

 - Need to add a launch.sh for Linux.
