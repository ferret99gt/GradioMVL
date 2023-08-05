from typing import Dict, List

import sounddevice as sd
from util.data_types import DeviceInfo


def get_devices() -> Dict[str, List[DeviceInfo]]:
    sd._terminate()
    sd._initialize()

    inputs = []
    outputs = []
    excludeInputs = []
    excludeOutputs = []

    for index, dinfo in enumerate(sd.query_devices()):
        # If multiple portaudio interface APIs exist, chooses the first one
        # to prevent multiple devices being shown to the user (some of which may not work)
        # Refs:
        #  1. http://files.portaudio.com/docs/v19-doxydocs/api_overview.html
        #  2. https://stackoverflow.com/questions/20943803/pyaudio-duplicate-devices
        if dinfo["hostapi"] == 0:
            device_info = DeviceInfo(
                **{
                    "name": dinfo["name"],
                    "index": index,
                    "max_input_channels": dinfo["max_input_channels"],
                    "max_output_channels": dinfo["max_output_channels"],
                    "default_sample_rate": dinfo["default_samplerate"],
                    "is_default_input": index == sd.default.device[0],
                    "is_default_output": index == sd.default.device[1],
                }
            )

            if device_info.is_duplex:
                inputs.append(device_info)
                outputs.append(device_info)
            elif device_info.max_input_channels:
                inputs.append(device_info)
            elif device_info.max_output_channels:
                outputs.append(device_info)
            else:
                raise ValueError(f"Unknown device, {str(device_info)}")

    return {
        "inputs": inputs,
        "outputs": outputs,
    }
