# This script installs the necessary requirements and launches GradioMVL.
import re
import subprocess
import os
import sys
import importlib.util
import platform

script_path = "."

python = sys.executable
index_url = os.environ.get('INDEX_URL', "")
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")

SUPPORTED_TORCH_BACKENDS = {"cuda", "rocm"}
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ROCM_VERSION = "7.2.1"
ROCM_WINDOWS_BASE_URL = f"https://repo.radeon.com/rocm/windows/rocm-rel-{ROCM_VERSION}"
ROCM_LINUX_BASE_URL = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{ROCM_VERSION}"

if 'GRADIO_ANALYTICS_ENABLED' not in os.environ:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if (major, minor) != (3, 12):
        raise RuntimeError(
            "GradioMVL requires Python 3.12. "
            f"The selected interpreter is Python {major}.{minor}.{micro}."
        )

def run(command, desc=None, errdesc=None, custom_env=None, live: bool = True) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None

def run_pip(command, desc=None, live=True, use_index_url=True):
    index_url_line = f' --index-url {index_url}' if use_index_url and index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0

def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all requirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    import importlib.metadata

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.split("#", 1)[0].strip()
            if line == "":
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if version_required and version_required != version_installed:
                return False

    return True


def requested_torch_backend():
    backend = os.environ.get("TORCH_BACKEND", "cuda").strip().lower()
    if backend not in SUPPORTED_TORCH_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_TORCH_BACKENDS))
        raise RuntimeError(f"Unsupported TORCH_BACKEND '{backend}'. Choose one of: {choices}.")
    return backend


def torch_install_commands(backend, system_name=None):
    """Return pip arguments and descriptions for the selected GPU backend."""
    system_name = system_name or platform.system()

    if backend == "cuda":
        return [
            (
                f"install torch torchaudio --index-url {CUDA_TORCH_INDEX}",
                "PyTorch for NVIDIA CUDA",
            )
        ]

    if backend != "rocm":
        raise RuntimeError(f"Unsupported Torch backend: {backend}")

    if system_name == "Windows":
        sdk_packages = " ".join(
            [
                f"{ROCM_WINDOWS_BASE_URL}/rocm_sdk_core-{ROCM_VERSION}-py3-none-win_amd64.whl",
                f"{ROCM_WINDOWS_BASE_URL}/rocm_sdk_devel-{ROCM_VERSION}-py3-none-win_amd64.whl",
                f"{ROCM_WINDOWS_BASE_URL}/rocm_sdk_libraries_custom-{ROCM_VERSION}-py3-none-win_amd64.whl",
                f"{ROCM_WINDOWS_BASE_URL}/rocm-{ROCM_VERSION}.tar.gz",
            ]
        )
        torch_packages = " ".join(
            [
                f"{ROCM_WINDOWS_BASE_URL}/torch-2.9.1%2Brocm{ROCM_VERSION}-cp312-cp312-win_amd64.whl",
                f"{ROCM_WINDOWS_BASE_URL}/torchaudio-2.9.1%2Brocm{ROCM_VERSION}-cp312-cp312-win_amd64.whl",
            ]
        )
        return [
            (f"install --no-cache-dir {sdk_packages}", "ROCm SDK 7.2.1"),
            (f"install --no-cache-dir {torch_packages}", "PyTorch for AMD ROCm"),
        ]

    if system_name == "Linux":
        torch_packages = " ".join(
            [
                f"{ROCM_LINUX_BASE_URL}/torch-2.9.1%2Brocm{ROCM_VERSION}.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl",
                f"{ROCM_LINUX_BASE_URL}/torchaudio-2.9.0%2Brocm{ROCM_VERSION}.gite3c6ee2b-cp312-cp312-linux_x86_64.whl",
                f"{ROCM_LINUX_BASE_URL}/triton-3.5.1%2Brocm{ROCM_VERSION}.gita272dfa8-cp312-cp312-linux_x86_64.whl",
            ]
        )
        return [
            (f"install --no-cache-dir {torch_packages}", "PyTorch for AMD ROCm"),
        ]

    raise RuntimeError("ROCm launch is supported only on Windows and Linux.")


def installed_torch_backend():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is installed but could not be imported. Delete the selected virtual "
            "environment and launch again to rebuild it."
        ) from exc

    if getattr(torch.version, "hip", None):
        backend = "rocm"
        runtime_version = torch.version.hip
    elif getattr(torch.version, "cuda", None):
        backend = "cuda"
        runtime_version = torch.version.cuda
    else:
        backend = "cpu"
        runtime_version = "none"

    return torch, backend, runtime_version


def verify_torch_backend(expected_backend):
    torch, actual_backend, runtime_version = installed_torch_backend()

    if actual_backend != expected_backend:
        environment = os.environ.get("VENV_DIR", "the selected virtual environment")
        raise RuntimeError(
            f"{environment} contains a {actual_backend.upper()} PyTorch build, but the "
            f"{expected_backend.upper()} launcher was selected. Use separate virtual "
            "environments or delete this one and launch again."
        )

    # PyTorch intentionally exposes AMD HIP devices through the torch.cuda API.
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"The {expected_backend.upper()} PyTorch build is installed, but no compatible "
            "GPU is available. Check the graphics driver and GPU support matrix."
        )

    device_name = torch.cuda.get_device_name(0)
    print(
        f"PyTorch {torch.__version__}: {expected_backend.upper()} {runtime_version}; "
        f"device 0 is {device_name}"
    )


def prepare_environment():
    requirements_file = "requirements.txt"

    check_python_version()

    backend = requested_torch_backend()

    print(f"Python {sys.version}")
    print(f"Requested GPU backend: {backend.upper()}")

    if not is_installed("torch"):
        for command, description in torch_install_commands(backend):
            run_pip(command, description, live=True, use_index_url=False)

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    verify_torch_backend(backend)


def start():
    print(f"Launching GradioMVL!")
    import mvl
    mvl.start()
