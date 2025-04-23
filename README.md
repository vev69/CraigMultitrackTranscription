# CraigMultitrackTranscription

## Overview

Transcribe Multitrack audio downloaded from discord using Craig. Started from Icavitt project (https://github.com/Icavitt/CraigMultiTrackTranscription) and modified (vibe coding ) to work for Italian Language.

This tool automates the transcription of multi-track `.flac` or `.m4a` audio recordings (typically from Craig). It performs an initial audio analysis, intelligent audio splitting, adaptive preprocessing (including volume normalization and conditional noise reduction), parallel processing for CPU-bound tasks, checkpointing for resumable sessions, and finally combines the results attributing text to speakers based on original filenames.

## Key Features

*   **Multiple Transcription Models:**
    *   **OpenAI Whisper:** Leverages OpenAI's powerful Whisper models. Supported: `whisper-medium`, `whisper-largev2`, `whisper-largev3`.
    *   **Hugging Face Optimized Whisper:** Utilizes Hugging Face's optimized versions. Supported: `hf_whisper-medium`, `hf_whisper-largev2`, `hf_whisper-largev3`.
    *   **Fine-Tuned Italian Model:** Includes `whispy/whisper_italian` via Hugging Face.
    *   **Sequential Model Execution:** Run multiple selected models sequentially on the same audio data.
*   **Intelligent Hybrid Chunking:** Automatically analyzes long audio files. For files exceeding a threshold (default > 11 minutes), it aims for a target chunk duration (default 10 minutes) but intelligently adjusts cut points to occur during detected silences nearby, minimizing mid-speech cuts. Shorter files are processed as a single chunk.
*   **Preliminary Audio Analysis:** Performs a parallel analysis of original audio files to determine key metrics like average loudness (LUFS), noise floor, and dynamic silence parameters used in subsequent steps.
*   **Adaptive Audio Preprocessing:** Applies preprocessing steps **conditionally** based on the analysis of the original audio:
    *   **Volume Boost:** Boosts the volume of speakers/files identified as significantly quieter than the loudest speaker (up to a configurable limit).
    *   **Conditional Noise Reduction:** Applies noise reduction (`noisereduce`) with varying intensity (or skips it) based on the estimated Signal-to-Noise Ratio (SNR) of the original audio. Aims for less aggressive reduction on cleaner audio.
    *   **LUFS Normalization:** Normalizes the final preprocessed chunks to a consistent target loudness level (default -19 LUFS, based on the loudest speaker up to a cap) using `pyloudnorm` for perceptually uniform volume across speakers before transcription. Includes safety clipping.
*   **Parallel Processing & RAM Awareness:** Leverages multiple CPU cores for the initial audio analysis and the splitting/copying phases. The analysis phase dynamically adjusts the number of parallel workers based on available system RAM and estimated file requirements to prevent memory errors.
*   **Checkpointing & Resumption:** Automatically saves progress (models processed, chunks transcribed per model) to `transcription_checkpoint.json`. Allows resuming interrupted sessions, skipping already completed steps.
*   **System Standby Prevention:** Prevents the system from entering sleep/standby during long processing times (Supports Windows, macOS, Linux via D-Bus).
*   **Speaker Attribution & Combined Output:** Identifies speakers based on original filenames (e.g., `1-SpeakerName.flac`). Combines all transcribed chunks for a given model into a single, chronologically ordered text file, formatted as `SpeakerName: Text`. Uses an improved sorting logic based on chunk start/end times to handle potential timestamp inaccuracies from the ASR.
*   **Robust Error Handling:** Manages common errors during analysis, splitting, preprocessing, transcription, and file operations.
*   **GPU Acceleration:** Utilizes CUDA (NVIDIA GPU) via PyTorch for significantly faster transcription when available.
*   **File Format Support:** Primarily designed for `.flac` and `.m4a` files.

## Getting Started

1.  **Dependencies:** Ensure you have the required dependencies installed:
    *   **Python:** Python 3.9 or later is recommended.
    *   **FFmpeg:** This is essential for audio processing libraries (`pydub`). Installation varies by OS (see original instructions or FFmpeg website). Ensure it's in your system's PATH.
    *   **PyTorch:** Required for Whisper and Transformers. Install following official instructions at [https://pytorch.org/](https://pytorch.org/), selecting the correct version for your OS and CUDA (if available).
    *   **Python Packages:** Clone this repository, navigate into its directory, and install the necessary Python libraries (ideally within a virtual environment):
        ```bash
        # Create/activate virtual environment (e.g., python -m venv .venv; source .venv/bin/activate)

        # Core Transcription & Splitting/Preprocessing Libs:
        pip install -U openai-whisper transformers torch soundfile noisereduce pydub numpy

        # For LUFS Normalization (Highly Recommended):
        pip install pyloudnorm scipy

        # For Dynamic RAM Worker Limiting (Recommended):
        pip install psutil

        # Optional for HF Optimization:
        pip install -U optimum[bettertransformer]

        # Optional for Linux Standby Prevention:
        # pip install pydbus

        # Optional for the separate analysis script (analyze_audio.py):
        # pip install matplotlib librosa
        ```
        *(Note: `scipy` is often a dependency of `pyloudnorm` or `librosa`)*

2.  **Configuration:** No separate configuration file. Options like model selection, HF parameters (`beams`, `batch_size`), directories, and CPU cores are requested interactively when running the main script for a new session. Preprocessing parameters (LUFS target, NR settings) are currently set as constants within the scripts (`splitAudio.py`, `transcribeAudio.py`) but could be exposed as arguments if needed.

3.  **Transcription:** Prepare your audio files (see Usage section) and run the main script from your terminal:
    ```bash
    python transcribeCraigAudio.py
    ```
    The script will then prompt you for the required information. See the **Usage** section below for more details.

## Usage

1.  **Prepare Audio:** Place original `.flac` or `.m4a` files in a dedicated directory. Ensure filenames follow a pattern like `[number]-[SpeakerName]_[optional_suffix].ext` (e.g., `1-Alice_123.flac`, `2-Bob.m4a`) for correct speaker identification.
2.  **Run:** Execute the script from your terminal (activate virtual environment first):
    ```bash
    python transcribeCraigAudio.py
    ```
3.  **Follow Prompts (New Session):**
    *   **Models:** Select transcription models (by number, name, or `tutti`).
    *   **HF Parameters (Optional):** Enter `num_beams` and `batch_size` for Hugging Face models (defaults are 1 and 16).
    *   **Audio Directory:** Provide the path to your original audio files.
    *   **CPU Cores:** Specify cores for parallel analysis/splitting (defaults to system count). Analysis phase might dynamically use fewer cores based on available RAM.
    *   **Skip Existing Split/Preproc (Optional):** If `audio_split`, `audio_preprocessed_chunks` and `split_manifest.json` exist (from a prior interrupted run where you chose 'n' at resume), you'll be asked if you want to skip re-running the splitting and preprocessing steps and use the existing files. Choose 's' to skip, 'n' to regenerate.
4.  **Processing Steps:**
    *   **(Optional) Analysis:** Analyzes original files to determine dynamic silence parameters and audio metrics (LUFS, SNR, etc.). Calculates the target LUFS for normalization. (Parallel, RAM-aware workers).
    *   **(Optional) Splitting:** Splits long files into chunks (~10 min target, cut near silences) or copies short files into `audio_split`. Creates `split_manifest.json` containing chunk info and original audio metrics. (Parallel workers).
    *   **(Optional) Preprocessing:** Processes chunks from `audio_split`, applies conditional boost/NR and LUFS normalization based on manifest data, saves results to `audio_preprocessed_chunks`. (Parallel workers).
    *   **Transcription:** Iterates through selected models. For each model, it transcribes the preprocessed chunks (using GPU if available), recording progress in `transcription_checkpoint.json`.
    *   **Combination:** After all chunks for a model are transcribed, combines the results into a single chronologically ordered file (`SpeakerName: Text` format) in the model's output directory within `transcription_output`.
5.  **Output:** Final combined transcripts are in `transcription_output/[model_name]/`. Intermediate files are in `audio_split` and `audio_preprocessed_chunks`.
6.  **Resume:** If interrupted (Ctrl+C), re-run the script. It detects the checkpoint and asks to resume, continuing transcription/combination for remaining models/chunks. Splitting/Preprocessing are *not* automatically resumed by default if interrupted during those phases (unless you choose 's' at the "Skip Existing" prompt if it appears).

## Troubleshooting

*   **`NameError` / `TypeError` related to function arguments:** Double-check that you have the latest matching versions of all `.py` files (`transcribeCraigAudio.py`, `splitAudio.py`, `preprocessAudioFiles.py`, `transcribeAudio.py`, `combineSpeakerTexts.py`) as function definitions and calls have changed between iterations.
*   **`WinError 32: File is being used...`**: Windows file locking issue during rename/delete. Ensure antivirus/other programs aren't interfering. Script delays might help, but restart may be needed.
*   **`CouldntDecodeError` (pydub):** Verify `ffmpeg` installation and PATH.
*   **`Unable to allocate...` (Memory Error during Analysis):** The dynamic RAM-based worker limit should prevent this now. If it persists, the RAM estimation might be inaccurate, or available RAM is extremely low. Try explicitly requesting fewer cores when prompted.
*   **`pyloudnorm not found` / `librosa not found` / `psutil not found`:** Install the missing dependency (`pip install pyloudnorm`, `pip install librosa`, `pip install psutil`). LUFS normalization and RAM-aware scaling won't work without them.
*   **CUDA Errors / OOM:** Try smaller model size or reduce HF `batch_size`. Verify PyTorch/CUDA compatibility.
*   **Incorrect Speaker Names:** Check original filenames match `Number-SpeakerName...ext` pattern.
*   **Incorrect Transcript Order:** The sorting heuristic helps but isn't perfect for rapid exchanges. Timestamp inaccuracies from ASR are the main limitation.
*   **"Ovattato" / Muffled Audio:** The current preprocessing aims to be less aggressive. If still an issue:
    *   Try disabling Noise Reduction entirely (by modifying the `apply_nr` logic or `noise_reduce=False` call in `preprocess_audio`). Whisper might handle the noise acceptably.
    *   Experiment with `nr_prop_decrease_*` values in `transcribeAudio.py`.
    *   Ensure LUFS normalization isn't boosting *too* much (check target LUFS).
*   **Slow Performance:** Ensure CUDA is active. Analysis/Splitting are CPU/disk bound. Preprocessing might be slow if using CPU for advanced NR (if added later). Transcription is GPU bound.

## Contributing

Please report issues encountered via the GitHub issue tracker for this repository.

## License

This project significantly modifies and builds upon the work started by Icavitt in the [CraigMultiTrackTranscription](https://github.com/Icavitt/CraigMultiTrackTranscription) repository. As the original repository does not specify a license, the licensing terms for the original code portions are unclear.

The modifications, enhancements, and new code introduced in **this specific repository** are released under the **MIT License**. Users should be aware of the potential ambiguity regarding the license of the original underlying code.