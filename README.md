# CraigMultitrackTranscription

## Overview

Transcribe Multitrack audio downloaded from discord using Craig. Started from Icavitt project (https://github.com/Icavitt/CraigMultiTrackTranscription) and modified (vibe coding ) to work for Italian Language. Added an experimental alternative using hugging face model https://huggingface.co/whispy/whisper_italian

## Key Features

*   **Multiple Transcription Models:**
    *   **OpenAI Whisper:** Leverages OpenAI's powerful Whisper models for accurate transcription. Supported sizes include [list specific supported sizes, e.g., "tiny," "base," "small," "medium," "large-v2"].
    *   **Hugging Face Optimized Whisper:** Utilizes Hugging Face's optimized versions of Whisper for potential performance improvements.
    *   **Fine-Tuned Italian Model:** A model specifically fine-tuned for Italian language transcription.
    *   **Dual Model Transcription:**  Allows simultaneous transcription using multiple models to improve accuracy ([specify models when using this].
*   **Checkpointing:** Supports resuming transcription from a checkpoint file, allowing for long audio files to be transcribed in stages.
*   **System Standby Prevention:**  Prevents the system from going into standby mode during transcription, ensuring uninterrupted processing of long audio files (Linux only).
*   **Speaker Separation/Combination:**  Provides functionality to separate and combine transcripts from different speakers.
*   **Customizable:** Allows specifying language for transcription ([specify available languages]).
*   **Error Handling:**  Includes robust error handling to gracefully manage potential issues during the transcription process.
*   **Optimized Performance:**  Utilizes hardware acceleration (CUDA) when available for faster transcription.
*   **File Extension Support:** Supports transcription from [list supported file types, e.g., ".mp3", ".wav", ".m4a"].

## Getting Started

1.  **Dependencies:** Ensure you have the required dependencies installed: [List dependencies and installation instructions. E.g., Python 3.7+, ffmpeg, etc.]
2.  **Configuration:** Configure the application with your desired settings [Provide instructions for configuration, e.g., a configuration file].
3.  **Transcription:** Run the application with the desired audio file [Provide command-line arguments/GUI instructions].

```markdown
## Usage

1.  **Prepare Audio:** Place original `.flac` or `.m4a` files in a dedicated directory. Ensure speaker names are identifiable in filenames (e.g., `1-SpeakerName.flac`).
2.  **Run:** Execute the script from your terminal:
    ```bash
    python transcribeCraigAudio.py
    ```
3.  **Follow Prompts:**
    *   **Models:** Select transcription models (by number, name, or `tutti`).
    *   **Parameters (Optional):** Enter `num_beams` and `batch_size` for Hugging Face models (or press Enter for defaults).
    *   **Audio Directory:** Provide the path to your audio files.
    *   **CPU Cores:** Specify cores for parallel processing (or press Enter for default).
    *   **Skip Split (Optional):** If `audio_split` and `split_manifest.json` exist from a previous run, you can choose to skip the splitting/copying phase.
4.  **Processing:** The script automatically splits/copies audio, preprocesses chunks (parallel), transcribes using selected models (GPU if available), saves progress (`transcription_checkpoint.json`), and combines results.
5.  **Output:** Final combined transcripts (`SpeakerName: Text`) are saved in model-specific subdirectories within `transcription_output`.
6.  **Resume:** If interrupted, re-run the script. It will detect the checkpoint and ask to resume.
```

```markdown
## Troubleshooting

*   **Installation/Dependency Errors:** Ensure PyTorch (with correct CUDA version if applicable), `ffmpeg`, and all packages listed in *Getting Started* are installed correctly. Verify `ffmpeg` is in your system's PATH.
*   **`CouldntDecodeError`:** Usually indicates an issue with `ffmpeg`. Check installation and PATH. The audio file might also be corrupted.
*   **`WinError 32: File used by another process`:** Try closing file explorers or antivirus software accessing the output directories. The script includes delays, but persistent locks might require a system restart.
*   **CUDA / OutOfMemoryError:** Try a smaller model size or decrease the `batch_size` parameter during the interactive prompts. Ensure your GPU drivers and PyTorch CUDA version are compatible.
*   **Incorrect Speaker Names:** Double-check your original audio filenames match the `[number]-[SpeakerName]...` pattern.
*   **Incorrect Transcript Order:** While improved sorting logic exists, perfect ordering is hard. Timestamp inaccuracies from the ASR model are the likely cause for remaining issues.
*   **Preprocessing/Splitting Failures:** Check console output for specific errors from `pydub`, `noisereduce`, or `soundfile`. Ensure sufficient disk space.
*   **General Issues:** Check the console output for specific error messages. If reporting an issue, please include the full console output and details about your environment (OS, Python version, hardware).
*   **Further Help:** [If applicable, link to the project's GitHub Issues page or documentation]
```

## License

This project significantly modifies and builds upon the work started by Icavitt in the [CraigMultiTrackTranscription](https://github.com/Icavitt/CraigMultiTrackTranscription) repository. As the original repository does not specify a license, the licensing terms for the original code portions are unclear.

The modifications, enhancements, and new code introduced in **this specific repository** are released under the **MIT License**. Users should be aware of the potential ambiguity regarding the license of the original underlying code.

