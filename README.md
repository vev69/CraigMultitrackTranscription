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

## Usage

[Provide example usage instructions, including common command-line arguments]

## Troubleshooting

[Provide links to documentation or support channels]

## Contributing

[Information about contributing to the project]

## License

[Specify the license under which the project is released]

