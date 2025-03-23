import os
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
import torch

SUPPORTED_FILE_EXTENSIONS = (".flac")

# Scelta del modello
MODEL_CHOICE = input("Scegli il modello (whisper / whispy_italian): ").strip().lower()

# Caricamento del modello
MODEL = transcribe.load_model(MODEL_CHOICE)

# Verifica se CUDA è disponibile
if torch.cuda.is_available():
    print('CUDA è disponibile, usando fp16')
else:
    print('CUDA non disponibile, terminazione del processo. Aggiorna le dipendenze per risolvere il problema.')
    sys.exit()

def transcribeFilesInDirectory(directoryPath: str, model):
    filesTranscribed = []
    directoryListDir = os.listdir(directoryPath)
    for file in directoryListDir:
        if(file.endswith(SUPPORTED_FILE_EXTENSIONS)):
            fileNameWithPath = f'{directoryPath}{os.sep}{file}'
            filesTranscribed.append(transcribe.transcribeAudioFile(fileNameWithPath, model, MODEL_CHOICE)) # Passa il modello e la scelta
        else:
            print(f'Skipping {file} as it\'s not a supported type')
            print(f'supported types are {SUPPORTED_FILE_EXTENSIONS}')
    return filesTranscribed

directoryOfFiles = input('enter the directory to audio files to transcribe: ')
transcribedSpeakerFiles = transcribeFilesInDirectory(directoryOfFiles, MODEL) # Passa il modello qui
combineSpeakers.combineTranscribedSpeakerFiles(directoryOfFiles)