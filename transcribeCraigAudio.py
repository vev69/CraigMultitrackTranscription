import os
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
import torch

SUPPORTED_FILE_EXTENSIONS = (".flac")

# Model choice
MODEL_CHOICE = input("Scegli il modello (whisper / whispy_italian): ").strip().lower()

# Load Model
MODEL = transcribe.load_model(MODEL_CHOICE)

# Check if CUDA is available
if torch.cuda.is_available():
    print('CUDA è disponibile')
else:
    print('CUDA non disponibile, terminazione del processo. Aggiorna le dipendenze per risolvere il problema.')
    sys.exit()

def transcribeFilesInDirectory(directoryPath: str, model):
    filesTranscribed = []
    directoryListDir = os.listdir(directoryPath)
    for file in directoryListDir:
        if(file.endswith(SUPPORTED_FILE_EXTENSIONS)):
            fileNameWithPath = f'{directoryPath}{os.sep}{file}'
            filesTranscribed.append(transcribe.transcribeAudioFile(fileNameWithPath, model, MODEL_CHOICE))
        else:
            print(f'Skippo {file} perche\'s il tipo non e\' supportato')
            print(f'i tipi supportati sono {SUPPORTED_FILE_EXTENSIONS}')
    return filesTranscribed

directoryOfFiles = input('inserisci la directory dei file audio da trascrivere: ')
transcribedSpeakerFiles = transcribeFilesInDirectory(directoryOfFiles, MODEL) 
combineSpeakers.combineTranscribedSpeakerFiles(directoryOfFiles)
