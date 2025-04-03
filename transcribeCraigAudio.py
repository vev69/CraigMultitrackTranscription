import os
import json
import sys
import signal
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
import torch

SUPPORTED_FILE_EXTENSIONS = (".flac")
CHECKPOINT_FILE = "checkpoint.json"

# Gestione dei segnali per intercettare l'interruzione
def signal_handler(sig, frame):
    print("\nInterruzione rilevata. Salvataggio del checkpoint...")
    save_checkpoint()
    print(f"Checkpoint salvato in {CHECKPOINT_FILE}. Puoi riavviare l'applicazione per continuare.")
    sys.exit(0)

# Registra il gestore per i segnali SIGINT (Ctrl+C) e SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Funzione per salvare il checkpoint
def save_checkpoint():
    checkpoint_data = {
        "directory": directoryOfFiles,
        "model_choice": MODEL_CHOICE,
        "files_processed": files_processed
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)

# Funzione per caricare il checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data
        except Exception as e:
            print(f"Errore nel caricamento del checkpoint: {e}")
    return None

# Funzione per transcrivere i file
def transcribeFilesInDirectory(directoryPath: str, model, already_processed=None):
    if already_processed is None:
        already_processed = []
    
    filesTranscribed = []
    global files_processed
    files_processed = already_processed.copy()
    
    directoryListDir = os.listdir(directoryPath)
    for file in directoryListDir:
        if file.endswith(SUPPORTED_FILE_EXTENSIONS):
            fileNameWithPath = f'{directoryPath}{os.sep}{file}'
            
            # Verifica se il file è già stato processato
            if fileNameWithPath in already_processed:
                print(f'Skippo {file} perché è già stato processato')
                filesTranscribed.append(fileNameWithPath)
                continue
                
            print(f'Trascrivo {file}...')
            filesTranscribed.append(transcribe.transcribeAudioFile(fileNameWithPath, model, MODEL_CHOICE))
            files_processed.append(fileNameWithPath)
            
            # Salva il checkpoint dopo ogni file
            save_checkpoint()
        else:
            print(f'Skippo {file} perché il tipo non è supportato')
            print(f'I tipi supportati sono {SUPPORTED_FILE_EXTENSIONS}')
    
    return filesTranscribed

# Controlla se esiste un checkpoint
checkpoint = load_checkpoint()
if checkpoint:
    print(f"Checkpoint trovato. Vuoi riprendere la trascrizione dalla directory '{checkpoint['directory']}'? (s/n)")
    resume = input().strip().lower()
    if resume == 's':
        MODEL_CHOICE = checkpoint['model_choice']
        directoryOfFiles = checkpoint['directory']
        files_processed = checkpoint['files_processed']
        print(f"Ripresa della trascrizione con il modello {MODEL_CHOICE} dalla directory {directoryOfFiles}")
    else:
        # Inizia una nuova trascrizione
        MODEL_CHOICE = input("Scegli il modello (whisper / whispy_italian): ").strip().lower()
        directoryOfFiles = input('Inserisci la directory dei file audio da trascrivere: ')
        files_processed = []
else:
    # Nessun checkpoint trovato, inizia normalmente
    MODEL_CHOICE = input("Scegli il modello (whisper / whispy_italian): ").strip().lower()
    directoryOfFiles = input('Inserisci la directory dei file audio da trascrivere: ')
    files_processed = []

# Carica il modello
MODEL = transcribe.load_model(MODEL_CHOICE)

# Verifica se CUDA è disponibile
if torch.cuda.is_available():
    print('CUDA è disponibile')
else:
    print('CUDA non disponibile, terminazione del processo. Aggiorna le dipendenze per risolvere il problema.')
    sys.exit()

# Avvia la trascrizione
transcribedSpeakerFiles = transcribeFilesInDirectory(directoryOfFiles, MODEL, files_processed)

# Combina i file dei relatori
print("Trascrizione completata. Combinazione dei file dei relatori...")
combineSpeakers.combineTranscribedSpeakerFiles(directoryOfFiles)

# Rimuovi il file di checkpoint quando tutto è completato con successo
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Processo completato con successo. Checkpoint rimosso.")
