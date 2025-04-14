# --- START OF transcribeCraigAudio.py (MODIFICATO v2) ---

import os
import json
import sys
import signal
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
import transcriptionUtils.preprocessAudioFiles as preprocessor
import transcriptionUtils.splitAudio as splitter
import torch
import shutil
import platform
import time
import atexit

# --- Moduli specifici per OS per prevenire lo standby (invariati) ---
try:
    if platform.system() == "Windows":
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        print("Importato ctypes per la gestione standby su Windows.")
    elif platform.system() == "Darwin": # macOS
        import subprocess
        caffeinate_process = None
        print("Importato subprocess per la gestione standby su macOS (caffeinate).")
    elif platform.system() == "Linux":
        try:
            import pydbus
            bus = pydbus.SessionBus()
            inhibitor = None
            print("Importato pydbus per la gestione standby su Linux (D-Bus).")
        except ImportError:
            print("Modulo pydbus non trovato. L'inibizione dello standby su Linux via D-Bus non è disponibile.")
            inhibitor = None
        except Exception as e:
             print(f"Errore durante l'inizializzazione di D-Bus: {e}")
             inhibitor = None
except ImportError as e: print(f"Errore import moduli specifici OS per standby: {e}")
except Exception as e: print(f"Errore generico import per standby: {e}")


SUPPORTED_FILE_EXTENSIONS = (".flac",".m4a")
CHECKPOINT_FILE = "transcription_checkpoint.json"
BASE_OUTPUT_FOLDER_NAME = "transcription_output"
# NUOVE COSTANTI PER LE DIRECTORY INTERMEDIE
SPLIT_FOLDER_NAME = "audio_split" # Cartella per audio diviso/copiato
PREPROCESSED_FOLDER_NAME = "audio_preprocessed_chunks" # Cartella per chunk processati

# Global variable for checkpointing
checkpoint_data = {}

# --- Valori di Default per Parametri Interattivi (NUOVO) ---
DEFAULT_NUM_BEAMS = 1
DEFAULT_BATCH_SIZE_HF = 24

# --- Funzioni Gestione Standby (invariate) ---
def prevent_sleep():
    # ... (codice invariato) ...
    global caffeinate_process, inhibitor
    print(">>> Attivazione richiesta per prevenire lo standby del sistema...")
    system = platform.system()
    try:
        if system == "Windows":
            result = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            if result == 0: print("!!! Errore SetThreadExecutionState.")
            else: print("Richiesta anti-standby attivata (Windows).")
        elif system == "Darwin":
            if caffeinate_process is None or caffeinate_process.poll() is not None:
                caffeinate_process = subprocess.Popen(['caffeinate', '-i'])
                print(f"Processo 'caffeinate -i' avviato (PID: {caffeinate_process.pid}).")
        elif system == "Linux":
            if 'pydbus' in sys.modules and inhibitor is None:
                 try:
                     mgr = bus.get('org.freedesktop.PowerManagement', '/org/freedesktop/PowerManagement/Inhibit')
                     app_id = "CraigTranscriptionScript"; reason = "Running audio transcription"
                     cookie = mgr.Inhibit(app_id, reason); inhibitor = cookie
                     print(f"Richiesta anti-standby D-Bus inviata (Inhibitor: {inhibitor}).")
                 except Exception as e_dbus: print(f"!!! Errore D-Bus Inhibit: {e_dbus}"); inhibitor = None
    except Exception as e: print(f"!!! Errore in prevent_sleep: {e}")

def allow_sleep():
    # ... (codice invariato) ...
    global caffeinate_process, inhibitor
    print(">>> Disattivazione richiesta anti-standby...")
    system = platform.system()
    try:
        if system == "Windows":
            result = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            if result == 0: print("!!! Errore reset SetThreadExecutionState.")
            else: print("Richiesta anti-standby disattivata (Windows).")
        elif system == "Darwin":
            if caffeinate_process is not None and caffeinate_process.poll() is None:
                print(f"Terminazione 'caffeinate' (PID: {caffeinate_process.pid})...")
                caffeinate_process.terminate()
                try: caffeinate_process.wait(timeout=1)
                except subprocess.TimeoutExpired: caffeinate_process.kill()
                caffeinate_process = None; print("Processo 'caffeinate' terminato.")
        elif system == "Linux":
            if inhibitor is not None and 'pydbus' in sys.modules:
                try:
                    print(f"Rilascio richiesta D-Bus (Inhibitor: {inhibitor})...")
                    mgr = bus.get('org.freedesktop.PowerManagement', '/org/freedesktop/PowerManagement/Inhibit')
                    mgr.UnInhibit(inhibitor); inhibitor = None
                    print("Richiesta D-Bus rilasciata.")
                except Exception as e_dbus: print(f"!!! Errore D-Bus UnInhibit: {e_dbus}")
    except Exception as e: print(f"!!! Errore in allow_sleep: {e}")

atexit.register(allow_sleep) # Registra la funzione di cleanup

# --- Gestione Segnali (invariata) ---
def signal_handler(sig, frame):
    # ... (codice invariato) ...
    print("\n*** Interruzione rilevata! Pulizia e salvataggio... ***")
    allow_sleep()
    save_checkpoint()
    print(f"Checkpoint salvato in {CHECKPOINT_FILE}.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Funzioni Checkpoint (invariate) ---
def save_checkpoint():
    # ... (codice invariato) ...
    global checkpoint_data
    if not checkpoint_data: return
    try:
        checkpoint_data['last_saved'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
    except Exception as e: print(f"!!! Errore salvataggio checkpoint: {e}")

def load_checkpoint():
    # ... (codice invariato) ...
    global checkpoint_data
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict) and 'base_input_directory' in loaded_data:
                     print(f"Checkpoint trovato: {CHECKPOINT_FILE} (Ultimo salvataggio: {loaded_data.get('last_saved', 'N/A')})")
                     checkpoint_data = loaded_data
                     return True
                else: print(f"Warn: Checkpoint {CHECKPOINT_FILE} invalido."); checkpoint_data = {}; return False
        except Exception as e: print(f"Errore caricamento checkpoint: {e}."); checkpoint_data = {}; return False
    else: checkpoint_data = {}; return False

# --- MODIFICATA Funzione transcribeFilesInDirectory (per accettare parametri) ---
def transcribeFilesInDirectory(split_manifest: dict,
                               preprocessed_dir: str,
                               model,
                               model_choice: str,
                               output_dir: str,
                               # NUOVI PARAMETRI AGGIUNTI
                               num_beams: int,
                               batch_size_hf: int):
    """
    Trascrive i file audio chunk (o file originali corti) listati nel manifest.
    Passa i parametri num_beams e batch_size_hf a transcribeAudioFile.
    """
    global checkpoint_data
    filesTranscribed_chunk_paths = [] # Lista dei percorsi dei chunk tentati
    processed_in_this_session = 0

    if not checkpoint_data:
        print("ERRORE: Dati checkpoint non inizializzati.")
        return []
    if not split_manifest or 'files' not in split_manifest or not split_manifest['files']:
        print("ERRORE: Manifest di split non valido o vuoto.")
        return []

    # Modello specifico per cui recuperare i chunk già processati
    # Le chiavi nel checkpoint sono i percorsi assoluti dei chunk file (da split_manifest)
    already_processed_chunks = checkpoint_data.setdefault('files_processed', {}).setdefault(model_choice, [])

    # Ordina i chunk da processare (opzionale ma buona pratica, es. per file originale poi index)
    # Usiamo i percorsi assoluti dei chunk come identificativi
    chunk_paths_to_process = sorted(split_manifest['files'].keys())
    total_chunks = len(chunk_paths_to_process)

    print(f"Trovati {total_chunks} chunk/file nel manifest da considerare per la trascrizione.")

    for file_index, chunk_abs_path in enumerate(chunk_paths_to_process):
        chunk_filename = os.path.basename(chunk_abs_path)
        preprocessed_chunk_path = os.path.join(preprocessed_dir, chunk_filename) # Percorso atteso preprocessato

        # 1. Controlla se già processato secondo il checkpoint
        if chunk_abs_path in already_processed_chunks:
            print(f"({file_index+1}/{total_chunks}) Skipping {chunk_filename} (già processato per '{model_choice}' secondo checkpoint)")
            filesTranscribed_chunk_paths.append(chunk_abs_path)
            continue

        # 2. Controlla se il file chunk preprocessato esiste
        if not os.path.exists(preprocessed_chunk_path):
            print(f"({file_index+1}/{total_chunks}) Skipping {chunk_filename}: File chunk preprocessato non trovato in '{preprocessed_dir}'.")
            # MARCA COME TENTATO NEL CHECKPOINT (usando il path assoluto del chunk)
            checkpoint_data['files_processed'].setdefault(model_choice, []).append(chunk_abs_path)
            filesTranscribed_chunk_paths.append(chunk_abs_path)
            processed_in_this_session += 1
            save_checkpoint()
            continue # Passa al prossimo chunk

        # 3. Se non già processato e il file preprocessato esiste, procedi con la trascrizione
        print(f"\n--- ({file_index+1}/{total_chunks}) Trascrizione '{model_choice}' per: {chunk_filename} (da: {os.path.basename(preprocessed_chunk_path)}) ---")

        # Chiama la trascrizione PASSANDO I NUOVI PARAMETRI
        output_file_path = transcribe.transcribeAudioFile(
            preprocessed_chunk_path,
            model,
            model_choice,
            output_dir,
            # Passa i parametri ricevuti
            num_beams=num_beams,
            batch_size_hf=batch_size_hf
        )
        # --- LOGICA SUCCESSO/FALLIMENTO (basata su output_file_path) ---
        # Aggiungi il chunk_abs_path al checkpoint *indipendentemente* dal successo/fallimento
        checkpoint_data['files_processed'].setdefault(model_choice, []).append(chunk_abs_path)
        filesTranscribed_chunk_paths.append(chunk_abs_path)
        processed_in_this_session += 1

        if output_file_path and os.path.exists(output_file_path) and "FAILED" not in os.path.basename(output_file_path).upper():
            print(f"Trascrizione chunk completata con successo per {chunk_filename}.")
        elif output_file_path and os.path.exists(output_file_path):
            print(f"ERRORE o INFO durante trascrizione chunk {chunk_filename}. File output: {os.path.basename(output_file_path)}")
        else:
            print(f"ERRORE grave: Trascrizione chunk {chunk_filename} non ha prodotto un file finale valido o ha restituito None.")

        save_checkpoint() # Salva checkpoint dopo ogni tentativo di chunk

    print(f"\n--- Completata sessione di trascrizione chunk per modello '{model_choice}' ---")
    print(f"Chunk considerati/tentati in totale per '{model_choice}': {len(filesTranscribed_chunk_paths)}")
    print(f"Chunk tentati (successo o fallimento) in questa sessione per '{model_choice}': {processed_in_this_session}")
    return filesTranscribed_chunk_paths


# ============================================================================
# --- Main Execution Logic ---
# ============================================================================

if __name__ == "__main__":

    prevent_sleep()

    # --- Variabili per i parametri (NUOVO) ---
    num_beams_to_use = DEFAULT_NUM_BEAMS
    batch_size_hf_to_use = DEFAULT_BATCH_SIZE_HF

    # --- Load Checkpoint or Initialize ---
    checkpoint_found = load_checkpoint()

    if checkpoint_found:
        # ... (Logica ripresa/nuova sessione invariata, ma mostra info aggiuntive) ...
        print(f"Ripresa da checkpoint.")
        # Recupera parametri da checkpoint, se presenti
        num_beams_to_use = checkpoint_data.get("num_beams", DEFAULT_NUM_BEAMS)
        batch_size_hf_to_use = checkpoint_data.get("batch_size_hf", DEFAULT_BATCH_SIZE_HF)
        print(f"Utilizzo parametri da checkpoint/default: Beams={num_beams_to_use}, BatchSize={batch_size_hf_to_use}")        
        print(f"  Directory input originale: {checkpoint_data.get('base_input_directory', 'N/A')}")
        print(f"  Directory split audio: {checkpoint_data.get('split_audio_directory', 'N/A')}")
        print(f"  Directory preprocessed chunks: {checkpoint_data.get('preprocessed_audio_directory', 'N/A')}")
        print(f"  Manifest split: {checkpoint_data.get('split_manifest_path', 'N/A')}")
        print(f"  Modelli scelti: {checkpoint_data.get('original_model_choice', 'N/A')}")
        print(f"  Modelli rimasti: {checkpoint_data.get('models_to_process', [])}")

        while True:
             resume = input("Vuoi riprendere la sessione precedente? (s/n): ").strip().lower()
             if resume in ['s', 'n']: break
             else: print("Risposta non valida. Inserisci 's' per sì o 'n' per no.")

        if resume == 'n':
            # ... (Logica rimozione checkpoint e reset) ...
            print("Avvio nuova sessione. RIMOZIONE del checkpoint precedente...")
            try:
                if os.path.exists(CHECKPOINT_FILE):
                    os.remove(CHECKPOINT_FILE); print(f"File checkpoint '{CHECKPOINT_FILE}' rimosso.")
                # Considera se rimuovere anche le cartelle intermedie della sessione precedente
                # base_dir_old = checkpoint_data.get('base_input_directory')
                # if base_dir_old:
                #     shutil.rmtree(os.path.join(base_dir_old, SPLIT_FOLDER_NAME), ignore_errors=True)
                #     shutil.rmtree(os.path.join(base_dir_old, PREPROCESSED_FOLDER_NAME), ignore_errors=True)
                #     shutil.rmtree(os.path.join(base_dir_old, BASE_OUTPUT_FOLDER_NAME), ignore_errors=True)
                checkpoint_data = {}; checkpoint_found = False
            except OSError as e:
                print(f"ATTENZIONE: Impossibile rimuovere checkpoint: {e}. Il vecchio file potrebbe rimanere.");
                checkpoint_data = {}; checkpoint_found = False
        else:
             print("Ripresa della sessione dal checkpoint.")
             # Verifica esistenza directory e manifest dal checkpoint
             if not os.path.isdir(checkpoint_data.get('split_audio_directory','')) or \
                not os.path.isdir(checkpoint_data.get('preprocessed_audio_directory','')) or \
                not os.path.isfile(checkpoint_data.get('split_manifest_path','')):
                 print("ERRORE: Directory intermedie o manifest dal checkpoint non trovati. Impossibile riprendere.")
                 print("Rimuovere il checkpoint ed eseguire di nuovo lo script.")
                 allow_sleep(); sys.exit(1)


    # --- Get User Input if NO valid checkpoint exists or user chose NOT to resume ---
    if not checkpoint_found:
        print("\n--- Nuova Sessione ---")
        # --- Selezione Modelli (invariata rispetto a v1) ---
        ALL_AVAILABLE_MODELS = [
            "whisper-medium", "whisper-largev2", "whisper-largev3",
            "hf_whisper-medium", "hf_whisper-largev2", "hf_whisper-largev3",
            "whispy_italian"
        ]
        selected_models = []
        while True:
            # ... (Codice per chiedere modelli all'utente: 'tutti', numeri o nomi) ...
            print("\nModelli di trascrizione disponibili:")
            for i, model_name in enumerate(ALL_AVAILABLE_MODELS): print(f"  {i+1}. {model_name}")
            print("\nInserisci i numeri o i nomi dei modelli da eseguire, separati da virgola.")
            print("Oppure scrivi 'tutti' per eseguirli tutti in sequenza.")
            model_input = input("Scegli modelli: ").strip()
            selected_models = []
            invalid_selection = False
            if model_input.lower() == 'tutti':
                selected_models = ALL_AVAILABLE_MODELS; break
            parts = [p.strip() for p in model_input.split(',') if p.strip()]
            if not parts: print("Selezione vuota. Riprova."); continue
            for part in parts:
                found = False
                if part.isdigit():
                    try:
                        index = int(part) - 1
                        if 0 <= index < len(ALL_AVAILABLE_MODELS):
                            model = ALL_AVAILABLE_MODELS[index]
                            if model not in selected_models: selected_models.append(model)
                            found = True
                        else: invalid_selection = True; print(f"Errore: Numero '{part}' non valido."); break
                    except ValueError: pass
                if not found:
                    match_found = False
                    for available_model in ALL_AVAILABLE_MODELS:
                        if available_model.lower() == part.lower():
                            if available_model not in selected_models: selected_models.append(available_model)
                            match_found = True; break
                    if not match_found: invalid_selection = True; print(f"Errore: Modello '{part}' non riconosciuto."); break
                    found = match_found
                if invalid_selection: break
            if not invalid_selection and selected_models: print(f"Modelli selezionati: {', '.join(selected_models)}"); break
            else: print("Selezione non valida. Riprova.")

        # --- Richiesta Parametri Interattiva (NUOVO) ---
        print("\n--- Configurazione Parametri Trascrizione (per modelli HF) ---")
        while True:
            try:
                beams_input = input(f"Numero di beams (default {DEFAULT_NUM_BEAMS}, Invio per default): ").strip()
                num_beams_to_use = DEFAULT_NUM_BEAMS if not beams_input else int(beams_input)
                if num_beams_to_use > 0: break
                else: print("Inserisci un numero intero positivo.")
            except ValueError: print("Inserisci un numero intero valido.")

        while True:
            try:
                batch_input = input(f"Batch size (default {DEFAULT_BATCH_SIZE_HF}, Invio per default): ").strip()
                batch_size_hf_to_use = DEFAULT_BATCH_SIZE_HF if not batch_input else int(batch_input)
                if batch_size_hf_to_use > 0: break
                else: print("Inserisci un numero intero positivo.")
            except ValueError: print("Inserisci un numero intero valido.")
        print(f"Parametri scelti: Beams={num_beams_to_use}, Batch Size={batch_size_hf_to_use}")

        # --- Chiedi directory input (invariato) ---
        while True:
            base_input_directory = os.path.abspath(input('Inserisci directory audio originale: ').strip()) # Usa path assoluto
            if os.path.isdir(base_input_directory): break
            else: print(f"Errore: '{base_input_directory}' non è una directory valida.")

        # --- RILEVAMENTO CORE e Richiesta (MODIFICATO) ---
        try:
            detected_cores = os.cpu_count()
            default_cores_prompt = f"(default: {detected_cores}, rilevato dal sistema)"
        except NotImplementedError:
            detected_cores = 8 # Fallback
            default_cores_prompt = f"(default: {detected_cores}, fallback)"

        while True:
            try:
                # Usa la variabile aggiornata nel prompt
                core_input = input(f"Numero di core CPU per split/preprocessing {default_cores_prompt}: ").strip()
                num_cores_to_use = detected_cores if not core_input else int(core_input)
                if num_cores_to_use > 0: break
                else: print("Inserisci un numero intero positivo.")
            except ValueError: print("Inserisci un numero intero valido.")
        print(f"Utilizzo di {num_cores_to_use} core per le fasi parallele.")

        # --- Definisci Percorsi Intermedi ---
        split_audio_dir = os.path.join(base_input_directory, SPLIT_FOLDER_NAME)
        preprocessed_audio_dir = os.path.join(base_input_directory, PREPROCESSED_FOLDER_NAME)
        base_output_dir = os.path.join(base_input_directory, BASE_OUTPUT_FOLDER_NAME)
        split_manifest_path = os.path.join(split_audio_dir, splitter.SPLIT_MANIFEST_FILENAME)

        # --- LOGICA PER SALTARE SPLIT/COPY (NUOVO) ---
        split_manifest_content = None
        perform_split = True # Flag per decidere se eseguire lo split
        if os.path.isdir(split_audio_dir) and os.path.isfile(split_manifest_path):
            print(f"\n--- Rilevata Directory Split Esistente ({split_audio_dir}) ---")
            print(f"--- e Manifest Esistente ({split_manifest_path}) ---")
            while True:
                skip_split = input("Vuoi saltare la fase di divisione/copia e usare i file esistenti? (s/n): ").strip().lower()
                if skip_split in ['s', 'n']: break
                else: print("Risposta non valida.")

            if skip_split == 's':
                perform_split = False # Non eseguire lo split
                print("Salto della fase di divisione/copia. Caricamento manifest esistente...")
                try:
                    with open(split_manifest_path, 'r', encoding='utf-8') as f:
                        split_manifest_content = json.load(f)
                    print("Manifest esistente caricato con successo.")
                    if not isinstance(split_manifest_content, dict) or 'files' not in split_manifest_content or not split_manifest_content['files']:
                        print("ERRORE/WARN: Il manifest esistente sembra non valido o vuoto. Verrà rigenerato.")
                        split_manifest_content = None
                        perform_split = True # Forza riesecuzione
                except Exception as e_load_manifest:
                    print(f"ERRORE durante il caricamento del manifest esistente: {e_load_manifest}")
                    print("Verrà rieseguita la fase di divisione/copia.")
                    split_manifest_content = None
                    perform_split = True # Forza riesecuzione
            else:
                print("L'utente ha scelto di rieseguire la fase di divisione/copia.")
                perform_split = True # Assicura che venga eseguito

        # --- ESEGUI SPLIT PARALLELO (solo se perform_split è True) ---
        if perform_split:
            print("\n--- Inizio Fase di Split/Copy Parallela ---")
            # Passa num_cores_to_use alla funzione di split
            split_manifest_path_new, split_manifest_content_new = splitter.split_large_audio_files(
                base_input_directory,
                split_audio_dir,
                num_workers=num_cores_to_use # Passa i core scelti
            )
            if not split_manifest_path_new or not split_manifest_content_new:
                 print("\nERRORE CRITICO: Divisione/Copia fallita.")
                 allow_sleep(); sys.exit(1)
            # Aggiorna le variabili principali se lo split è stato eseguito
            split_manifest_path = split_manifest_path_new
            split_manifest_content = split_manifest_content_new
            print("Fase di divisione/copia file completata.")
        # Ora split_manifest_path e split_manifest_content sono sicuramente popolati (o da file o da esecuzione)

        # --- ESEGUI PREPROCESSING PARALLELO dei CHUNK ---
        print("\n--- Inizio Fase di Preprocessing Parallelo dei Chunk ---")
        success_count, failure_count, valid_preprocessed_paths = preprocessor.run_parallel_preprocessing(
            split_audio_dir, # Input: directory con i chunk
            preprocessed_audio_dir, # Output: directory per chunk preprocessati
            num_workers=num_cores_to_use,
            supported_extensions=SUPPORTED_FILE_EXTENSIONS # Estensione dei chunk (es .flac)
        )
        if success_count == 0 and failure_count > 0 :
             print("\nErrore critico: Nessun chunk audio è stato preprocessato con successo.")
             allow_sleep(); sys.exit(1)
        elif not valid_preprocessed_paths:
             print("\nAttenzione: Nessun chunk audio trovato o processato durante il preprocessing.")
        print("Fase di preprocessing parallelo dei chunk completata.")

        # --- Inizializza Checkpoint (MODIFICATO per includere parametri) ---
        total_chunks_expected_calc = len(split_manifest_content.get('files', {})) if split_manifest_content else 0
        checkpoint_data = {
            "base_input_directory": base_input_directory,
            "split_audio_directory": split_audio_dir,
            "preprocessed_audio_directory": preprocessed_audio_dir,
            "base_output_directory": base_output_dir,
            "split_manifest_path": split_manifest_path,
            "original_model_choice": model_input,
            "models_to_process": selected_models,
            "num_workers_used": num_cores_to_use, # Salva core scelti
            "num_beams": num_beams_to_use, # Salva beams scelti
            "batch_size_hf": batch_size_hf_to_use, # Salva batch size scelto
            "total_chunks_expected": total_chunks_expected_calc,
            "files_processed": {},
            "current_model_processing": None,
            "last_saved": None
        }
        print("Dati nuova sessione inizializzati nel checkpoint.")
        save_checkpoint() # Salva lo stato dopo split e preprocessing

    # --- Setup Generale post-input/checkpoint ---
    # Carica il manifest dallo path salvato nel checkpoint
    split_manifest_path = checkpoint_data.get('split_manifest_path')
    split_manifest_content = {}
    if split_manifest_path and os.path.isfile(split_manifest_path):
        try:
            with open(split_manifest_path, 'r', encoding='utf-8') as f:
                split_manifest_content = json.load(f)
            #print(f"Manifest di split caricato da: {split_manifest_path}")
        except Exception as e:
            print(f"ERRORE CRITICO: Impossibile caricare il manifest '{split_manifest_path}': {e}")
            allow_sleep(); sys.exit(1)
    else:
        print("ERRORE CRITICO: Percorso del manifest non trovato nel checkpoint o file non esistente.")
        allow_sleep(); sys.exit(1)

    preprocessed_audio_dir = checkpoint_data['preprocessed_audio_directory']
    base_output_dir = checkpoint_data['base_output_directory']
    num_beams_to_use = checkpoint_data.get("num_beams", DEFAULT_NUM_BEAMS)
    batch_size_hf_to_use = checkpoint_data.get("batch_size_hf", DEFAULT_BATCH_SIZE_HF)

    # --- Main Processing Loop (Trascrizione Modelli) ---
    models_to_process = list(checkpoint_data.get('models_to_process', [])) # Crea copia per iterare sicuro
    if not models_to_process:
         print("Nessun modello specificato per l'elaborazione.")
         # ... (Pulizia checkpoint vuoto) ...
         allow_sleep(); sys.exit(0)

    try: os.makedirs(base_output_dir, exist_ok=True); #print(f"\nDirectory output base: {base_output_dir}")
    except OSError as e: print(f"!!! Errore creazione output base: {e}"); allow_sleep(); sys.exit(1)

    if torch.cuda.is_available(): print('CUDA disponibile.')
    else: print('ATTENZIONE: CUDA non disponibile.')

    
    processing_completed_normally = True # Assumi successo finché non fallisce

    try:
        while checkpoint_data.get('models_to_process'): # Continua finché ci sono modelli nella lista del checkpoint
            current_model_choice_full = checkpoint_data['models_to_process'][0] # Prendi il primo dalla lista
            current_model_folder_name = current_model_choice_full.replace('/', '_')

            print(f"\n{'='*20} Elaborazione Modello: {current_model_choice_full} ({len(checkpoint_data['models_to_process'])} rimasti) {'='*20}")

            checkpoint_data['current_model_processing'] = current_model_choice_full
            save_checkpoint()

            model_output_dir = os.path.join(base_output_dir, current_model_folder_name)
            try: os.makedirs(model_output_dir, exist_ok=True); #print(f"Directory output modello: {model_output_dir}")
            except OSError as e:
                print(f"!!! Errore creazione dir output per {current_model_folder_name}: {e}. Salto modello.");
                checkpoint_data['models_to_process'].pop(0) # Rimuovi modello fallito
                checkpoint_data['current_model_processing'] = None
                save_checkpoint()
                continue

            MODEL = None
            try:
                print(f"Caricamento modello '{current_model_choice_full}'...")
                MODEL = transcribe.load_model(current_model_choice_full)
                if MODEL is None: raise ValueError("load_model ha restituito None.")

                print(f"\nAvvio trascrizione chunk per '{current_model_choice_full}'...")
                # Chiama la funzione modificata passando il manifest e la dir preprocessata
                processed_chunk_paths_tentativi = transcribeFilesInDirectory(
                    split_manifest_content,
                    preprocessed_audio_dir,
                    MODEL,
                    current_model_choice_full,
                    model_output_dir,
                    # Passa i valori corretti
                    num_beams=num_beams_to_use,
                    batch_size_hf=batch_size_hf_to_use
                )

                # --- COMBINAZIONE (deve ricevere il manifest) ---
                print(f"\nCombinazione file per '{current_model_choice_full}'...")
                # Assicurati che combineSpeakerFiles riceva il path del manifest
                combineSpeakers.combineTranscribedSpeakerFiles(
                    model_output_dir,
                    split_manifest_path # Passa il percorso del manifest
                )
                print(f"Combinazione (o tentativo) completata per '{current_model_choice_full}'.")

                # --- Mark model as done nel checkpoint ---
                checkpoint_data['models_to_process'].pop(0) # Rimuovi il modello completato dalla lista
                checkpoint_data['current_model_processing'] = None
                save_checkpoint() # Salva lo stato aggiornato

            except Exception as e_inner_loop:
                 print(f"\n!!! Errore IRRECUPERABILE durante elaborazione modello '{current_model_choice_full}': {e_inner_loop}")
                 print("Salvataggio checkpoint e interruzione dello script.")
                 import traceback; traceback.print_exc()
                 save_checkpoint()
                 processing_completed_normally = False
                 break # Esce dal while dei modelli

            finally:
                # Cleanup GPU Memory (invariato)
                if MODEL is not None:
                    # ... (codice cleanup GPU) ...
                    print(f"Rilascio risorse modello '{current_model_choice_full}'...")
                    del MODEL; MODEL = None
                    import gc; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache(); print("Cache CUDA svuotata.")
                    print("Risorse modello rilasciate.")
                time.sleep(2)

        # Se il loop while termina perché la lista models_to_process è vuota
        if not checkpoint_data.get('models_to_process'):
            pass # processing_completed_normally rimane True se non ci sono stati errori
        else: # Se è uscito per un break da errore
             processing_completed_normally = False


    except Exception as e_outer_loop:
         print(f"\n!!! Errore IRRECUPERABILE nel ciclo di elaborazione principale: {e_outer_loop}")
         import traceback; traceback.print_exc()
         processing_completed_normally = False

    finally:
        # --- Final Cleanup (leggermente modificato) ---
        print("\n=== Fine Elaborazione Script ===")
        allow_sleep()

        if processing_completed_normally and not checkpoint_data.get('models_to_process'):
            print("Elaborazione completata normalmente per tutti i modelli richiesti.")
            # Opzionale: Rimuovi cartelle intermedie (split, preprocessed) e manifest
            # remove_intermediate = input("Rimuovere cartelle intermedie (split, preprocessed) e manifest? (s/n): ").lower()
            # if remove_intermediate == 's':
            #     try:
            #         shutil.rmtree(checkpoint_data.get('split_audio_directory','dummy_path_split'), ignore_errors=True)
            #         shutil.rmtree(checkpoint_data.get('preprocessed_audio_directory','dummy_path_prep'), ignore_errors=True)
            #         if checkpoint_data.get('split_manifest_path'): os.remove(checkpoint_data['split_manifest_path'])
            #         print("Cartelle intermedie e manifest rimossi.")
            #     except Exception as e_clean: print(f"Errore rimozione intermedie: {e_clean}")

            # Rimuovi checkpoint finale
            if os.path.exists(CHECKPOINT_FILE):
                try: os.remove(CHECKPOINT_FILE); print(f"File checkpoint '{CHECKPOINT_FILE}' rimosso.")
                except OSError as e: print(f"Warn: Impossibile rimuovere checkpoint finale: {e}")
        else:
            print("Elaborazione terminata con errori, interrotta o non tutti i modelli completati.")
            if os.path.exists(CHECKPOINT_FILE):
                print(f"Il file di checkpoint '{CHECKPOINT_FILE}' è stato conservato.")
                print(f"Modelli rimasti (se presenti): {checkpoint_data.get('models_to_process', [])}")
            else:
                 print("Il file di checkpoint non è presente (potrebbe essere stato un errore molto precoce).")


    print("\nScript terminato.")

# --- END OF transcribeCraigAudio.py (MODIFICATO v2) ---