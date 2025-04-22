# --- START OF transcribeCraigAudio.py (CORRETTO - Passa parametri per Preproc Adattivo) ---

import os, json, sys, signal, shutil, platform, time, atexit, traceback
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
import transcriptionUtils.preprocessAudioFiles as preprocessor
import transcriptionUtils.splitAudio as splitter
try: from transcriptionUtils.preprocessAudioFiles import preprocessing_pool_global_ref
except ImportError: preprocessing_pool_global_ref = None
try: from transcriptionUtils.splitAudio import analysis_pool_global_ref, splitting_pool_global_ref
except ImportError: analysis_pool_global_ref = None; splitting_pool_global_ref = None
import torch; import multiprocessing as mp # type: ignore


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
            import pydbus # type: ignore
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

# --- Valori di Default Parametri ---
DEFAULT_NUM_BEAMS = 1; DEFAULT_BATCH_SIZE_HF = 16
TARGET_AVG_DBFS = -18.0; BOOST_THRESHOLD_DB = 6

# Global variable for checkpointing
checkpoint_data = {}

# --- Valori di Default per Parametri Interattivi (NUOVO) ---
DEFAULT_NUM_BEAMS = 1
DEFAULT_BATCH_SIZE_HF = 16

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

# --- Gestione Segnali (Termina Pool) ---
def signal_handler(sig, frame):
    print("\n*** Interruzione rilevata! Pulizia... ***")
    print(">>> Terminazione Pool...")
    pools_terminated = 0
    pools_to_check = {"Analisi": analysis_pool_global_ref, "Splitting": splitting_pool_global_ref, "Preprocessing": preprocessing_pool_global_ref}
    for name, pool_ref in pools_to_check.items():
        if pool_ref is not None: # ... (logica terminate/join pool) ...
             print(f"    Terminating pool '{name}'...")
             try: pool_ref.terminate(); pool_ref.join(timeout=5); pools_terminated += 1
             except Exception as e: print(f"    Error terminating pool '{name}': {e}")
    if pools_terminated == 0: print("    Nessun pool attivo.")
    allow_sleep(); save_checkpoint(); print("Checkpoint salvato. Uscita."); os._exit(1)

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
    try:
        if platform.system() != "Linux": mp.set_start_method('spawn', force=True)
    except RuntimeError: pass # Già impostato
    except AttributeError: pass # Vecchio Python
    prevent_sleep()

    # --- Variabili per i parametri (NUOVO) ---
    num_beams_to_use = DEFAULT_NUM_BEAMS
    batch_size_hf_to_use = DEFAULT_BATCH_SIZE_HF
    # Variabili per percorsi e contenuti che verranno determinati
    base_input_directory = None
    split_audio_dir = None
    preprocessed_audio_dir = None
    base_output_dir = None
    split_manifest_path = None
    selected_models = []
    model_input = "" # Per salvare l'input originale dell'utente sui modelli

    # --- Load Checkpoint or Initialize ---
    checkpoint_found = load_checkpoint()
    perform_split_and_preprocess = True # Assumi di dover fare tutto per una nuova sessione
    split_manifest_content = None

    if checkpoint_found:
        # ... (Logica ripresa/nuova sessione invariata, ma mostra info aggiuntive) ...
        print(f"Ripresa da checkpoint.")
        # Recupera parametri da checkpoint, se presenti
        base_input_directory = checkpoint_data.get("base_input_directory")
        split_audio_dir = checkpoint_data.get("split_audio_directory")
        preprocessed_audio_dir = checkpoint_data.get("preprocessed_audio_directory")
        base_output_dir = checkpoint_data.get("base_output_directory")
        split_manifest_path = checkpoint_data.get("split_manifest_path")
        checkpoint_data.get("num_workers_used", os.cpu_count()) # Recupera core usati
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
            checkpoint_data = {}; checkpoint_found = False; 
            perform_split_and_preprocess = True
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
             perform_split_and_preprocess = False
             # Verifica esistenza directory e manifest dal checkpoint
             if not os.path.isdir(checkpoint_data.get('split_audio_directory','')) or \
                not os.path.isdir(checkpoint_data.get('preprocessed_audio_directory','')) or \
                not os.path.isfile(checkpoint_data.get('split_manifest_path','')):
                 print("ERRORE: Directory intermedie o manifest dal checkpoint non trovati. Impossibile riprendere.")
                 print("Rimuovere il checkpoint ed eseguire di nuovo lo script.")
                 allow_sleep(); sys.exit(1)


    # --- Get User Input if NO valid checkpoint exists or user chose NOT to resume ---
    if not checkpoint_found or perform_split_and_preprocess:
        if not checkpoint_found:
          print("\n--- Configurazione Nuova Sessione ---")
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

           # --- Definisci Percorsi (anche se resume='n', recupera da checkpoint o ridefinisci) ---
          if not base_input_directory: base_input_directory = checkpoint_data.get("base_input_directory") # Fallback
          if not base_input_directory: print("ERRORE CRITICO: base_input_directory non definito!"); allow_sleep(); sys.exit(1)
          split_audio_dir = os.path.join(base_input_directory, SPLIT_FOLDER_NAME)
          preprocessed_audio_dir = os.path.join(base_input_directory, PREPROCESSED_FOLDER_NAME)
          base_output_dir = os.path.join(base_input_directory, BASE_OUTPUT_FOLDER_NAME) # Definito qui
          split_manifest_path = os.path.join(split_audio_dir, splitter.SPLIT_MANIFEST_FILENAME)

          # --- LOGICA PER SALTARE SPLIT/COPY (NUOVO) ---
          split_manifest_content = None
          perform_split = True # Flag per decidere se eseguire lo split
          if not checkpoint_found and os.path.isdir(split_audio_dir) and os.path.isfile(split_manifest_path):
            print(f"\n--- Rilevata Directory Split Esistente ({split_audio_dir}) ---")
            print(f"--- e Manifest Esistente ({split_manifest_path}) ---")
            while True:
                skip_split = input("Vuoi saltare la fase di divisione/copia e preprocessing e usare i file esistenti? (s/n): ").strip().lower()
                if skip_split in ['s', 'n']: break
                else: print("Risposta non valida.")

            if skip_split == 's':
                perform_split_and_preprocess = False # Non eseguire lo split
                print("Salto della fase di divisione/copia e preprocessing. Caricamento manifest esistente...")
                try:
                    with open(split_manifest_path, 'r', encoding='utf-8') as f:
                        split_manifest_content = json.load(f)
                    print("Manifest esistente caricato con successo.")
                    if not isinstance(split_manifest_content, dict) or 'files' not in split_manifest_content or not split_manifest_content['files']: 
                        print("ERRORE/WARN: Il manifest esistente sembra non valido o vuoto. Verrà rigenerato.")
                        split_manifest_content = None
                        perform_split_and_preprocess = True # Forza riesecuzione
                except Exception as e_load_manifest:
                    print(f"ERRORE durante il caricamento del manifest esistente: {e_load_manifest}")
                    print("Verrà rieseguita la fase di divisione/copia.")
                    split_manifest_content = None
                    perform_split_and_preprocess  = True # Forza riesecuzione
            else:
                print("L'utente ha scelto di rieseguire la fase di divisione/copia.")
                perform_split_and_preprocess  = True # Assicura che venga eseguito

        # --- ESEGUI SPLIT/PREPROCESSING (se perform_split_and_preprocess è True) ---
        if perform_split_and_preprocess:
            # FASE 1: SPLIT (che include analisi)
            split_manifest_path_new, split_manifest_content_new = splitter.split_large_audio_files(
                base_input_directory, split_audio_dir, num_workers=num_cores_to_use
                # Passa qui altri parametri se resi configurabili
            )
            if not split_manifest_path_new or not split_manifest_content_new: print("\nERRORE CRITICO: Split fallito."); allow_sleep(); sys.exit(1)
            split_manifest_path = split_manifest_path_new; split_manifest_content = split_manifest_content_new

            # FASE 2: PREPROCESSING
            # Recupera il target LUFS calcolato e salvato nel manifest
            effective_target_lufs = split_manifest_content.get("effective_target_lufs_for_norm", splitter.FALLBACK_TARGET_LUFS)
            print("\n--- Inizio Fase di Preprocessing Parallelo dei Chunk (con Boost Selettivo via Manifest) ---")
            success_count, failure_count, valid_preprocessed_paths = preprocessor.run_parallel_preprocessing(
                input_chunk_dir=split_audio_dir, preprocessed_output_dir=preprocessed_audio_dir,
                num_workers=num_cores_to_use, manifest_path=split_manifest_path,
                #target_avg_dbfs=effective_target_lufs, # Usa target LUFS calcolato
                boost_threshold_db=BOOST_THRESHOLD_DB,
                supported_extensions=SUPPORTED_FILE_EXTENSIONS )
            if success_count == 0 and failure_count > 0: print("\nERRORE: Nessun chunk preprocessato."); allow_sleep(); sys.exit(1)
            elif not valid_preprocessed_paths: print("\nWarn: Nessun chunk preprocessato valido.")

        # --- Crea/Aggiorna Checkpoint DOPO split/preprocess (se eseguiti) o alla prima run ---
        total_chunks_expected_calc = len(split_manifest_content.get('files', {})) if split_manifest_content else 0
        # Se era una vera nuova sessione, INIZIALIZZA checkpoint_data da zero
        if not checkpoint_found:
            checkpoint_data = {
                "base_input_directory": base_input_directory,
                "split_audio_directory": split_audio_dir,
                "preprocessed_audio_directory": preprocessed_audio_dir,
                "base_output_directory": base_output_dir, # <-- ASSICURATI SIA QUI
                "split_manifest_path": split_manifest_path,
                "original_model_choice": model_input, # Dall'input utente
                "models_to_process": selected_models, # Dall'input utente
                "num_workers_used": num_cores_to_use, # Dall'input utente
                "num_beams": num_beams_to_use,       # Dall'input utente
                "batch_size_hf": batch_size_hf_to_use, # Dall'input utente
                "total_chunks_expected": total_chunks_expected_calc,
                "files_processed": {}, # Inizia vuoto
                "current_model_processing": None,
                "last_saved": None
            }
            print("Dati nuova sessione inizializzati nel checkpoint.")
        else: # Altrimenti (resume='n' o skip_split='n'), aggiorna solo campi specifici
             checkpoint_data.update({
                 "base_input_directory": base_input_directory,
                 "split_audio_directory": split_audio_dir,
                 "preprocessed_audio_directory": preprocessed_audio_dir,
                 "base_output_directory": base_output_dir, # <-- AGGIORNA ANCHE QUI
                 "split_manifest_path": split_manifest_path,
                 "num_workers_used": num_cores_to_use, # Aggiorna se è stato richiesto di nuovo
                 "num_beams": num_beams_to_use, # Aggiorna se è stato richiesto di nuovo
                 "batch_size_hf": batch_size_hf_to_use, # Aggiorna se è stato richiesto di nuovo
                 "total_chunks_expected": total_chunks_expected_calc
                 # Non toccare models_to_process, files_processed, current_model se resume=n
             })
             print("Aggiornati percorsi/parametri/chunk totali nel checkpoint esistente.")
        save_checkpoint() # Salva lo stato COMPLETO

    # --- Setup Generale post-input/checkpoint ---
    # Assicurati che tutti i path necessari siano definiti prima del loop principale
    if not all([ 'base_input_directory' in checkpoint_data, 'split_audio_directory' in checkpoint_data,
                 'preprocessed_audio_directory' in checkpoint_data, 'base_output_dir' in checkpoint_data,
                 'split_manifest_path' in checkpoint_data]):
         print("ERRORE CRITICO: Dati essenziali mancanti nel checkpoint prima del loop modelli.")
         allow_sleep(); sys.exit(1)

    split_manifest_path = checkpoint_data.get('split_manifest_path')
    split_manifest_content = {}; # Ricarica sempre per avere l'ultima versione
    if split_manifest_path and os.path.isfile(split_manifest_path):
        try:
            with open(split_manifest_path, 'r', encoding='utf-8') as f: split_manifest_content = json.load(f)
        except Exception as e: print(f"ERRORE CRITICO caricando manifest prima del loop: {e}"); allow_sleep(); sys.exit(1)
    else: print("ERRORE CRITICO: Manifest non trovato prima del loop."); allow_sleep(); sys.exit(1)
    preprocessed_audio_dir = checkpoint_data['preprocessed_audio_directory']
    base_output_dir = checkpoint_data['base_output_directory']
    num_beams_to_use = checkpoint_data.get("num_beams", DEFAULT_NUM_BEAMS)
    batch_size_hf_to_use = checkpoint_data.get("batch_size_hf", DEFAULT_BATCH_SIZE_HF)

    # --- Main Processing Loop ---
    if torch.cuda.is_available(): print('\nCUDA disponibile.')
    else: print('\nATTENZIONE: CUDA non disponibile.')
    processing_completed_normally = True

    try:
        os.makedirs(base_output_dir, exist_ok=True) # Assicurati esista
        while checkpoint_data.get('models_to_process'):
            current_model_choice_full = checkpoint_data['models_to_process'][0]
            current_model_folder_name = current_model_choice_full.replace('/', '_').replace('\\', '_') # Sanitize folder name
            print(f"\n{'='*20} Elaborazione Modello: {current_model_choice_full} ({len(checkpoint_data['models_to_process'])} rimasti) {'='*20}")
            checkpoint_data['current_model_processing'] = current_model_choice_full; save_checkpoint()
            model_output_dir = os.path.join(base_output_dir, current_model_folder_name)
            try: os.makedirs(model_output_dir, exist_ok=True)
            except OSError as e: print(f"!!! Errore dir output: {e}. Salto."); checkpoint_data['models_to_process'].pop(0); checkpoint_data['current_model_processing'] = None; save_checkpoint(); continue

            MODEL = None
            try:
                print(f"Caricamento modello '{current_model_choice_full}'...")
                MODEL = transcribe.load_model(current_model_choice_full);
                if MODEL is None: raise ValueError("load_model ha restituito None.")
                print(f"Avvio trascrizione chunk (Beams={num_beams_to_use}, Batch={batch_size_hf_to_use})...")
                processed_chunk_paths_tentativi = transcribeFilesInDirectory(
                    split_manifest_content, preprocessed_audio_dir, MODEL,
                    current_model_choice_full, model_output_dir,
                    num_beams=num_beams_to_use, batch_size_hf=batch_size_hf_to_use
                )
                print(f"\nCombinazione file per '{current_model_choice_full}'...")
                combineSpeakers.combineTranscribedSpeakerFiles(model_output_dir, split_manifest_path)
                print(f"Combinazione completata per '{current_model_choice_full}'.")
                # Rimuovi modello dalla lista *solo se completato con successo*
                checkpoint_data['models_to_process'].pop(0); checkpoint_data['current_model_processing'] = None; save_checkpoint()
            except Exception as e_inner_loop:
                 print(f"\n!!! Errore IRRECUPERABILE durante elaborazione modello '{current_model_choice_full}': {e_inner_loop}"); traceback.print_exc(); save_checkpoint(); processing_completed_normally = False; break # Esce dal loop modelli
            finally: # Cleanup GPU
                if MODEL is not None:
                    print(f"Rilascio modello '{current_model_choice_full}'..."); del MODEL; MODEL = None; import gc; gc.collect();
                    if torch.cuda.is_available(): torch.cuda.empty_cache();
                    print("Risorse modello rilasciate.")
                time.sleep(1) # Breve pausa

        # Controlla se il loop è finito perché non ci sono più modelli
        if not checkpoint_data.get('models_to_process') and processing_completed_normally:
            print("\nTutti i modelli richiesti sono stati elaborati.")
        elif processing_completed_normally: # Loop finito ma lista non vuota? Strano.
             print("WARN: Il loop modelli è terminato ma la lista non è vuota nel checkpoint.")
             processing_completed_normally = False
        # else: processing_completed_normally è già False per errore/interruzione

    except KeyboardInterrupt: print("\nInterruzione manuale..."); processing_completed_normally = False
    except Exception as e_outer_loop: print(f"\n!!! Errore Esterno: {e_outer_loop}"); traceback.print_exc(); processing_completed_normally = False; save_checkpoint()

    # --- Final Cleanup ---
    finally:
        print("\n=== Fine Elaborazione Script ===")
        allow_sleep() # Assicurati sia chiamato
        if processing_completed_normally and not checkpoint_data.get('models_to_process'):
            print("Elaborazione completata normalmente.")
            # --- Rimozione opzionale ---
            while True:
                 remove_intermediate = input("Rimuovere cartelle intermedie (split, preprocessed) e manifest? (s/n): ").lower().strip()
                 if remove_intermediate in ['s','n']: break
                 else: print("Risposta non valida.")
            if remove_intermediate == 's':
                 print("Rimozione cartelle intermedie...")
                 deleted_count = 0
                 try: shutil.rmtree(checkpoint_data.get('split_audio_directory','dummy1'), ignore_errors=True); deleted_count+=1
                 except Exception as e1: print(f" Errore rimozione split: {e1}")
                 try: shutil.rmtree(checkpoint_data.get('preprocessed_audio_directory','dummy2'), ignore_errors=True); deleted_count+=1
                 except Exception as e2: print(f" Errore rimozione preprocessed: {e2}")
                 try:
                      manifest_to_remove = checkpoint_data.get('split_manifest_path')
                      if manifest_to_remove and os.path.isfile(manifest_to_remove): os.remove(manifest_to_remove); deleted_count+=1
                 except Exception as e3: print(f" Errore rimozione manifest: {e3}")
                 print(f"Rimossi {deleted_count} elementi intermedi.")
            # --- Fine rimozione opzionale ---
            if os.path.exists(CHECKPOINT_FILE):
                 try: os.remove(CHECKPOINT_FILE); print("Checkpoint finale rimosso.")
                 except OSError as e: print(f"Warn: Impossibile rimuovere checkpoint: {e}")
        else:
            print("Elaborazione terminata con errori/interruzione o non completata.")
            if os.path.exists(CHECKPOINT_FILE): print(f"Checkpoint '{CHECKPOINT_FILE}' conservato.")
    print("\nScript terminato.")

# --- END OF transcribeCraigAudio.py (CORRETTO per Approccio B e Terminazione Pool) ---