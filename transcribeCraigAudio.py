# --- START OF transcribeCraigAudio.py (MODIFICATO) ---

import os
import json
import sys
import signal
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
# NUOVA IMPORTAZIONE PER IL PREPROCESSING PARALLELO
import transcriptionUtils.preprocessAudioFiles as preprocessor
import torch
import shutil
import platform
import time
import atexit # Assicurati sia importato

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


SUPPORTED_FILE_EXTENSIONS = (".flac",) # Usiamo una tupla
CHECKPOINT_FILE = "transcription_checkpoint.json"
BASE_OUTPUT_FOLDER_NAME = "transcription_output"
PREPROCESSED_FOLDER_NAME = "audio_preprocessed"

# Global variable for checkpointing
checkpoint_data = {}

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

# --- RIMOSSA FUNZIONE preprocess_all_audio (spostata nel modulo) ---

# --- MODIFICATA Funzione transcribeFilesInDirectory ---
# Ora itera sui file *originali*, controlla se il preprocessato esiste, e poi trascrive.
def transcribeFilesInDirectory(original_base_dir: str,
                               preprocessed_dir: str,
                               model,
                               model_choice: str,
                               output_dir: str):
    """
    Trascrive i file audio basandosi sulla lista di file originali.
    Controlla l'esistenza del file preprocessato corrispondente prima di procedere.
    Usa il checkpoint globale basato sui nomi file originali.
    """
    global checkpoint_data
    filesTranscribed_original_paths = [] # Lista dei percorsi originali tentati (successo o fallimento)
    processed_in_this_session = 0

    if not checkpoint_data:
        print("ERRORE: Dati checkpoint non inizializzati.")
        return []
    if 'base_input_directory' not in checkpoint_data:
         print("ERRORE: 'base_input_directory' non trovato nel checkpoint.")
         # Fallback: prova a usare original_base_dir passato come argomento se disponibile
         if not original_base_dir or not os.path.isdir(original_base_dir):
              print("ERRORE: Impossibile determinare la directory di input originale.")
              return []
         # Non ideale, ma permette di continuare se l'argomento è corretto
         checkpoint_data['base_input_directory'] = original_base_dir

    # Modello specifico per cui recuperare i file già processati
    already_processed_originals = checkpoint_data.setdefault('files_processed', {}).setdefault(model_choice, [])

    try:
        # Elenca i file *originali* supportati
        original_files = sorted([
            f for f in os.listdir(original_base_dir)
            if os.path.isfile(os.path.join(original_base_dir, f)) and f.lower().endswith(SUPPORTED_FILE_EXTENSIONS)
        ])

        if not original_files:
            print(f"Nessun file con estensioni supportate trovato in {original_base_dir}")
            return []

        print(f"Trovati {len(original_files)} file audio originali da considerare per la trascrizione.")

        for file_index, original_filename in enumerate(original_files):
            original_path = os.path.join(original_base_dir, original_filename)
            preprocessed_path = os.path.join(preprocessed_dir, original_filename) # Percorso atteso preprocessato

            # 1. Controlla se già processato secondo il checkpoint
            if original_path in already_processed_originals:
                print(f"({file_index+1}/{len(original_files)}) Skipping {original_filename} (già processato per '{model_choice}' secondo checkpoint)")
                filesTranscribed_original_paths.append(original_path)
                continue

            # 2. Controlla se il file preprocessato esiste
            if not os.path.exists(preprocessed_path):
                print(f"({file_index+1}/{len(original_files)}) Skipping {original_filename}: File preprocessato non trovato in '{preprocessed_dir}'. Preprocessing fallito o file non generato.")
                # MARCA COME TENTATO NEL CHECKPOINT per non riprovare inutilmente
                # Questo evita loop se il preprocessing fallisce consistentemente per un file.
                checkpoint_data['files_processed'].setdefault(model_choice, []).append(original_path)
                filesTranscribed_original_paths.append(original_path) # Aggiungi alla lista locale
                processed_in_this_session += 1 # Conta come tentativo (fallito prima della trascrizione)
                save_checkpoint()
                continue # Passa al prossimo file originale

            # 3. Se non già processato e il file preprocessato esiste, procedi con la trascrizione
            print(f"\n--- ({file_index+1}/{len(original_files)}) Trascrizione '{model_choice}' per: {original_filename} (da preprocessato: {os.path.basename(preprocessed_path)}) ---")

            # Chiama la trascrizione usando il file preprocessato
            # transcribeAudioFile ritorna il percorso del file .txt finale o del file di errore/log
            output_file_path = transcribe.transcribeAudioFile(preprocessed_path, model, model_choice, output_dir)

            # --- LOGICA DI SUCCESSO/FALLIMENTO (come prima, basata su output_file_path) ---
            # Aggiungi l'original_path al checkpoint *indipendentemente* dal successo o fallimento
            # della trascrizione, perché abbiamo *tentato* di processarlo.
            checkpoint_data['files_processed'].setdefault(model_choice, []).append(original_path)
            filesTranscribed_original_paths.append(original_path) # Aggiungi alla lista locale dei tentati
            processed_in_this_session += 1

            if output_file_path and os.path.exists(output_file_path) and "FAILED" not in os.path.basename(output_file_path).upper():
                print(f"Trascrizione completata con successo per {original_filename}.")
                # (Checkpoint già aggiornato sopra)
            elif output_file_path and os.path.exists(output_file_path):
                print(f"ERRORE o INFO durante trascrizione di {original_filename}. File di output: {os.path.basename(output_file_path)}")
                # (Checkpoint già aggiornato sopra)
            else:
                print(f"ERRORE grave: Trascrizione per {original_filename} non ha prodotto un file finale valido o ha restituito None.")
                # (Checkpoint già aggiornato sopra)

            save_checkpoint() # Salva checkpoint dopo ogni tentativo

    except Exception as e:
        print(f"Errore imprevisto in transcribeFilesInDirectory: {e}")
        import traceback
        traceback.print_exc() # Stampa traceback per debug
        save_checkpoint() # Salva comunque lo stato attuale
        # Ritorna quello che è stato processato finora
        return filesTranscribed_original_paths

    print(f"\n--- Completata sessione di trascrizione per modello '{model_choice}' ---")
    print(f"File originali considerati/tentati in totale per '{model_choice}': {len(filesTranscribed_original_paths)}")
    print(f"File tentati (successo o fallimento) in questa sessione per '{model_choice}': {processed_in_this_session}")
    return filesTranscribed_original_paths


# ============================================================================
# --- Main Execution Logic ---
# ============================================================================

if __name__ == "__main__":

    prevent_sleep() # Attiva anti-standby

    # --- Load Checkpoint or Initialize ---
    checkpoint_found = load_checkpoint()

    if checkpoint_found:
        # ... (Logica ripresa/nuova sessione invariata) ...
        print(f"Ripresa da checkpoint. Directory input originale: {checkpoint_data.get('base_input_directory', 'N/A')}")
        print(f"Modelli originariamente scelti: {checkpoint_data.get('original_model_choice', 'N/A')}") # Potrebbe essere la lista ora
        print(f"Modelli rimasti da processare: {checkpoint_data.get('models_to_process', [])}")

        while True:
             resume = input("Vuoi riprendere la sessione precedente? (s/n): ").strip().lower()
             if resume in ['s', 'n']:
                  break
             else:
                  print("Risposta non valida. Inserisci 's' per sì o 'n' per no.")

        if resume == 'n':
            print("Avvio nuova sessione. RIMOZIONE del checkpoint precedente...")
            try:
                if os.path.exists(CHECKPOINT_FILE):
                    os.remove(CHECKPOINT_FILE)
                    print(f"File di checkpoint '{CHECKPOINT_FILE}' rimosso.")
                checkpoint_data = {} # Resetta i dati in memoria
                checkpoint_found = False # Segna che non stiamo più usando un checkpoint caricato
            except OSError as e:
                print(f"ATTENZIONE: Impossibile rimuovere il file di checkpoint '{CHECKPOINT_FILE}': {e}")
                print("Lo script continuerà come nuova sessione, ma il vecchio file potrebbe rimanere.")
                checkpoint_data = {}
                checkpoint_found = False
        else:
             print("Ripresa della sessione dal checkpoint.")
             # checkpoint_data è già popolato da load_checkpoint()

    # --- Get User Input if NO valid checkpoint exists or user chose NOT to resume ---
    if not checkpoint_found:
        print("\n--- Nuova Sessione ---")

        # MODIFICATO: Selezione Modelli Flessibile
        ALL_AVAILABLE_MODELS = [
            "whisper-medium", "whisper-largev2", "whisper-largev3",
            "hf_whisper-medium", "hf_whisper-largev2", "hf_whisper-largev3",
            "whispy_italian"
        ]
        while True:
            print("\nModelli di trascrizione disponibili:")
            for i, model_name in enumerate(ALL_AVAILABLE_MODELS):
                print(f"  {i+1}. {model_name}")
            print("\nInserisci i numeri o i nomi dei modelli da eseguire, separati da virgola.")
            print("Oppure scrivi 'tutti' per eseguirli tutti in sequenza.")
            print("Esempio: '1, 3' oppure 'whisper-medium, hf_whisper-largev3' oppure 'tutti'")

            model_input = input("Scegli modelli: ").strip()
            selected_models = []
            invalid_selection = False

            if model_input.lower() == 'tutti':
                selected_models = ALL_AVAILABLE_MODELS
                print(f"Selezionati tutti i modelli: {', '.join(selected_models)}")
                break # Esce dal while della selezione modelli

            # Prova a processare la selezione (numeri o nomi)
            parts = [p.strip() for p in model_input.split(',') if p.strip()]
            if not parts:
                print("Selezione vuota. Riprova.")
                continue

            for part in parts:
                found = False
                # Prova a matchare per numero
                if part.isdigit():
                    try:
                        index = int(part) - 1
                        if 0 <= index < len(ALL_AVAILABLE_MODELS):
                            selected_model = ALL_AVAILABLE_MODELS[index]
                            if selected_model not in selected_models: # Evita duplicati
                                selected_models.append(selected_model)
                            found = True
                        else:
                            print(f"Errore: Numero '{part}' non valido.")
                            invalid_selection = True; break
                    except ValueError: # Non dovrebbe succedere con isdigit ma per sicurezza
                        pass
                # Se non è un numero valido o non trovato, prova a matchare per nome
                if not found:
                    model_name_lower = part.lower()
                    match_found = False
                    for available_model in ALL_AVAILABLE_MODELS:
                        if available_model.lower() == model_name_lower:
                             if available_model not in selected_models:
                                 selected_models.append(available_model)
                             match_found = True
                             break # Trovato match per nome
                    if not match_found:
                         print(f"Errore: Modello '{part}' non riconosciuto.")
                         invalid_selection = True; break
                    found = match_found # Aggiorna found se il nome è stato trovato

                if invalid_selection: break # Esce dal loop for part

            if not invalid_selection and selected_models:
                print(f"Modelli selezionati da eseguire (in ordine): {', '.join(selected_models)}")
                break # Esce dal while della selezione modelli
            elif not selected_models and not invalid_selection:
                 print("Nessun modello valido selezionato. Riprova.")
            else: # C'è stato un errore
                 print("Selezione non valida. Correggi e riprova.")
                 # Loop while continua


        # Chiedi directory input
        while True:
            base_input_directory = input('Inserisci directory audio originale: ').strip()
            if os.path.isdir(base_input_directory): break
            else: print(f"Errore: '{base_input_directory}' non è una directory valida.")

        # Chiedi numero core per preprocessing
        default_cores = 8
        while True:
            try:
                core_input = input(f"Numero di core CPU per il preprocessing (default {default_cores}, premi Invio per default): ").strip()
                if not core_input:
                    num_cores_to_use = default_cores
                    break
                num_cores_to_use = int(core_input)
                if num_cores_to_use > 0:
                    break
                else:
                    print("Inserisci un numero intero positivo.")
            except ValueError:
                print("Inserisci un numero intero valido.")

        # Inizializza checkpoint_data per la nuova sessione
        checkpoint_data = {
            "base_input_directory": base_input_directory,
            "original_model_choice": model_input, # Salva l'input utente originale
            "models_to_process": selected_models, # Lista dei modelli validati
            "num_preprocessing_workers": num_cores_to_use, # Salva numero core scelto
            "files_processed": {}, # Dizionario per tracciare file per modello
            "preprocessing_output_paths": [], # Lista dei file preprocessati con successo
            "current_model_processing": None,
            "last_saved": None
        }
        print("Dati nuova sessione inizializzati.")
        # Salviamo subito il checkpoint iniziale per non perdere la configurazione
        save_checkpoint()

    # --- Setup Directories (invariato) ---
    base_input_dir = checkpoint_data['base_input_directory']
    preprocessed_audio_dir = os.path.join(base_input_dir, PREPROCESSED_FOLDER_NAME)
    base_output_dir = os.path.join(base_input_dir, BASE_OUTPUT_FOLDER_NAME)
    num_cores = checkpoint_data.get('num_preprocessing_workers', 8) # Recupera o usa default

    # --- ESEGUI PREPROCESSING PARALLELO ---
    # Controlla se il preprocessing è già stato completato (guardando la lista nel checkpoint)
    # Questo evita di rieseguire il preprocessing se lo script viene riavviato dopo questa fase.
    if not checkpoint_data.get('preprocessing_output_paths'):
        print("\n--- Inizio Fase di Preprocessing ---")
        # Chiama la nuova funzione parallela
        success_count, failure_count, valid_output_paths = preprocessor.run_parallel_preprocessing(
            base_input_dir,
            preprocessed_audio_dir,
            num_workers=num_cores,
            supported_extensions=SUPPORTED_FILE_EXTENSIONS
        )

        # Salva i risultati del preprocessing nel checkpoint
        checkpoint_data['preprocessing_output_paths'] = valid_output_paths
        save_checkpoint()

        # Controlla se almeno un file è stato processato con successo
        if success_count == 0 and failure_count > 0 :
             print("\nErrore critico: Nessun file audio è stato preprocessato con successo.")
             allow_sleep(); sys.exit(1)
        elif not valid_output_paths:
             print("\nAttenzione: Nessun file audio trovato o processato durante il preprocessing.")
             # Potrebbe non essere un errore se la cartella era vuota, quindi continuiamo
             # ma la trascrizione probabilmente non farà nulla.
    else:
        print("\n--- Skipping Preprocessing ---")
        print("Trovati risultati di preprocessing nel checkpoint.")
        print(f"Numero di file preprocessati precedentemente: {len(checkpoint_data['preprocessing_output_paths'])}")


    # --- Main Processing Loop (Trascrizione) ---
    models_to_process = checkpoint_data.get('models_to_process', [])
    if not models_to_process:
         print("Nessun modello specificato per l'elaborazione nel checkpoint o selezione utente.")
         if os.path.exists(CHECKPOINT_FILE):
             try: os.remove(CHECKPOINT_FILE); print("Checkpoint vuoto rimosso.")
             except OSError as e: print(f"Warn: Impossibile rimuovere checkpoint vuoto: {e}")
         allow_sleep(); sys.exit(0)

    try: os.makedirs(base_output_dir, exist_ok=True); print(f"\nDirectory output base: {base_output_dir}")
    except OSError as e: print(f"!!! Errore creazione output base: {e}"); allow_sleep(); sys.exit(1)

    if torch.cuda.is_available(): print('CUDA disponibile.')
    else: print('ATTENZIONE: CUDA non disponibile.')

    current_model_index = 0
    processing_completed_normally = False
    try:
        while current_model_index < len(models_to_process):
            current_model_choice_full = models_to_process[current_model_index]
            current_model_folder_name = current_model_choice_full.replace('/', '_') # Sicuro per nomi cartella

            print(f"\n{'='*20} Elaborazione Modello {current_model_index + 1}/{len(models_to_process)}: {current_model_choice_full} {'='*20}")

            checkpoint_data['current_model_processing'] = current_model_choice_full
            save_checkpoint() # Salva quale modello stiamo per iniziare

            model_output_dir = os.path.join(base_output_dir, current_model_folder_name)
            try: os.makedirs(model_output_dir, exist_ok=True); print(f"Directory output modello: {model_output_dir}")
            except OSError as e:
                print(f"!!! Errore creazione dir output per {current_model_folder_name}: {e}. Salto modello.");
                # Rimuovi modello dalla lista e salva checkpoint aggiornato
                checkpoint_data['models_to_process'] = models_to_process[current_model_index + 1:]
                checkpoint_data['current_model_processing'] = None
                save_checkpoint()
                current_model_index += 1; continue # Passa al prossimo modello

            MODEL = None
            try:
                print(f"Caricamento modello '{current_model_choice_full}'...")
                MODEL = transcribe.load_model(current_model_choice_full)
                if MODEL is None: raise ValueError("load_model ha restituito None.")

                print(f"\nAvvio trascrizione per '{current_model_choice_full}'...")
                # Chiama la funzione modificata passando la dir originale e quella preprocessata
                processed_original_paths_tentativi = transcribeFilesInDirectory(
                    base_input_dir, # Directory originale
                    preprocessed_audio_dir, # Directory con audio preprocessato
                    MODEL,
                    current_model_choice_full,
                    model_output_dir
                )

                # --- CONTROLLO SUCCESSO e COMBINAZIONE (Logica leggermente adattata) ---
                # Controlliamo se ci sono stati tentativi (anche falliti) per questo modello
                processed_files_for_model_in_ckpt = checkpoint_data.get('files_processed', {}).get(current_model_choice_full, [])

                # Se non ci sono file nel checkpoint per questo modello, significa che
                # o non c'erano file originali, o tutti i preprocessing sono falliti,
                # o transcribeFilesInDirectory è fallita subito.
                if not processed_files_for_model_in_ckpt:
                     print(f"Nessun file risulta processato (o tentato) per '{current_model_choice_full}' secondo il checkpoint.")
                     print("Possibili cause: nessun file .flac originale, fallimento totale del preprocessing, errore iniziale nella trascrizione.")
                     # In questo caso, non c'è nulla da combinare.
                else:
                    # Se ci sono file nel checkpoint, tentiamo la combinazione.
                    # La funzione di combinazione è già robusta e gestirà l'assenza
                    # di file .txt validi se le trascrizioni sono fallite.
                    print(f"\nCombinazione file per '{current_model_choice_full}'...")
                    combineSpeakers.combineTranscribedSpeakerFiles(model_output_dir)
                    print(f"Combinazione (o tentativo) completata per '{current_model_choice_full}'.")

                # --- Mark model as done nel checkpoint ---
                # Rimuovi il modello corrente dalla lista 'models_to_process'
                checkpoint_data['models_to_process'] = models_to_process[current_model_index + 1:]
                checkpoint_data['current_model_processing'] = None # Nessun modello attivo
                save_checkpoint() # Salva lo stato aggiornato
                current_model_index += 1 # Passa all'indice del prossimo modello

            except Exception as e_inner_loop:
                 print(f"\n!!! Errore IRRECUPERABILE durante elaborazione modello '{current_model_choice_full}': {e_inner_loop}")
                 print("Salvataggio checkpoint e interruzione dello script.")
                 import traceback
                 traceback.print_exc()
                 save_checkpoint()
                 # Non sollevare l'eccezione di nuovo per permettere il finally, ma esci
                 processing_completed_normally = False
                 break # Esce dal while dei modelli

            finally:
                # Cleanup GPU Memory
                if MODEL is not None:
                    print(f"Rilascio risorse modello '{current_model_choice_full}'...")
                    # Per modelli HF (pipeline): può bastare del MODEL
                    # Per modelli OpenAI: può bastare del MODEL
                    # Forzare garbage collection potrebbe aiutare
                    del MODEL
                    MODEL = None
                    import gc
                    gc.collect() # Forza garbage collection
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("Cache CUDA svuotata.")
                    print("Risorse modello rilasciate.")
                time.sleep(2) # Pausa leggermente più lunga tra modelli

        # Se il loop while è terminato senza 'break' dovuto a errori irrecuperabili
        if current_model_index == len(models_to_process):
             processing_completed_normally = True

    except Exception as e_outer_loop:
         print(f"\n!!! Errore IRRECUPERABILE nel ciclo di elaborazione principale: {e_outer_loop}")
         import traceback
         traceback.print_exc()
         processing_completed_normally = False
         # Il checkpoint dovrebbe essere stato salvato nel blocco finally o nell'handler segnali

    finally:
        # --- Final Cleanup ---
        print("\n=== Fine Elaborazione Script ===")
        allow_sleep() # Disattiva anti-standby SEMPRE

        if processing_completed_normally and os.path.exists(CHECKPOINT_FILE):
            print("Elaborazione completata normalmente per tutti i modelli richiesti.")
            # Qui potresti aggiungere la rimozione della cartella preprocessata, se desiderato.
            # remove_preprocessed = input(f"Rimuovere la cartella '{PREPROCESSED_FOLDER_NAME}'? (s/n): ").lower()
            # if remove_preprocessed == 's':
            #    try: shutil.rmtree(preprocessed_audio_dir); print("Cartella preprocessata rimossa.")
            #    except Exception as e_clean: print(f"Errore rimozione cartella preprocessata: {e_clean}")
            try:
                os.remove(CHECKPOINT_FILE)
                print(f"File checkpoint '{CHECKPOINT_FILE}' rimosso.")
            except OSError as e:
                print(f"Warn: Impossibile rimuovere il file di checkpoint '{CHECKPOINT_FILE}': {e}")
        elif not processing_completed_normally:
            print("Elaborazione terminata con errori o interrotta prima del completamento.")
            print(f"Il file di checkpoint '{CHECKPOINT_FILE}' è stato conservato per una possibile ripresa.")
            print(f"Modelli rimasti (se presenti): {checkpoint_data.get('models_to_process', [])}")
        else:
            # Caso in cui processing_completed_normally è False ma il checkpoint non esiste più
            # (potrebbe succedere se l'errore avviene proprio durante la rimozione del checkpoint?)
            print("Elaborazione completata, ma lo stato finale è incerto (checkpoint non trovato).")

    print("\nScript terminato.")

# --- END OF transcribeCraigAudio.py (MODIFICATO) ---