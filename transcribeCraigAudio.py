# --- START OF transcribeCraigAudio.py ---

import os
import json
import sys
import signal
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.combineSpeakerTexts as combineSpeakers
import torch
import shutil
import platform
import time

# --- Moduli specifici per OS per prevenire lo standby ---
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


SUPPORTED_FILE_EXTENSIONS = (".flac")
CHECKPOINT_FILE = "transcription_checkpoint.json"
BASE_OUTPUT_FOLDER_NAME = "transcription_output"
PREPROCESSED_FOLDER_NAME = "audio_preprocessed" # Cartella per audio processato

# Global variable for checkpointing
checkpoint_data = {}

# --- Funzioni Gestione Standby (invariate) ---
def prevent_sleep():
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
            # else: print("Processo 'caffeinate' già attivo.") # Meno verboso
        elif system == "Linux":
            if 'pydbus' in sys.modules and inhibitor is None:
                 try:
                     mgr = bus.get('org.freedesktop.PowerManagement', '/org/freedesktop/PowerManagement/Inhibit')
                     app_id = "CraigTranscriptionScript"; reason = "Running audio transcription"
                     cookie = mgr.Inhibit(app_id, reason); inhibitor = cookie
                     print(f"Richiesta anti-standby D-Bus inviata (Inhibitor: {inhibitor}).")
                 except Exception as e_dbus: print(f"!!! Errore D-Bus Inhibit: {e_dbus}"); inhibitor = None
            # elif inhibitor is not None: print("Richiesta D-Bus anti-standby già attiva.") # Meno verboso
            # else: print("Funzionalità anti-standby per Linux non attiva.") # Meno verboso
        # else: print(f"OS '{system}' non supportato per gestione standby.") # Meno verboso
    except Exception as e: print(f"!!! Errore in prevent_sleep: {e}")

def allow_sleep():
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
            # else: print("Nessun processo 'caffeinate' attivo.") # Meno verboso
        elif system == "Linux":
            if inhibitor is not None and 'pydbus' in sys.modules:
                try:
                    print(f"Rilascio richiesta D-Bus (Inhibitor: {inhibitor})...")
                    mgr = bus.get('org.freedesktop.PowerManagement', '/org/freedesktop/PowerManagement/Inhibit')
                    mgr.UnInhibit(inhibitor); inhibitor = None
                    print("Richiesta D-Bus rilasciata.")
                except Exception as e_dbus: print(f"!!! Errore D-Bus UnInhibit: {e_dbus}")
            # else: print("Nessuna richiesta D-Bus attiva.") # Meno verboso
        # else: pass # Meno verboso
    except Exception as e: print(f"!!! Errore in allow_sleep: {e}")

import atexit
atexit.register(allow_sleep)

# --- Gestione Segnali ---
def signal_handler(sig, frame):
    print("\n*** Interruzione rilevata! Pulizia e salvataggio... ***")
    allow_sleep()
    save_checkpoint()
    print(f"Checkpoint salvato in {CHECKPOINT_FILE}.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Funzioni Checkpoint ---
def save_checkpoint():
    global checkpoint_data
    if not checkpoint_data: return
    try:
        # Aggiorna il timestamp del salvataggio
        checkpoint_data['last_saved'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
            # print(f"Checkpoint salvato.") # Log meno verboso
    except Exception as e: print(f"!!! Errore salvataggio checkpoint: {e}")

def load_checkpoint():
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

# --- NUOVA FUNZIONE DI PREPROCESSING DI TUTTI I FILE ---
def preprocess_all_audio(base_input_dir: str, preprocessed_output_dir: str) -> bool:
    """
    Esegue il preprocessing (riduzione rumore, normalizzazione) su tutti i file
    .flac nella directory di input e li salva nella directory di output.
    Ritorna True se almeno un file è stato processato con successo, False altrimenti.
    """
    print(f"\n--- Avvio Preprocessing Audio da '{base_input_dir}' a '{preprocessed_output_dir}' ---")
    if not os.path.isdir(base_input_dir):
        print(f"Errore: Directory di input base non trovata: {base_input_dir}"); return False

    try:
        os.makedirs(preprocessed_output_dir, exist_ok=True)
        print(f"Directory per audio preprocessato: {preprocessed_output_dir}")
    except OSError as e:
        print(f"Errore creazione directory preprocessato: {e}"); return False

    files_to_process = [f for f in os.listdir(base_input_dir) if f.endswith(SUPPORTED_FILE_EXTENSIONS)]
    if not files_to_process:
        print("Nessun file .flac trovato nella directory di input."); return False

    print(f"Trovati {len(files_to_process)} file .flac da processare.")
    success_count = 0
    failure_count = 0

    for filename in sorted(files_to_process):
        input_path = os.path.join(base_input_dir, filename)
        output_path = os.path.join(preprocessed_output_dir, filename) # Salva con lo stesso nome

        # Salta se il file preprocessato esiste già (permette ripresa parziale)
        if os.path.exists(output_path):
            print(f"Skipping {filename} (già preprocessato).")
            success_count += 1 # Conta come successo se esiste già
            continue

        print(f"\nPreprocessing {filename}...")
        try:
            # Chiama la funzione di preprocessing da transcribeAudio
            processed_ok = transcribe.preprocess_audio(input_path, output_path,
                                                      noise_reduce=True, # Abilita/Disabilita qui
                                                      normalize=True)    # Abilita/Disabilita qui
            if processed_ok:
                success_count += 1
                print(f"Preprocessing di {filename} completato.")
            else:
                failure_count += 1
                print(f"Preprocessing di {filename} fallito o saltato (nessun file di output generato). Si userà l'originale se necessario.")
                # Opzionale: copiare l'originale se il preprocessing fallisce ma si vuole continuare?
                # try: shutil.copy2(input_path, output_path); print("  Copiato file originale.")
                # except Exception as e_copy: print(f"  Errore copia file originale: {e_copy}"); failure_count += 1

        except Exception as e:
            failure_count += 1
            print(f"!!! Errore critico durante preprocessing di {filename}: {e}")
            # Considera se fermare tutto o continuare con gli altri file

    print(f"\n--- Preprocessing Audio Completato ---")
    print(f"File processati/skippati con successo: {success_count}")
    print(f"File falliti/saltati: {failure_count}")

    # Ritorna True se almeno un file è stato processato o era già presente
    return success_count > 0

# --- Funzione transcribeFilesInDirectory (MODIFICATA per leggere da preprocessed_dir) ---
def transcribeFilesInDirectory(preprocessed_dir: str, model, model_choice: str, output_dir: str):
    """
    Trascrive i file .flac dalla directory preprocessata, salvando l'output in output_dir.
    Usa il checkpoint globale basato sui nomi file originali.
    """
    global checkpoint_data
    filesTranscribed_original_paths = [] # Lista dei percorsi *originali* dei file processati
    processed_in_this_session = 0

    # Lista dei file GIA' processati per questo modello (basata sui percorsi originali)
    # Assumiamo che il checkpoint 'files_processed' usi i percorsi originali come chiave
    original_base_dir = checkpoint_data.get('base_input_directory', None)
    if not original_base_dir:
         print("ERRORE: 'base_input_directory' non trovato nel checkpoint per controllare i file già processati.")
         return [] # Impossibile procedere senza sapere quali file originali sono stati fatti

    already_processed_originals = checkpoint_data.setdefault('files_processed', {}).get(model_choice, [])

    try:
        if not os.path.isdir(preprocessed_dir):
            print(f"Errore: Directory audio preprocessato non trovata: {preprocessed_dir}")
            return []

        # Lista dei file effettivamente presenti nella cartella preprocessata
        preprocessed_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith(SUPPORTED_FILE_EXTENSIONS)])
        print(f"Trovati {len(preprocessed_files)} file preprocessati in {preprocessed_dir}")
        if not preprocessed_files: return [] # Esce se non ci sono file preprocessati

        for file_index, filename in enumerate(preprocessed_files):
            preprocessed_path = os.path.join(preprocessed_dir, filename)
            original_path = os.path.join(original_base_dir, filename) # Ricostruisce il percorso originale

            # Controlla se il file ORIGINALE è già stato processato per questo modello
            if original_path in already_processed_originals:
                print(f"({file_index+1}/{len(preprocessed_files)}) Skipping {filename} (già processato per '{model_choice}')")
                filesTranscribed_original_paths.append(original_path) # Aggiungi comunque alla lista per consistenza
                continue

            print(f"\n--- ({file_index+1}/{len(preprocessed_files)}) Trascrizione '{model_choice}' per: {filename} (da preprocessato) ---")
            # Passa il percorso del file preprocessato alla funzione di trascrizione
            output_file_path = transcribe.transcribeAudioFile(preprocessed_path, model, model_choice, output_dir)

            final_file_exists = output_file_path and os.path.exists(output_file_path)
            is_explicit_failure = output_file_path and "FAILED" in os.path.basename(output_file_path).upper()

            if final_file_exists and not is_explicit_failure:
                print(f"Trascrizione completata con successo per {filename}.")
                # Aggiungi al checkpoint (basato su percorso originale)
                filesTranscribed_original_paths.append(original_path)
                checkpoint_data['files_processed'].setdefault(model_choice, []).append(original_path)
                processed_in_this_session += 1
                save_checkpoint()
            elif final_file_exists and is_explicit_failure:
                # Un file di log errore è stato creato da transcribeAudioFile
                print(f"ERRORE grave durante trascrizione di {filename}. Log: {os.path.basename(output_file_path)}")
                # Marca come tentato nel checkpoint per non riprovare all'infinito
                filesTranscribed_original_paths.append(original_path)
                checkpoint_data['files_processed'].setdefault(model_choice, []).append(original_path)
                processed_in_this_session += 1
                save_checkpoint()
            else:
                # Nessun file restituito o il file restituito non esiste -> Fallimento non gestito?
                print(f"ERRORE/WARN: Trascrizione per {filename} non ha prodotto un file finale valido o ha restituito None.")
                # Marca comunque come tentato per evitare loop
                filesTranscribed_original_paths.append(original_path)
                checkpoint_data['files_processed'].setdefault(model_choice, []).append(original_path)
                processed_in_this_session += 1
                save_checkpoint()

    except Exception as e:
        print(f"Errore in transcribeFilesInDirectory: {e}")
        save_checkpoint() # Salva stato corrente anche in caso di errore nel loop
        return filesTranscribed_original_paths # Ritorna lista parziale

    print(f"\n--- Completata trascrizione per modello '{model_choice}' ---")
    print(f"File processati/skippati in totale per '{model_choice}': {len(filesTranscribed_original_paths)}")
    print(f"File processati/con errore in questa sessione per '{model_choice}': {processed_in_this_session}")
    return filesTranscribed_original_paths


# ============================================================================
# --- Main Execution Logic ---
# ============================================================================

if __name__ == "__main__":

    prevent_sleep() # Attiva anti-standby

    # --- Load Checkpoint or Initialize ---
    if load_checkpoint():
        print(f"Ripresa da checkpoint. Modello originale: '{checkpoint_data.get('original_model_choice', 'N/A')}'")
        print(f"Modelli rimasti: {checkpoint_data.get('models_to_process', [])}")
        print(f"Directory input originale: {checkpoint_data.get('base_input_directory', 'N/A')}")
        resume = input("Vuoi riprendere la sessione? (s/n): ").strip().lower()
        if resume != 's': print("Avvio nuova sessione."); checkpoint_data = {}; load_checkpoint()

    # --- Get User Input if needed ---
    if not checkpoint_data:
        print("\n--- Nuova Sessione ---")
        while True:
            print("\nModelli disponibili (specificare modello-dimensione):")
            print("  whisper-medium        (Originale OpenAI)")
            print("  whisper-largev2       (Originale OpenAI)")
            print("  whisper-largev3       (Originale OpenAI)")
            print("  hf_whisper-medium     (OpenAI via HF Ottimizzato)")
            print("  hf_whisper-largev2    (OpenAI via HF Ottimizzato)")
            print("  hf_whisper-largev3    (OpenAI via HF Ottimizzato)")
            print("  whispy_italian        (Fine-tuned Italiano via HF)")
            print("  entrambi              (Esegue whisper-medium poi hf_whisper-medium)")
            original_model_choice_input = input("Scegli modello (es. 'whisper-largev2', 'entrambi'): ").strip().lower()

            valid_choices = ["whisper-medium", "whisper-largev2", "whisper-largev3",
                             "hf_whisper-medium", "hf_whisper-largev2", "hf_whisper-largev3",
                             "whispy_italian", "entrambi"]
            if original_model_choice_input in valid_choices: break
            else: print("Scelta non valida.")

        while True:
            base_input_directory = input('Inserisci directory audio originale: ').strip()
            if os.path.isdir(base_input_directory): break
            else: print(f"Errore: '{base_input_directory}' non è una directory valida.")

        # Determina i modelli da processare
        models_to_process_list = []
        if original_model_choice_input == "entrambi":
            # Definisci qui quali modelli eseguire per "entrambi"
            models_to_process_list = ["whisper-medium", "hf_whisper-medium"]
            print("Opzione 'entrambi' selezionata: verranno eseguiti 'whisper-medium' e 'hf_whisper-medium'.")
        else:
            models_to_process_list = [original_model_choice_input]

        checkpoint_data = {
            "base_input_directory": base_input_directory,
            "original_model_choice": original_model_choice_input, # Salva scelta utente originale
            "models_to_process": models_to_process_list, # Lista dei modelli da eseguire
            "files_processed": {},
            "current_model_processing": None,
            "last_saved": None
        }
        print("Dati sessione inizializzati.")
        save_checkpoint()

    # --- Setup Directories ---
    base_input_dir = checkpoint_data['base_input_directory']
    preprocessed_audio_dir = os.path.join(base_input_dir, PREPROCESSED_FOLDER_NAME)
    base_output_dir = os.path.join(base_input_dir, BASE_OUTPUT_FOLDER_NAME)

    # --- ESEGUI PREPROCESSING ---
    if not preprocess_all_audio(base_input_dir, preprocessed_audio_dir):
         print("\nErrore critico durante il preprocessing o nessun file trovato/processato.")
         allow_sleep(); sys.exit(1)

    # --- Main Processing Loop ---
    models_to_process = checkpoint_data.get('models_to_process', [])
    if not models_to_process:
         print("Nessun modello specificato per l'elaborazione nel checkpoint.")
         if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE) # Pulisci checkpoint vuoto
         allow_sleep(); sys.exit(0)

    try: os.makedirs(base_output_dir, exist_ok=True); print(f"\nDirectory output base: {base_output_dir}")
    except OSError as e: print(f"!!! Errore creazione output base: {e}"); allow_sleep(); sys.exit(1)

    if torch.cuda.is_available(): print('CUDA disponibile.')
    else: print('ATTENZIONE: CUDA non disponibile.')

    current_model_index = 0
    processing_completed_normally = False
    try:
        while current_model_index < len(models_to_process):
            # Usa il nome completo (con dimensione) come identificativo
            current_model_choice_full = models_to_process[current_model_index]
            # Crea un nome cartella sicuro dall'identificativo completo
            current_model_folder_name = current_model_choice_full.replace('/', '_') # Sostituisci / se presenti

            print(f"\n{'='*20} Elaborazione Modello: {current_model_choice_full} {'='*20}")

            checkpoint_data['current_model_processing'] = current_model_choice_full
            save_checkpoint()

            model_output_dir = os.path.join(base_output_dir, current_model_folder_name)
            try: os.makedirs(model_output_dir, exist_ok=True); print(f"Directory output: {model_output_dir}")
            except OSError as e:
                print(f"!!! Errore creazione dir output per {current_model_folder_name}: {e}. Salto.");
                checkpoint_data['models_to_process'] = models_to_process[current_model_index + 1:]; save_checkpoint()
                current_model_index += 1; continue

            MODEL = None
            try:
                print(f"Caricamento modello '{current_model_choice_full}'...")
                MODEL = transcribe.load_model(current_model_choice_full)
                if MODEL is None: raise ValueError("load_model ha restituito None.")

                print(f"\nAvvio trascrizione per '{current_model_choice_full}' (da audio preprocessato)...")
                # Questa funzione ora ritorna la lista dei percorsi ORIGINALI processati/tentati
                processed_original_paths = transcribeFilesInDirectory(
                    preprocessed_audio_dir, MODEL, current_model_choice_full, model_output_dir
                )

                # --- CONTROLLO SUCCESSO MODIFICATO ---
                # Verifichiamo se almeno un file .txt finale è stato creato per questo modello
                # (presuppone che transcribeAudioFile ritorni il percorso del file completato)
                any_output_file_created = False
                processed_files_for_model = checkpoint_data.get('files_processed', {}).get(current_model_choice_full, [])
                if processed_files_for_model: # Se c'è almeno un file processato nel checkpoint
                    # Ricostruiamo il percorso atteso per l'ultimo file processato
                    last_original_path = processed_files_for_model[-1]
                    last_filename = os.path.basename(last_original_path)
                    expected_completed_file = os.path.join(model_output_dir, f"{os.path.splitext(last_filename)[0]}-TranscribedAudio.txt")
                    # Questo controllo non è perfetto, ma dà un'idea se l'output esiste
                    if os.path.exists(expected_completed_file):
                         any_output_file_created = True
                         print(f"Verifica successo: Trovato file finale per l'ultimo elemento processato ({os.path.basename(expected_completed_file)}). Assumo successo generale.")
                    else:
                         # Potrebbe essere un file di errore o fallimento nell'ultimo step
                         print(f"Verifica successo: NON trovato file finale per l'ultimo elemento processato ({os.path.basename(expected_completed_file)}). Potrebbero esserci stati errori.")

                # Se è stato creato output (o se non c'erano file da processare), procedi con la combinazione
                # Se non c'erano file audio preprocessati, processed_original_paths sarà vuoto ma non è un errore.
                if any_output_file_created or not processed_original_paths: # Procede anche se non c'erano file da trascrivere
                    if processed_files_for_model: # Combina solo se ci sono file processati
                        print(f"\nCombinazione file per '{current_model_choice_full}'...")
                        combineSpeakers.combineTranscribedSpeakerFiles(model_output_dir)
                        print(f"Combinazione completata per '{current_model_choice_full}'.")
                    else:
                        print(f"Nessun file processato per '{current_model_choice_full}'. Salto combinazione.")
                else:
                     # Se c'erano file da processare ma non è stato creato l'output finale atteso
                     print(f"WARN: Nessun file finale .txt trovato per '{current_model_choice_full}', anche se il processing è stato tentato. Salto combinazione.")

                # --- Fine controllo successo ---

                # --- Mark model as done ---
                checkpoint_data['models_to_process'] = models_to_process[current_model_index + 1:]
                checkpoint_data['current_model_processing'] = None; save_checkpoint()
                current_model_index += 1

            except Exception as e_inner_loop:
                 print(f"\n!!! Errore durante elaborazione modello '{current_model_choice_full}': {e_inner_loop}")
                 save_checkpoint(); raise

            finally:
                # ... (Cleanup GPU Memory invariato) ...
                if MODEL is not None:
                     # ... (codice cleanup) ...
                     pass # Inserisci qui il codice di cleanup GPU precedente
                if torch.cuda.is_available(): torch.cuda.empty_cache(); # print("Cache CUDA svuotata.")
                time.sleep(1) # Breve pausa

        processing_completed_normally = True

    except Exception as e_outer_loop:
         print("\n!!! Ciclo di elaborazione principale interrotto a causa di un errore.")

    finally:
        # --- Final Cleanup ---
        print("\n=== Fine Elaborazione ===")
        allow_sleep() # Disattiva anti-standby

        if processing_completed_normally and os.path.exists(CHECKPOINT_FILE):
            print("Elaborazione completata normalmente.")
            # Opzionale: chiedere all'utente se rimuovere la cartella preprocessata
            # remove_preprocessed = input(f"Rimuovere la cartella '{PREPROCESSED_FOLDER_NAME}'? (s/n): ").lower()
            # if remove_preprocessed == 's':
            #    try: shutil.rmtree(preprocessed_audio_dir); print("Cartella preprocessata rimossa.")
            #    except Exception as e_clean: print(f"Errore rimozione cartella preprocessata: {e_clean}")
            try: os.remove(CHECKPOINT_FILE); print(f"File checkpoint {CHECKPOINT_FILE} rimosso.")
            except OSError as e: print(f"Warn: Impossibile rimuovere checkpoint: {e}")
        elif not processing_completed_normally: print("Elaborazione terminata con errori/interruzione. Checkpoint conservato.")
        else: print("Elaborazione completata.")

    print("\nScript terminato.")

# --- END OF transcribeCraigAudio.py ---