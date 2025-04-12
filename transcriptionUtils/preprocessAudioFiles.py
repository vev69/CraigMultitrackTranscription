# --- START OF transcriptionUtils/preprocessAudioFiles.py ---

import os
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import transcriptionUtils.transcribeAudio as transcribe # Importa per usare preprocess_audio

# --- Funzione Worker per il Preprocessing di un Singolo File ---
def _preprocess_worker(input_path: str, output_path: str) -> tuple[str | None, bool]:
    """
    Worker eseguito in un processo separato per preprocessare un file audio.
    Chiama la funzione preprocess_audio dal modulo transcribeAudio.

    Args:
        input_path: Percorso del file audio originale.
        output_path: Percorso dove salvare il file preprocessato.

    Returns:
        Una tupla: (percorso_output_effettivo, successo_booleano).
        Il percorso di output è None se il preprocessing fallisce o viene saltato.
    """
    filename = os.path.basename(input_path)
    print(f"  [Worker {os.getpid()}] Starting preprocessing for: {filename}")
    start_time = time.time()
    success = False
    try:
        # Salta se il file preprocessato esiste già
        if os.path.exists(output_path):
            print(f"  [Worker {os.getpid()}] Skipping: {filename} (already preprocessed).")
            success = True # Considera successo se esiste già
        else:
            # Chiama la funzione di preprocessing effettiva
            # Assicurati che preprocess_audio ritorni True/False in base al successo
            success = transcribe.preprocess_audio(input_path, output_path,
                                                  noise_reduce=True, # Abilita/Disabilita qui
                                                  normalize=True)    # Abilita/Disabilita qui
            if success:
                print(f"  [Worker {os.getpid()}] SUCCESS preprocessing for: {filename} (Output: {os.path.basename(output_path)})")
            else:
                # preprocess_audio dovrebbe già stampare errori specifici
                print(f"  [Worker {os.getpid()}] FAILED or SKIPPED preprocessing for: {filename} (preprocess_audio returned False)")

    except Exception as e:
        print(f"!!! [Worker {os.getpid()}] CRITICAL ERROR preprocessing {filename}: {e}")
        success = False # Fallimento critico

    end_time = time.time()
    print(f"  [Worker {os.getpid()}] Finished preprocessing for: {filename} in {end_time - start_time:.2f}s (Success: {success})")

    # Ritorna il percorso di output solo se il file è stato creato/esisteva
    return output_path if success and os.path.exists(output_path) else None, success

# --- Funzione Principale per Eseguire il Preprocessing in Parallelo ---
def run_parallel_preprocessing(base_input_dir: str,
                               preprocessed_output_dir: str,
                               num_workers: int,
                               supported_extensions=(".flac",)) -> tuple[int, int, list[str]]:
    """
    Esegue il preprocessing audio in parallelo utilizzando ProcessPoolExecutor.

    Args:
        base_input_dir: Directory contenente i file audio originali.
        preprocessed_output_dir: Directory dove salvare i file preprocessati.
        num_workers: Numero di processi worker da utilizzare.
        supported_extensions: Tuple di estensioni file da processare.

    Returns:
        Una tupla: (numero_successi, numero_fallimenti, lista_percorsi_output_validi).
    """
    print(f"\n--- Avvio Preprocessing Audio Parallelo (Workers: {num_workers}) ---")
    print(f"Input directory: {base_input_dir}")
    print(f"Output directory: {preprocessed_output_dir}")

    if not os.path.isdir(base_input_dir):
        print(f"Errore: Directory di input base non trovata: {base_input_dir}")
        return 0, 0, []

    try:
        os.makedirs(preprocessed_output_dir, exist_ok=True)
    except OSError as e:
        print(f"Errore creazione directory preprocessato: {e}")
        return 0, 0, []

    # Trova i file da processare
    files_to_process = [
        f for f in os.listdir(base_input_dir)
        if os.path.isfile(os.path.join(base_input_dir, f)) and f.lower().endswith(supported_extensions)
    ]

    if not files_to_process:
        print("Nessun file con estensioni supportate trovato nella directory di input.")
        return 0, 0, []

    print(f"Trovati {len(files_to_process)} file da processare.")

    success_count = 0
    failure_count = 0
    output_file_paths = [] # Lista per i percorsi dei file *effettivamente* creati/trovati

    # Usa ProcessPoolExecutor per parallelizzare
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        # Sottometti i task
        for filename in sorted(files_to_process):
            input_path = os.path.join(base_input_dir, filename)
            output_path = os.path.join(preprocessed_output_dir, filename) # Salva con lo stesso nome
            # Sottometti il worker e mappa il future al nome file originale per logging
            future = executor.submit(_preprocess_worker, input_path, output_path)
            futures[future] = filename

        print(f"Sottomessi {len(futures)} task di preprocessing alla pool...")

        # Processa i risultati man mano che completano
        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            processed_count += 1
            try:
                output_file_path, success = future.result() # Ottieni la tupla (path_or_none, success_bool)
                if success:
                    success_count += 1
                    if output_file_path: # Aggiungi solo se un percorso valido è stato restituito
                        output_file_paths.append(output_file_path)
                else:
                    failure_count += 1
                # Aggiornamento progresso (opzionale, può essere verboso)
                # print(f"  Progress: {processed_count}/{len(files_to_process)} completed ({filename} - Success: {success})")

            except Exception as e:
                print(f"!!! Errore ottenendo risultato per {filename}: {e}")
                failure_count += 1

    print(f"\n--- Preprocessing Audio Parallelo Completato ---")
    print(f"File processati/skippati con successo: {success_count}")
    print(f"File falliti: {failure_count}")
    print(f"Percorsi di output validi generati: {len(output_file_paths)}")

    # Ritorna conteggi e la lista dei percorsi *validi* generati
    return success_count, failure_count, sorted(output_file_paths)

# --- END OF transcriptionUtils/preprocessAudioFiles.py ---