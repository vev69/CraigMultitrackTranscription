# --- START OF transcriptionUtils/preprocessAudioFiles.py (MODIFICATO) ---

import os
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.splitAudio as splitter

# --- Funzione Worker _preprocess_worker (INVARIATA NELLA LOGICA INTERNA) ---
def _preprocess_worker(input_path: str, output_path: str) -> tuple[str | None, bool]:
    # ... (codice del worker invariato, chiama sempre transcribe.preprocess_audio) ...
    filename = os.path.basename(input_path)
    print(f"  [Worker {os.getpid()}] Starting preprocessing for: {filename}")
    start_time = time.time()
    success = False
    try:
        if os.path.exists(output_path):
            print(f"  [Worker {os.getpid()}] Skipping: {filename} (already preprocessed).")
            success = True
        else:
            # Assicurarsi che la directory di output esista per questo worker
            # Potrebbe essere necessario se i worker scrivono in sotto-directory diverse
            # Ma in questo caso scrivono tutti nella stessa preprocessed_output_dir
            # os.makedirs(os.path.dirname(output_path), exist_ok=True) # Opzionale
            success = transcribe.preprocess_audio(input_path, output_path,
                                                  noise_reduce=True,
                                                  normalize=True)
            if success:
                print(f"  [Worker {os.getpid()}] SUCCESS preprocessing for: {filename} (Output: {os.path.basename(output_path)})")
            else:
                print(f"  [Worker {os.getpid()}] FAILED or SKIPPED preprocessing for: {filename}")
    except Exception as e:
        print(f"!!! [Worker {os.getpid()}] CRITICAL ERROR preprocessing {filename}: {e}")
        success = False
    end_time = time.time()
    print(f"  [Worker {os.getpid()}] Finished preprocessing for: {filename} in {end_time - start_time:.2f}s (Success: {success})")
    return output_path if success and os.path.exists(output_path) else None, success


# --- Funzione run_parallel_preprocessing MODIFICATA leggermente per chiarezza input/output ---
def run_parallel_preprocessing(input_chunk_dir: str, # Input sono i chunk
                               preprocessed_output_dir: str, # Output per chunk processati
                               num_workers: int,
                               supported_extensions=(".flac",)
                               ) -> tuple[int, int, list[str]]:
    """
    Esegue il preprocessing sui file chunk in parallelo.

    Args:
        input_chunk_dir: Directory contenente i file audio divisi (chunk).
        preprocessed_output_dir: Directory dove salvare i chunk preprocessati.
        num_workers: Numero di processi worker.
        supported_extensions: Tuple di estensioni file da processare.

    Returns:
        Tupla: (success_count, failure_count, list_of_valid_preprocessed_chunk_paths).
    """
    print(f"\n--- Avvio Preprocessing Parallelo dei Chunk (Workers: {num_workers}) ---")
    print(f"Input directory (chunks): {input_chunk_dir}")
    print(f"Output directory (preprocessed chunks): {preprocessed_output_dir}")

    if not os.path.isdir(input_chunk_dir):
        print(f"Errore: Directory input chunk non trovata: {input_chunk_dir}")
        return 0, 0, []

    try:
        os.makedirs(preprocessed_output_dir, exist_ok=True)
    except OSError as e:
        print(f"Errore creazione directory preprocessato: {e}")
        return 0, 0, []

    # Trova i file chunk da processare
    files_to_process = [
        f for f in os.listdir(input_chunk_dir)
        # Escludi il file manifest stesso!
        if os.path.isfile(os.path.join(input_chunk_dir, f)) and \
           f.lower().endswith(supported_extensions) and \
           f != splitter.SPLIT_MANIFEST_FILENAME # Usa la costante dal modulo splitAudio
    ]

    if not files_to_process:
        print("Nessun file chunk con estensioni supportate trovato nella directory di input.")
        return 0, 0, []

    print(f"Trovati {len(files_to_process)} file chunk da preprocessare.")

    success_count = 0
    failure_count = 0
    output_file_paths = [] # Lista per percorsi chunk *preprocessati* validi

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for filename in sorted(files_to_process):
            input_path = os.path.join(input_chunk_dir, filename)
            output_path = os.path.join(preprocessed_output_dir, filename) # Salva con lo stesso nome chunk
            future = executor.submit(_preprocess_worker, input_path, output_path)
            futures[future] = filename

        print(f"Sottomessi {len(futures)} task di preprocessing chunk...")

        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            processed_count += 1
            try:
                output_file_path, success = future.result()
                if success:
                    success_count += 1
                    if output_file_path:
                        output_file_paths.append(output_file_path)
                else:
                    failure_count += 1
            except Exception as e:
                print(f"!!! Errore ottenendo risultato per chunk {filename}: {e}")
                failure_count += 1

    print(f"\n--- Preprocessing Parallelo Chunk Completato ---")
    print(f"Chunk processati/skippati con successo: {success_count}")
    print(f"Chunk falliti: {failure_count}")
    print(f"Percorsi chunk preprocessati validi generati: {len(output_file_paths)}")

    return success_count, failure_count, sorted(output_file_paths)

# --- END OF transcriptionUtils/preprocessAudioFiles.py (MODIFICATO) ---