# --- START OF transcriptionUtils/preprocessAudioFiles.py (MODIFICATO) ---

import os
import time
import json # Per caricare manifest
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.splitAudio as splitter
import multiprocessing as mp
import platform # Per start_method

# Variabile globale per cache manifest per processo worker
manifest_data_cache = None
manifest_path_cache = None

# --- Funzione Worker (Legge manifest e passa avg_dbfs) ---
def _preprocess_worker(input_chunk_path: str,
                       output_chunk_path: str,
                       manifest_path: str,
                       target_avg_dbfs: float,
                       boost_threshold_db: float
                       ) -> tuple[str | None, bool]:
    global manifest_data_cache, manifest_path_cache
    filename = os.path.basename(input_chunk_path)
    success = False; output_file_generated = None; original_avg_dbfs = None

    try:
        # Carica Manifest (con cache)
        if manifest_data_cache is None or manifest_path_cache != manifest_path:
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f: manifest_data_cache = json.load(f)
                manifest_path_cache = manifest_path
                if not manifest_data_cache or 'files' not in manifest_data_cache:
                     print(f"  [PreprocWorker] ERROR: Manifest invalido {manifest_path}"); manifest_data_cache = None
            except Exception as e: print(f"  !!! [PreprocWorker] ERROR loading manifest {manifest_path}: {e}"); manifest_data_cache = None

        # Recupera original_avg_dbfs
        if manifest_data_cache:
            abs_input_chunk_path = os.path.abspath(input_chunk_path)
            chunk_metadata = manifest_data_cache['files'].get(abs_input_chunk_path)
            if chunk_metadata: original_avg_dbfs = chunk_metadata.get('original_avg_dbfs') # Può essere None o -999.0
            # else: print(f"  WARN: Metadata not found for {abs_input_chunk_path}") # Meno verboso

        # Esegui Preprocessing
        if os.path.exists(output_chunk_path):
            success = True; output_file_generated = output_chunk_path
        else:
            success = transcribe.preprocess_audio(
                input_path=input_chunk_path, output_path=output_chunk_path,
                noise_reduce=True, normalize=True,
                original_avg_dbfs=original_avg_dbfs, # Passa il valore (può essere None)
                target_avg_dbfs=target_avg_dbfs,
                boost_threshold_db=boost_threshold_db )
            if success and os.path.exists(output_chunk_path): output_file_generated = output_chunk_path

    except Exception as e: print(f"!!! [PreprocWorker] CRITICAL ERROR preprocessing {filename}: {e}"); import traceback; traceback.print_exc(); success = False

    return output_file_generated if success else None, success

# --- Funzione Principale (Passa path manifest e target) ---
def run_parallel_preprocessing(input_chunk_dir: str, preprocessed_output_dir: str, num_workers: int,
                               manifest_path: str, target_avg_dbfs: float, boost_threshold_db: float,
                               supported_extensions: tuple) -> tuple[int, int, list[str]]:
    print(f"\n--- Avvio Preprocessing Parallelo dei Chunk (con Boost Selettivo) ---")
    # ... (Stampa parametri, check dirs...) ...
    if not os.path.isdir(input_chunk_dir): return 0, 0, []
    if not os.path.isfile(manifest_path): print(f"ERRORE: Manifest non trovato: {manifest_path}"); return 0, 0, []
    try: os.makedirs(preprocessed_output_dir, exist_ok=True)
    except OSError as e: print(f"Errore dir preprocessato: {e}"); return 0, 0, []

    files_to_process = [ f for f in os.listdir(input_chunk_dir) if os.path.isfile(os.path.join(input_chunk_dir, f)) and f.lower().endswith(supported_extensions) and f != splitter.SPLIT_MANIFEST_FILENAME ]
    if not files_to_process: print("Nessun chunk da preprocessare."); return 0, 0, []
    print(f"Trovati {len(files_to_process)} chunk da preprocessare.")

    success_count = 0; failure_count = 0; output_file_paths = []
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    global preprocessing_pool_global_ref; preprocessing_pool_global_ref = None
    pool_error_occurred = False

    try:
        with context.Pool(processes=num_workers) as pool:
            preprocessing_pool_global_ref = pool
            tasks_args = [
                (os.path.join(input_chunk_dir, filename), os.path.join(preprocessed_output_dir, filename),
                 manifest_path, target_avg_dbfs, boost_threshold_db)
                for filename in files_to_process ]
            results = []
            try: results = pool.starmap(_preprocess_worker, tasks_args); print(f"Completati {len(results)} task preprocessing.")
            except Exception as pool_error: pool_error_occurred = True; print(f"!!! Errore Pool Preprocessing: {pool_error}")

            if not pool_error_occurred:
                for result_tuple in results:
                    try:
                        output_file_path, success = result_tuple
                        if success: success_count += 1;
                        if success and output_file_path: output_file_paths.append(output_file_path)
                        elif not success: failure_count += 1
                    except Exception as e: failure_count += 1; print(f"!!! Errore processando risultato preproc: {e}")
    finally: preprocessing_pool_global_ref = None

    if pool_error_occurred: print("Preprocessing terminato con errori gravi nel pool.") # Non ritorna success/failure counts accurati
    print(f"\n--- Preprocessing Parallelo Chunk Completato ---")
    print(f"Chunk processati/skippati: {success_count} (falliti: {failure_count})")
    print(f"Percorsi chunk preprocessati validi: {len(output_file_paths)}")
    return success_count, failure_count, sorted(output_file_paths)

# Riferimento globale per signal handler
preprocessing_pool_global_ref = None


# --- END OF transcriptionUtils/preprocessAudioFiles.py (MODIFICATO) ---