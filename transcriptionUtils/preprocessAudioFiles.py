# --- START OF transcriptionUtils/preprocessAudioFiles.py (CORRETTO per leggere Manifest Completo) ---

import os
import time
import json
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import transcriptionUtils.transcribeAudio as transcribe
import transcriptionUtils.splitAudio as splitter
import multiprocessing as mp
import platform
import numpy as np # Necessario per None check

manifest_data_cache = None; manifest_path_cache = None

# Worker Preprocessing (MODIFICATO)
def _preprocess_worker(input_chunk_path: str, output_chunk_path: str,
                       manifest_path: str,
                       boost_threshold_db: float # Soglia per decidere se boostare (relativo al target)
                       # Nota: target LUFS viene letto dal manifest ora
                       ) -> tuple[str | None, bool]:
    global manifest_data_cache, manifest_path_cache
    filename = os.path.basename(input_chunk_path); success = False; output_file_generated = None
    original_metrics = {} # Dizionario per metriche originali
    target_lufs_from_manifest = splitter.FALLBACK_TARGET_LUFS # Usa fallback di splitAudio

    try:
        # Carica Manifest (cache)
        if manifest_data_cache is None or manifest_path_cache != manifest_path:
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f: manifest_data_cache = json.load(f)
                manifest_path_cache = manifest_path
                if not manifest_data_cache or 'files' not in manifest_data_cache: manifest_data_cache = None
            except Exception as e: print(f"  !!! [PreprocWorker] ERROR loading manifest: {e}"); manifest_data_cache = None

        # Recupera target LUFS globale e metriche specifiche del chunk
        if manifest_data_cache:
            target_lufs_from_manifest = manifest_data_cache.get("effective_target_lufs_for_norm", splitter.FALLBACK_TARGET_LUFS)
            abs_input_chunk_path = os.path.abspath(input_chunk_path)
            chunk_metadata = manifest_data_cache['files'].get(abs_input_chunk_path)
            if chunk_metadata:
                # Estrai le metriche salvate nel manifest
                original_metrics['loudness_lufs'] = chunk_metadata.get('original_lufs')
                original_metrics['avg_dbfs'] = chunk_metadata.get('original_avg_dbfs') # RMS dBFS
                original_metrics['snr'] = chunk_metadata.get('original_snr')
                original_metrics['noise_floor'] = chunk_metadata.get('noise_floor_15p_dbfs') # Se l'abbiamo salvata
                original_metrics['peak'] = chunk_metadata.get('peak_dbfs') # Se l'abbiamo salvata
            # else: print(f"  WARN: Metadata non trovati per {abs_input_chunk_path}") # Meno verboso

        # Esegui Preprocessing
        if os.path.exists(output_chunk_path): success = True; output_file_generated = output_chunk_path
        else:
            # Chiama preprocess_audio passando le metriche originali e il target LUFS
            success = transcribe.preprocess_audio(
                input_path=input_chunk_path, output_path=output_chunk_path,
                noise_reduce=True, normalize=True, # Abilita sempre qui, la logica interna decide
                original_metrics=original_metrics, # Passa il dizionario metriche
                target_norm_lufs=target_lufs_from_manifest, # Target LUFS letto/fallback
                boost_threshold_db=boost_threshold_db # Soglia per decidere boost
                # Parametri NR e clipping sono definiti dentro preprocess_audio
            )
            if success and os.path.exists(output_chunk_path): output_file_generated = output_chunk_path

    except Exception as e: print(f"!!! [PreprocWorker] CRITICAL ERR {filename}: {e}"); import traceback; traceback.print_exc(); success = False
    return output_file_generated if success else None, success

# Funzione Principale Preprocessing (MODIFICATA per passare solo parametri necessari)
def run_parallel_preprocessing(input_chunk_dir: str, preprocessed_output_dir: str, num_workers: int,
                               manifest_path: str, # Path al manifest JSON
                               boost_threshold_db: float, # Soglia per boost
                               supported_extensions: tuple
                               ) -> tuple[int, int, list[str]]:
    """Esegue preprocessing con boost/NR condizionale basato su metriche nel manifest."""
    print(f"\n--- Avvio Preprocessing Parallelo (Boost/NR Adattivo via Manifest) ---")
    # ... (Check dirs, trova files...) ...
    if not os.path.isdir(input_chunk_dir): return 0, 0, []
    if not os.path.isfile(manifest_path): print(f"ERRORE: Manifest non trovato: {manifest_path}"); return 0, 0, []
    try: os.makedirs(preprocessed_output_dir, exist_ok=True)
    except OSError as e: print(f"Errore dir preprocessato: {e}"); return 0, 0, []
    files_to_process = [ f for f in os.listdir(input_chunk_dir) if os.path.isfile(os.path.join(input_chunk_dir, f)) and f.lower().endswith(supported_extensions) and f != splitter.SPLIT_MANIFEST_FILENAME ]
    if not files_to_process: print("Nessun chunk da preprocessare."); return 0, 0, []
    print(f"Trovati {len(files_to_process)} chunk da sottoporre a preprocessing.")

    success_count = 0; failure_count = 0; output_file_paths = []
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    global preprocessing_pool_global_ref; preprocessing_pool_global_ref = None
    pool_error_occurred = False
    try:
        with context.Pool(processes=num_workers) as pool:
            preprocessing_pool_global_ref = pool
            # Passa solo i parametri necessari al worker
            tasks_args = [
                (os.path.join(input_chunk_dir, filename), os.path.join(preprocessed_output_dir, filename),
                 manifest_path, boost_threshold_db)
                for filename in files_to_process ]
            results = []
            try: results = pool.starmap(_preprocess_worker, tasks_args); print(f"Completati {len(results)} task preprocessing.")
            except Exception as pool_error: pool_error_occurred = True; print(f"!!! Errore Pool Preprocessing: {pool_error}")
            if not pool_error_occurred: # Processa risultati
                for result_tuple in results:
                    try:
                        output_file_path, success = result_tuple
                        if success: success_count += 1;
                        if success and output_file_path: output_file_paths.append(output_file_path)
                        elif not success: failure_count += 1
                    except Exception as e: failure_count += 1; print(f"!!! Errore processando risultato preproc: {e}")
    finally: preprocessing_pool_global_ref = None
    if pool_error_occurred: print("Preprocessing terminato con errori.")
    print(f"\n--- Preprocessing Parallelo Chunk Completato ---")
    print(f"Chunk processati/skippati: {success_count} (falliti: {failure_count})")
    print(f"Percorsi chunk preprocessati validi: {len(output_file_paths)}")
    return success_count, failure_count, sorted(output_file_paths)

# Riferimento globale per signal handler
preprocessing_pool_global_ref = None

# --- END OF transcriptionUtils/preprocessAudioFiles.py (CORRETTO per leggere Manifest Completo) ---