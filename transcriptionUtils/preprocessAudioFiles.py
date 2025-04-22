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
                       #target_avg_dbfs: float,
                       boost_threshold_db: float # Soglia per decidere se boostare (relativo al target)
                       # Nota: target LUFS viene letto dal manifest ora
                       ) -> tuple[str | None, bool]:
    global manifest_data_cache, manifest_path_cache
    filename = os.path.basename(input_chunk_path); success = False; 
    output_file_generated = None
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

        # Recupera metriche originali dal manifest (inclusi lufs/snr)
        if manifest_data_cache:
            target_lufs_from_manifest = manifest_data_cache.get("effective_target_lufs_for_norm", splitter.FALLBACK_TARGET_LUFS)
            abs_input_chunk_path = os.path.abspath(input_chunk_path)
            chunk_metadata = manifest_data_cache['files'].get(abs_input_chunk_path)
            if chunk_metadata:
                original_metrics['loudness_lufs'] = chunk_metadata.get('original_lufs')
                original_metrics['avg_dbfs'] = chunk_metadata.get('original_avg_dbfs')
                original_metrics['snr'] = chunk_metadata.get('original_snr') # Se l'abbiamo salvata
            # else: print(f"  WARN: Metadata non trovati per {abs_input_chunk_path}") # Meno verboso
        else:
             original_metrics = None # O un dizionario vuoto? Meglio None
        # Esegui Preprocessing
        if os.path.exists(output_chunk_path): success = True; output_file_generated = output_chunk_path
        else:
            # Chiama preprocess_audio
            success = transcribe.preprocess_audio(
                input_path=input_chunk_path, output_path=output_chunk_path,
                noise_reduce=True, normalize=True,
                original_metrics=original_metrics,         # Passa il dizionario metriche lette
                target_norm_lufs=target_lufs_from_manifest,# Passa il target LUFS letto
                boost_threshold_db=boost_threshold_db     # Passa la soglia boost ricevuta
            )
            if success and os.path.exists(output_chunk_path): output_file_generated = output_chunk_path

    except Exception as e: print(f"!!! [PreprocWorker] CRITICAL ERR {filename}: {e}"); import traceback; traceback.print_exc(); success = False
    return output_file_generated if success else None, success

# Funzione Principale Preprocessing (MODIFICATA per passare solo parametri necessari)
def run_parallel_preprocessing(input_chunk_dir: str, preprocessed_output_dir: str, num_workers: int,
                               manifest_path: str,
                               #target_avg_dbfs: float, # Manteniamo questo nome anche se contiene LUFS target
                               boost_threshold_db: float,
                               supported_extensions: tuple
                               ) -> tuple[int, int, list[str]]:
    """Esegue preprocessing con boost/NR condizionale basato su metriche nel manifest."""
    print(f"\n--- Avvio Preprocessing Parallelo (Boost/NR Adattivo  via Manifest) ---")
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
    pool_error_occurred = False # Flag per errore nel pool
    results = [] # Inizializza risultati

    try:
        with context.Pool(processes=num_workers) as pool:
            preprocessing_pool_global_ref = pool
            tasks_args = [
                (os.path.join(input_chunk_dir, filename),
                 os.path.join(preprocessed_output_dir, filename),
                 manifest_path,
                 #target_avg_dbfs,      # Passa target (anche se è LUFS)
                 boost_threshold_db    # Passa soglia
                 )
                for filename in files_to_process ]
            try:
                # Esegui i task
                results = pool.starmap(_preprocess_worker, tasks_args)
                print(f"Completati {len(results)} task preprocessing.")
            except Exception as pool_error_obj: # Cattura l'oggetto errore
                 pool_error_occurred = True # Imposta il flag
                 # Stampa l'errore QUI, dove la variabile è definita
                 print(f"!!! Errore durante l'esecuzione del Pool di Preprocessing: {pool_error_obj}")
                 # traceback.print_exc() # Aggiungi se vuoi il traceback completo
    finally:
        # Assicurati che il riferimento globale venga resettato
        preprocessing_pool_global_ref = None

    # --- CORREZIONE QUI ---
    # Controlla il FLAG, non tentare di accedere a pool_error
    if pool_error_occurred:
        print("Processamento risultati interrotto a causa di errore nel Pool.")
        # Ritorna conteggi/lista correnti (probabilmente vuoti o parziali)
        return success_count, failure_count, output_file_paths
    # --- FINE CORREZIONE ---
    else:
        # Processa i risultati solo se il pool è terminato normalmente
        for result_tuple in results:
            try:
                output_file_path, success = result_tuple
                if success:
                    success_count += 1
                    if output_file_path: output_file_paths.append(output_file_path)
                else:
                    failure_count += 1
            except Exception as e:
                failure_count += 1
                print(f"!!! Errore processando risultato task preproc: {e}")

    # Stampa Riepilogo Finale
    print(f"\n--- Preprocessing Parallelo Chunk Completato ---")
    print(f"Chunk processati/skippati: {success_count} (falliti: {failure_count})")
    print(f"Percorsi chunk preprocessati validi: {len(output_file_paths)}")
    return success_count, failure_count, sorted(output_file_paths)

# Riferimento globale per signal handler
preprocessing_pool_global_ref = None

# --- END OF transcriptionUtils/preprocessAudioFiles.py (CORRETTO per leggere Manifest Completo) ---