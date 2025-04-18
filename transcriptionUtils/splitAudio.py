# --- START OF transcriptionUtils/splitAudio.py (MODIFICATO per Terminazione Pool) ---

import os
import json
import math
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
from pydub.silence import detect_nonsilent, detect_silence # type: ignore
import shutil
import numpy as np
import platform
import multiprocessing as mp
# Import per il signal handling (se necessario ridefinirlo qui)
# import signal
# import sys

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
ANALYSIS_CHUNK_MS = 50

# --- Variabili Globali per i Pool ---
# Verranno popolate quando i pool sono attivi
analysis_pool_global_ref = None
splitting_pool_global_ref = None


# --- Worker Analisi Soglie (INVARIATO) ---
def _analyze_audio_for_thresholds(original_path: str, default_silence_thresh_dbfs: float, default_min_silence_len_ms: int) -> tuple[str, float | None, int | None]:
    # ... (Codice worker analisi invariato) ...
    # ... (assicurati che non ci siano riferimenti ai pool globali qui dentro) ...
    pass # Placeholder - usa il codice completo dalla risposta precedente

# --- Funzione Orchestratrice Analisi (MODIFICATA per pool globale) ---
def _run_parallel_audio_analysis(original_audio_dir: str,
                                 supported_extensions: tuple,
                                 num_workers: int,
                                 default_silence_thresh_dbfs: float,
                                 default_min_silence_len_ms: int
                                 ) -> dict[str, dict]:
    global analysis_pool_global_ref # Usa la variabile globale
    print(f"\n--- Avvio Analisi Audio Parallela per Soglie Dinamiche (Workers: {num_workers}) ---")
    custom_params_dict: dict[str, dict] = {}
    # ... (trova original_files) ...
    original_files = sorted([f for f in os.listdir(original_audio_dir) if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)])
    if not original_files: return custom_params_dict

    # Setup Pool
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    analysis_pool_global_ref = None # Resetta prima della creazione
    pool_error_occurred = False

    try:
        with context.Pool(processes=num_workers) as pool:
            analysis_pool_global_ref = pool # Assegna il pool alla variabile globale
            tasks_args = [(os.path.abspath(os.path.join(original_audio_dir, filename)), default_silence_thresh_dbfs, default_min_silence_len_ms) for filename in original_files]
            analysis_results = []
            try:
                analysis_results = pool.starmap(_analyze_audio_for_thresholds, tasks_args)
                print(f"Completata analisi preliminare per {len(analysis_results)} file.")
            except Exception as pool_error:
                 pool_error_occurred = True # Segna l'errore
                 print(f"!!! Errore durante l'esecuzione del Pool di Analisi: {pool_error}")

            # Processa i risultati solo se non ci sono stati errori gravi nel pool
            if not pool_error_occurred:
                for original_path, calc_thresh, calc_min_len in analysis_results:
                    custom_params_dict[original_path] = {'threshold': calc_thresh if calc_thresh is not None else default_silence_thresh_dbfs, 'min_len': calc_min_len if calc_min_len is not None else default_min_silence_len_ms}

    finally:
        analysis_pool_global_ref = None # Resetta la variabile globale quando il pool è chiuso

    if pool_error_occurred:
        print("Pool di analisi terminato con errori, ritorno dizionario vuoto.")
        return {} # Ritorna vuoto se il pool ha fallito

    print("--- Fine Analisi Audio Parallela ---")
    return custom_params_dict


# --- Worker Splitting (INVARIATO) ---
def _process_single_audio_file_for_split(original_path: str, split_audio_dir: str, silence_thresh_dbfs: float, min_silence_len_ms: int, keep_silence_ms: int, split_naming_threshold_seconds: float, max_silence_between_ms: int, target_max_chunk_duration_ms: int) -> tuple[dict, int, int, int]:
    # ... (Codice worker splitting invariato) ...
     # ... (assicurati che non ci siano riferimenti ai pool globali qui dentro) ...
    pass # Placeholder - usa il codice completo dalla risposta precedente

# --- Funzione Principale split_large_audio_files (MODIFICATA per pool globale) ---
def split_large_audio_files(original_audio_dir: str, split_audio_dir: str, split_naming_threshold_seconds: float = 45 * 60, default_min_silence_len_ms: int = 700, default_silence_thresh_dbfs: float = -35.0, keep_silence_ms: int = 150, max_silence_between_ms: int = 1500, target_max_chunk_duration_ms: int = 10 * 60 * 1000, supported_extensions=(".flac",".m4a"), num_workers: int | None = None) -> tuple[str | None, dict]:
    global splitting_pool_global_ref # Usa la variabile globale
    # ... (Stampa parametri iniziali, setup num_workers, check dir...) ...

    # --- FASE 1: Analisi Parallela (Chiama la funzione aggiornata) ---
    custom_params_dict = _run_parallel_audio_analysis(
        original_audio_dir, supported_extensions, num_workers if num_workers is not None else os.cpu_count() or 4, # Passa num_workers
        default_silence_thresh_dbfs, default_min_silence_len_ms
    )

    # --- FASE 2: Splitting/Copia Parallela ---
    print(f"\n--- Avvio Splitting/Copia/Raggruppamento Parallelo (Workers: {num_workers}) ---")
    # ... (Inizializzazione split_manifest, manifest_path, contatori...) ...
    # ... (Ottieni original_files da custom_params_dict.keys()) ...
    split_manifest = {
        "split_method": "silence_detection_dynamic_threshold_grouped",
        "default_min_silence_len_ms": default_min_silence_len_ms,
        "default_silence_thresh_dbfs": default_silence_thresh_dbfs,
        "keep_silence_ms": keep_silence_ms,
        "split_naming_threshold_seconds": split_naming_threshold_seconds,
        "max_silence_between_ms": max_silence_between_ms,
        "target_max_chunk_duration_ms": target_max_chunk_duration_ms,
        "files": {}
    }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    total_files_processed_success = 0
    total_chunks_exported = 0
    total_errors = 0
    original_files = sorted(custom_params_dict.keys())
    if not original_files: return None, {}


    # Setup Pool per lo splitting
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    splitting_pool_global_ref = None # Resetta
    pool_error_occurred_split = False

    try:
        with context.Pool(processes=num_workers) as pool:
            splitting_pool_global_ref = pool # Assegna alla variabile globale
            tasks_args = []
            for original_path in original_files:
                params = custom_params_dict[original_path]
                tasks_args.append(
                    (original_path, split_audio_dir,
                     params['threshold'], params['min_len'],
                     keep_silence_ms, split_naming_threshold_seconds,
                     max_silence_between_ms, target_max_chunk_duration_ms)
                )

            split_results = []
            try:
                split_results = pool.starmap(_process_single_audio_file_for_split, tasks_args)
                print(f"\nCompletati {len(split_results)} task di splitting/copy/grouping dalla pool.")
            except Exception as pool_error:
                 pool_error_occurred_split = True
                 print(f"!!! Errore durante l'esecuzione del Pool di Splitting: {pool_error}")

            # Processa i risultati
            if not pool_error_occurred_split:
                for result_tuple in split_results:
                    try:
                        file_manifest_entries, success_flag, chunks_exported, _ = result_tuple
                        if success_flag:
                            total_files_processed_success += 1
                            split_manifest["files"].update(file_manifest_entries)
                            total_chunks_exported += chunks_exported
                        else: total_errors += 1
                    except Exception as e:
                        total_errors += 1
                        print(f"!!! Errore processando risultato task splitting: {e}")

    finally:
        splitting_pool_global_ref = None # Resetta la variabile globale

    # --- Riepilogo e Salvataggio Manifest (solo se il pool non ha fallito) ---
    if pool_error_occurred_split:
         print("Pipeline di Splitting terminata con errori gravi nel pool.")
         return None, {} # Non salvare manifest parziale

    # ... (Stampa riepilogo finale) ...
    print(f"\n--- Pipeline di Splitting (con Raggruppamento) Completata ---")
    print(f"File originali processati con successo (analisi + split/copy): {total_files_processed_success}")
    print(f"  - Totale chunk finali esportati: {total_chunks_exported}")
    print(f"Errori riscontrati (analisi o split/copy falliti): {total_errors}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")
    if not split_manifest["files"]: return None, {}

    # ... (Salvataggio manifest) ...
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e: print(f"!!! Errore critico durante salvataggio manifest: {e}"); return None, {}


# --- END OF transcriptionUtils/splitAudio.py (MODIFICATO per Terminazione Pool) ---