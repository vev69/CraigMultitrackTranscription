# --- START OF transcriptionUtils/splitAudio.py (MODIFICATO per Taglio Ibrido Tempo/Silenzio) ---

import os
import json
import math
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
# Importa solo detect_silence ora
from pydub.silence import detect_silence # type: ignore
import shutil
# Numpy non più strettamente necessario qui, ma pydub potrebbe usarlo internamente
import numpy as np
import platform
import multiprocessing as mp

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
ANALYSIS_CHUNK_MS = 50

# --- PARAMETRI DI DEFAULT CONFIGURABILI ---
# Per Analisi Dinamica (usati anche come fallback)
DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC: int = 700
DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC: float = -35.0
# Per Splitting Ibrido e Padding
TARGET_CHUNK_DURATION_SECONDS: float = 10 * 60  # 10 minuti
MIN_DURATION_FOR_SPLIT_SECONDS: float = TARGET_CHUNK_DURATION_SECONDS + (1 * 60) # Splitta solo se > 11 min
DEFAULT_KEEP_SILENCE_MS: int = 250                       # Padding
# Parametri specifici per trovare taglio silenzioso
SILENCE_SEARCH_RANGE_MS: int = 10 * 1000        # Cerca silenzio in +/- 10 secondi


# --- WORKER: Analisi Audio (Calcola soglie E avg_dbfs) ---
def _analyze_audio_for_thresholds(original_path: str,
                                  default_silence_thresh_dbfs: float,
                                  default_min_silence_len_ms: int
                                  ) -> tuple[str, float | None, int | None, float | None]:
    filename = os.path.basename(original_path)
    calculated_threshold = None; calculated_min_len = None; calculated_avg_dbfs = None
    try:
        audio = AudioSegment.from_file(original_path)
        if len(audio) == 0: return original_path, None, None, None
        if audio.rms > 0: calculated_avg_dbfs = audio.dBFS
        else: calculated_avg_dbfs = -np.inf
        dbfs_values = [chunk.dBFS for i in range(0, len(audio), ANALYSIS_CHUNK_MS) if len(chunk := audio[i:i+ANALYSIS_CHUNK_MS]) > 0]
        valid_dbfs = [db for db in dbfs_values if np.isfinite(db)]
        if not valid_dbfs: calculated_threshold = default_silence_thresh_dbfs
        else:
            try: # Logica Istogramma per Soglia
                hist, bin_edges = np.histogram(valid_dbfs, bins='auto'); bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
                noise_peak_estimate = np.percentile(valid_dbfs, 15); speech_peak_estimate = np.percentile(valid_dbfs, 80)
                noise_bin_idx = np.argmin(np.abs(bin_centers - noise_peak_estimate)); speech_bin_idx = np.argmin(np.abs(bin_centers - speech_peak_estimate))
                start_idx, end_idx = min(noise_bin_idx, speech_bin_idx), max(noise_bin_idx, speech_bin_idx)
                valley_threshold = default_silence_thresh_dbfs
                if start_idx < end_idx:
                    try:
                        valley_idx_rel = np.argmin(hist[start_idx:end_idx+1]); valley_idx_abs = start_idx + valley_idx_rel
                        potential_threshold = bin_edges[valley_idx_abs + 1]; valley_threshold = max(-60.0, min(-15.0, potential_threshold))
                    except (ValueError, IndexError): valley_threshold = default_silence_thresh_dbfs
                else: valley_threshold = max(-60.0, min(-15.0, noise_peak_estimate + 10.0))
                calculated_threshold = valley_threshold
            except Exception: calculated_threshold = default_silence_thresh_dbfs
        if calculated_threshold is not None: # Calcolo Min Len
            try:
                silence_ranges = detect_silence(audio, 100, calculated_threshold, 1)
                silence_durations_ms = [(end - start) for start, end in silence_ranges]
                meaningful_silences_ms = [d for d in silence_durations_ms if d >= 150]
                if meaningful_silences_ms:
                    potential_min_len = np.percentile(meaningful_silences_ms, 65); calculated_min_len = int(max(250, min(2000, potential_min_len)))
                else: calculated_min_len = default_min_silence_len_ms
            except Exception: calculated_min_len = default_min_silence_len_ms
        else: calculated_min_len = default_min_silence_len_ms
    except Exception as e: print(f"  !!! [AnalysisWorker] ERROR analyzing {filename}: {e}")
    return original_path, calculated_threshold, calculated_min_len, calculated_avg_dbfs

# --- Funzione Orchestratrice Analisi (Raccoglie avg_dbfs) ---
def _run_parallel_audio_analysis(original_audio_dir: str, supported_extensions: tuple, num_workers: int,
                                 default_silence_thresh_dbfs: float, default_min_silence_len_ms: int) -> dict[str, dict]:
    print(f"\n--- Avvio Analisi Audio Parallela (Soglie Dinamiche + Vol Avg) ---")
    custom_params_dict: dict[str, dict] = {}
    original_files = sorted([f for f in os.listdir(original_audio_dir) if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)])
    if not original_files: return custom_params_dict
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    global analysis_pool_global_ref; analysis_pool_global_ref = None
    try:
        with context.Pool(processes=num_workers) as pool:
            analysis_pool_global_ref = pool
            tasks_args = [(os.path.abspath(os.path.join(original_audio_dir, filename)), default_silence_thresh_dbfs, default_min_silence_len_ms) for filename in original_files]
            analysis_results = []; pool_error_occurred = False
            try: analysis_results = pool.starmap(_analyze_audio_for_thresholds, tasks_args); print(f"Completata analisi per {len(analysis_results)} file.")
            except Exception as pool_error: print(f"!!! Errore Pool Analisi: {pool_error}"); pool_error_occurred = True
            if pool_error_occurred: return {} # Ritorna vuoto se fallisce
            for original_path, calc_thresh, calc_min_len, calc_avg_dbfs in analysis_results:
                custom_params_dict[original_path] = {
                    'threshold': calc_thresh if calc_thresh is not None else default_silence_thresh_dbfs,
                    'min_len': calc_min_len if calc_min_len is not None else default_min_silence_len_ms,
                    'avg_dbfs': calc_avg_dbfs if calc_avg_dbfs is not None else -999.0 }
    finally: analysis_pool_global_ref = None
    print("--- Fine Analisi Audio Parallela ---")
    return custom_params_dict

# --- WORKER per Splitting Ibrido (Riceve soglie/avg_dbfs e salva avg_dbfs) ---
def _process_single_audio_file_hybrid_split(original_path: str, split_audio_dir: str,
                                            silence_thresh_dbfs: float, min_silence_len_ms: int, original_avg_dbfs: float,
                                            keep_silence_ms: int, min_duration_for_split_seconds: float,
                                            target_chunk_duration_seconds: float, silence_search_range_ms: int
                                            ) -> tuple[dict, int, int, int]:
    filename = os.path.basename(original_path); file_manifest_entries = {}; chunks_exported_count = 0; success_flag = 0
    try:
        base_name, ext = os.path.splitext(filename); audio = AudioSegment.from_file(original_path)
        duration_ms = len(audio); duration_seconds = duration_ms / 1000.0
        if duration_seconds == 0: raise ValueError("Audio file is empty.")

        if duration_seconds <= min_duration_for_split_seconds: # File corto
            output_filename = f"{base_name}{ext}"; output_path = os.path.abspath(os.path.join(split_audio_dir, output_filename))
            if not os.path.exists(output_path):
                 try: audio.export(output_path, format=ext.lstrip('.'))
                 except Exception as export_err: print(f"  !!! ERROR exporting short {output_filename}: {export_err}"); raise
            file_manifest_entries[output_path] = {
                "original_file": original_path, "is_chunk": False, "chunk_index": 0, "start_time_abs": 0.0,
                "end_time_abs": duration_seconds, "original_duration_seconds": duration_seconds, "original_avg_dbfs": original_avg_dbfs }
            chunks_exported_count = 1; success_flag = 1
        else: # File lungo
            target_chunk_duration_ms = int(target_chunk_duration_seconds * 1000)
            num_ideal_chunks = math.ceil(duration_ms / target_chunk_duration_ms)
            cut_points_ms = [0]; last_cut_ms = 0
            for i in range(1, num_ideal_chunks): # Trova punti taglio
                ideal_cut_point_ms = min(duration_ms, last_cut_ms + target_chunk_duration_ms)
                search_start_ms = max(last_cut_ms + min_silence_len_ms, ideal_cut_point_ms - silence_search_range_ms)
                search_end_ms = min(duration_ms - min_silence_len_ms, ideal_cut_point_ms + silence_search_range_ms)
                actual_cut_point_ms = ideal_cut_point_ms
                if search_start_ms < search_end_ms:
                    audio_slice_for_search = audio[search_start_ms:search_end_ms]
                    # Usa soglie DINAMICHE per trovare il taglio
                    silences = detect_silence(audio_slice_for_search, min_silence_len_ms, silence_thresh_dbfs, seek_step=1)
                    if silences:
                        best_silence_center_ms = -1; min_dist_to_ideal = float('inf')
                        for s_start, s_end in silences:
                             s_center = search_start_ms + s_start + (s_end - s_start) / 2; dist = abs(s_center - ideal_cut_point_ms)
                             if dist < min_dist_to_ideal: min_dist_to_ideal = dist; best_silence_center_ms = int(s_center)
                        actual_cut_point_ms = best_silence_center_ms
                actual_cut_point_ms = int(round(max(last_cut_ms + 1000, min(duration_ms - 1000, actual_cut_point_ms))))
                if actual_cut_point_ms > last_cut_ms: cut_points_ms.append(actual_cut_point_ms); last_cut_ms = actual_cut_point_ms
                else: break
            cut_points_ms.append(duration_ms)
            # Esporta chunk
            chunks_exported_count = 0
            for i in range(len(cut_points_ms) - 1):
                 start_ms = cut_points_ms[i]; end_ms = cut_points_ms[i+1]
                 padded_start_ms = max(0, start_ms - keep_silence_ms); padded_end_ms = min(duration_ms, end_ms + keep_silence_ms)
                 if i > 0: padded_start_ms = max(padded_start_ms, cut_points_ms[i])
                 if i < len(cut_points_ms) - 2: padded_end_ms = min(padded_end_ms, cut_points_ms[i+1])
                 if padded_start_ms >= padded_end_ms: continue
                 chunk = audio[padded_start_ms:padded_end_ms]
                 chunk_filename = f"{base_name}_part{i:03d}{ext}"
                 chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
                 start_time_abs = padded_start_ms / 1000.0; end_time_abs = padded_end_ms / 1000.0
                 if not os.path.exists(chunk_output_path):
                     try: chunk.export(chunk_output_path, format=ext.lstrip('.'))
                     except Exception as export_err: print(f"  !!! ERROR exporting {chunk_filename}: {export_err}"); continue
                 file_manifest_entries[chunk_output_path] = {
                     "original_file": original_path, "is_chunk": True, "chunk_index": i,
                     "start_time_abs": start_time_abs, "end_time_abs": end_time_abs,
                     "original_duration_seconds": duration_seconds,
                     "original_avg_dbfs": original_avg_dbfs # Salva avg_dbfs
                 }
                 chunks_exported_count += 1
            if chunks_exported_count > 0: success_flag = 1
    except Exception as e: print(f"  !!! [SplitWorkerHybrid] ERROR processing {filename}: {e}"); import traceback; traceback.print_exc()
    return file_manifest_entries, success_flag, chunks_exported_count, 0

# --- Funzione Principale (Orchestra Analisi + Splitting Ibrido) ---
def split_large_audio_files(original_audio_dir: str, split_audio_dir: str,
                            # Parametri di default configurabili
                            target_chunk_duration_seconds: float = TARGET_CHUNK_DURATION_SECONDS,
                            min_duration_for_split_seconds: float = MIN_DURATION_FOR_SPLIT_SECONDS,
                            default_min_silence_len_ms: int = DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC,
                            default_silence_thresh_dbfs: float = DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC,
                            keep_silence_ms: int = DEFAULT_KEEP_SILENCE_MS,
                            silence_search_range_ms: int = SILENCE_SEARCH_RANGE_MS,
                            supported_extensions=(".flac",".m4a"),
                            num_workers: int | None = None
                            ) -> tuple[str | None, dict]:
    # Stampa parametri effettivi
    print(f"\n--- Avvio Pipeline di Splitting Parallelo (Analisi + Ibrido) ---")
    print(f"Directory originale: {original_audio_dir}")
    # ... (altre stampe parametri) ...
    if num_workers is None:
        try: num_workers = os.cpu_count(); num_workers = 4 if num_workers is None else num_workers
        except NotImplementedError: num_workers = 4
    print(f"Numero workers: {num_workers}")
    if not os.path.isdir(original_audio_dir): return None, {}
    try: os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e: print(f"Errore creazione directory split: {e}"); return None, {}

    # --- FASE 1: Analisi Parallela ---
    custom_params_dict = _run_parallel_audio_analysis(
        original_audio_dir, supported_extensions, num_workers,
        default_silence_thresh_dbfs, default_min_silence_len_ms
    )

    # --- FASE 2: Splitting/Copia Ibrida Parallela ---
    print(f"\n--- Avvio Splitting/Copia Ibrida Parallela (Workers: {num_workers}) ---")
    split_manifest = { # Inizializza manifest con tutti i parametri usati
        "split_method": "hybrid_silence_dynamic_threshold",
        "target_chunk_duration_seconds": target_chunk_duration_seconds,
        "min_duration_for_split_seconds": min_duration_for_split_seconds,
        "default_min_silence_len_ms": default_min_silence_len_ms,
        "default_silence_thresh_dbfs": default_silence_thresh_dbfs,
        "keep_silence_ms": keep_silence_ms,
        "silence_search_range_ms": silence_search_range_ms,
        "files": {} }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    total_files_processed_success = 0; total_chunks_exported = 0; total_errors = 0
    original_files = sorted(custom_params_dict.keys())
    if not original_files: return None, {}

    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    global splitting_pool_global_ref; splitting_pool_global_ref = None
    pool_error_occurred = False

    try:
        with context.Pool(processes=num_workers) as pool:
            splitting_pool_global_ref = pool
            tasks_args = []
            for original_path in original_files:
                params = custom_params_dict[original_path] # Recupera T, L, A
                tasks_args.append(
                    (original_path, split_audio_dir,
                     params['threshold'], params['min_len'], params['avg_dbfs'], # Passa i valori calcolati
                     keep_silence_ms, min_duration_for_split_seconds, # Passa globali/config
                     target_chunk_duration_seconds, silence_search_range_ms) )
            split_results = []
            try: split_results = pool.starmap(_process_single_audio_file_hybrid_split, tasks_args); print(f"\nCompletati {len(split_results)} task splitting.")
            except Exception as pool_error: pool_error_occurred = True; print(f"!!! Errore Pool Splitting: {pool_error}")
            if not pool_error_occurred: # Processa risultati solo se pool OK
                for result_tuple in split_results:
                    try:
                        file_manifest_entries, success_flag, chunks_exported, _ = result_tuple
                        if success_flag: total_files_processed_success += 1; split_manifest["files"].update(file_manifest_entries); total_chunks_exported += chunks_exported
                        else: total_errors += 1
                    except Exception as e: total_errors += 1; print(f"!!! Errore processando risultato split: {e}")
    finally: splitting_pool_global_ref = None

    if pool_error_occurred: print("Pipeline Splitting terminata con errori."); return None, {}
    print(f"\n--- Pipeline di Splitting (Analisi + Ibrido) Completata ---")
    print(f"File processati: {total_files_processed_success} (errori: {total_errors})")
    print(f"  - Totale chunk esportati: {total_chunks_exported}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")
    if not split_manifest["files"]: return None, {}
    try: # Salva manifest
        with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e: print(f"!!! Errore salvataggio manifest: {e}"); return None, {}

# Riferimenti globali per signal handler
analysis_pool_global_ref = None
splitting_pool_global_ref = None

# --- END OF transcriptionUtils/splitAudio.py (CORRETTO - Analisi + Splitting Ibrido + Salva AvgDBFS) ---