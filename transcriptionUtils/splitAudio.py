# --- START OF transcriptionUtils/splitAudio.py (MODIFICATO per Raggruppamento Aggressivo) ---

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

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
ANALYSIS_CHUNK_MS = 50

# --- PARAMETRI DI DEFAULT CONFIGURABILI (più aggressivi) ---
DEFAULT_SPLIT_NAMING_THRESHOLD_SECONDS: float = 45 * 60 # Soglia durata per nome '_partXXX'
DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC: int = 700        # Default MINIMO per analisi dinamica
DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC: float = -35.0     # Default per analisi dinamica
DEFAULT_KEEP_SILENCE_MS: int = 250                     # Padding silenzio (aumentato leggermente)
DEFAULT_MAX_SILENCE_BETWEEN_MS: int = 3500             # Max silenzio da includere in un chunk (3.5s - AUMENTATO)
DEFAULT_TARGET_MAX_CHUNK_DURATION_MS: int = 15 * 60 * 1000 # Max durata chunk (15 min - AUMENTATO)


# --- Worker Analisi Soglie (INVARIATO nel codice, ma usa nuovi default se analisi fallisce) ---
def _analyze_audio_for_thresholds(original_path: str,
                                  # I default passati qui sono quelli definiti sopra
                                  default_silence_thresh_dbfs: float,
                                  default_min_silence_len_ms: int
                                  ) -> tuple[str, float | None, int | None]:
    filename = os.path.basename(original_path)
    calculated_threshold = None
    calculated_min_len = None
    try:
        audio = AudioSegment.from_file(original_path)
        if len(audio) == 0: return original_path, None, None
        # Usa list comprehension per efficienza
        dbfs_values = [chunk.dBFS for i in range(0, len(audio), ANALYSIS_CHUNK_MS) if len(chunk := audio[i:i+ANALYSIS_CHUNK_MS]) > 0]
        valid_dbfs = [db for db in dbfs_values if np.isfinite(db)]
        if not valid_dbfs:
            calculated_threshold = default_silence_thresh_dbfs
        else:
            try:
                hist, bin_edges = np.histogram(valid_dbfs, bins='auto')
                bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
                noise_peak_estimate = np.percentile(valid_dbfs, 15)
                speech_peak_estimate = np.percentile(valid_dbfs, 80)
                noise_bin_idx = np.argmin(np.abs(bin_centers - noise_peak_estimate))
                speech_bin_idx = np.argmin(np.abs(bin_centers - speech_peak_estimate))
                start_idx = min(noise_bin_idx, speech_bin_idx)
                end_idx = max(noise_bin_idx, speech_bin_idx)
                valley_threshold = default_silence_thresh_dbfs
                if start_idx < end_idx:
                    try:
                        valley_idx_rel = np.argmin(hist[start_idx:end_idx+1])
                        valley_idx_abs = start_idx + valley_idx_rel
                        potential_threshold = bin_edges[valley_idx_abs + 1]
                        valley_threshold = max(-60.0, min(-15.0, potential_threshold))
                    except (ValueError, IndexError): valley_threshold = default_silence_thresh_dbfs
                else: valley_threshold = max(-60.0, min(-15.0, noise_peak_estimate + 10.0))
                calculated_threshold = valley_threshold
            except Exception: calculated_threshold = default_silence_thresh_dbfs
        if calculated_threshold is not None:
            try:
                # Trova silenzi >= 100ms per l'analisi della durata
                silence_ranges = detect_silence(audio, 100, calculated_threshold, 1)
                silence_durations_ms = [(end - start) for start, end in silence_ranges]
                meaningful_silences_ms = [d for d in silence_durations_ms if d >= 150]
                if meaningful_silences_ms:
                    potential_min_len = np.percentile(meaningful_silences_ms, 65)
                    # Limiti più aggressivi: min 500ms, max 4000ms? Dipende molto dall'audio.
                    # Manteniamo limiti precedenti per ora: min 250, max 2000
                    calculated_min_len = int(max(250, min(2000, potential_min_len)))
                else: calculated_min_len = default_min_silence_len_ms
            except Exception: calculated_min_len = default_min_silence_len_ms
        else: calculated_min_len = default_min_silence_len_ms
    except Exception as e: print(f"  !!! [AnalysisWorker] ERROR analyzing {filename}: {e}")
    return original_path, calculated_threshold, calculated_min_len


# --- Funzione Orchestratrice Analisi (INVARIATA, passa i nuovi default) ---
def _run_parallel_audio_analysis(original_audio_dir: str,
                                 supported_extensions: tuple,
                                 num_workers: int,
                                 # Passa i default definiti a livello di modulo
                                 default_silence_thresh_dbfs: float = DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC,
                                 default_min_silence_len_ms: int = DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC
                                 ) -> dict[str, dict]:
    print(f"\n--- Avvio Analisi Audio Parallela per Soglie Dinamiche (Workers: {num_workers}) ---")
    custom_params_dict: dict[str, dict] = {}
    original_files = sorted([f for f in os.listdir(original_audio_dir) if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)])
    if not original_files: return custom_params_dict
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    with context.Pool(processes=num_workers) as pool:
        # Passa i default globali al worker
        tasks_args = [(os.path.abspath(os.path.join(original_audio_dir, filename)), default_silence_thresh_dbfs, default_min_silence_len_ms) for filename in original_files]
        analysis_results = []
        try:
            analysis_results = pool.starmap(_analyze_audio_for_thresholds, tasks_args)
            print(f"Completata analisi preliminare per {len(analysis_results)} file.")
        except Exception as pool_error: print(f"!!! Errore durante l'esecuzione del Pool di Analisi: {pool_error}"); return custom_params_dict
        # Popola il dizionario usando i default come fallback
        for original_path, calc_thresh, calc_min_len in analysis_results:
            custom_params_dict[original_path] = {'threshold': calc_thresh if calc_thresh is not None else default_silence_thresh_dbfs, 'min_len': calc_min_len if calc_min_len is not None else default_min_silence_len_ms}
    print("--- Fine Analisi Audio Parallela ---")
    return custom_params_dict


# --- Worker Splitting (INVARIATO nella logica, usa parametri passati) ---
def _process_single_audio_file_for_split(original_path: str,
                                         split_audio_dir: str,
                                         silence_thresh_dbfs: float,
                                         min_silence_len_ms: int,
                                         keep_silence_ms: int,
                                         split_naming_threshold_seconds: float,
                                         max_silence_between_ms: int,
                                         target_max_chunk_duration_ms: int
                                         ) -> tuple[dict, int, int, int]:
    filename = os.path.basename(original_path)
    file_manifest_entries = {}
    chunks_exported_count = 0
    success_flag = 0
    try:
        base_name, ext = os.path.splitext(filename)
        audio = AudioSegment.from_file(original_path)
        duration_seconds = len(audio) / 1000.0
        if duration_seconds == 0: raise ValueError("Audio file is empty.")

        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_dbfs, seek_step=1)
        if not nonsilent_ranges: return {}, 0, 0, 0

        print(f"  [SplitWorker {os.getpid()}] Found {len(nonsilent_ranges)} raw non-silent ranges for {filename}. Grouping (MaxSilence={max_silence_between_ms}ms, MaxDur={target_max_chunk_duration_ms/1000/60:.1f}min)...")
        grouped_ranges = []
        if nonsilent_ranges: # Assicurati che non sia vuoto
            current_group_start, current_group_end = nonsilent_ranges[0]
            current_group_duration = current_group_end - current_group_start

            for i in range(len(nonsilent_ranges) - 1):
                silence_between = nonsilent_ranges[i+1][0] - nonsilent_ranges[i][1]
                next_segment_duration = nonsilent_ranges[i+1][1] - nonsilent_ranges[i+1][0]

                # Verifica se aggiungere il prossimo segmento è possibile
                if silence_between < max_silence_between_ms and \
                   (current_group_duration + silence_between + next_segment_duration) <= target_max_chunk_duration_ms:
                    current_group_end = nonsilent_ranges[i+1][1]
                    current_group_duration = current_group_end - current_group_start
                else:
                    grouped_ranges.append((current_group_start, current_group_end))
                    current_group_start, current_group_end = nonsilent_ranges[i+1]
                    current_group_duration = current_group_end - current_group_start
            grouped_ranges.append((current_group_start, current_group_end)) # Aggiungi l'ultimo gruppo

        print(f"  [SplitWorker {os.getpid()}] Grouped into {len(grouped_ranges)} final chunks for {filename}.")

        chunks_exported_count = 0
        is_considered_long = duration_seconds > split_naming_threshold_seconds

        for i, (start_ms, end_ms) in enumerate(grouped_ranges):
             padded_start_ms = max(0, start_ms - keep_silence_ms)
             padded_end_ms = min(len(audio), end_ms + keep_silence_ms)
             if padded_start_ms >= padded_end_ms: continue
             chunk = audio[padded_start_ms:padded_end_ms]

             if is_considered_long or len(grouped_ranges) > 1:
                  chunk_filename = f"{base_name}_part{i:03d}{ext}"
                  is_chunk_flag = True; chunk_index = i
             else:
                  chunk_filename = f"{base_name}{ext}"
                  is_chunk_flag = False; chunk_index = 0

             chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
             start_time_abs = padded_start_ms / 1000.0
             end_time_abs = padded_end_ms / 1000.0

             if not os.path.exists(chunk_output_path):
                 try: chunk.export(chunk_output_path, format=ext.lstrip('.'))
                 except Exception as export_err: print(f"  !!! ERROR exporting {chunk_filename}: {export_err}"); continue

             file_manifest_entries[chunk_output_path] = {
                 "original_file": original_path, "is_chunk": is_chunk_flag, "chunk_index": chunk_index,
                 "start_time_abs": start_time_abs, "end_time_abs": end_time_abs,
                 "original_duration_seconds": duration_seconds
             }
             chunks_exported_count += 1
        if chunks_exported_count > 0: success_flag = 1

    except Exception as e: print(f"  !!! [SplitWorker {os.getpid()}] ERROR processing {filename}: {e}"); import traceback; traceback.print_exc()

    return file_manifest_entries, success_flag, chunks_exported_count, 0


# --- Funzione Principale (MODIFICATA per usare i nuovi default) ---
def split_large_audio_files(original_audio_dir: str,
                            split_audio_dir: str,
                            # Usa i default definiti all'inizio
                            split_naming_threshold_seconds: float = DEFAULT_SPLIT_NAMING_THRESHOLD_SECONDS,
                            default_min_silence_len_ms: int = DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC,
                            default_silence_thresh_dbfs: float = DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC,
                            keep_silence_ms: int = DEFAULT_KEEP_SILENCE_MS,
                            max_silence_between_ms: int = DEFAULT_MAX_SILENCE_BETWEEN_MS,
                            target_max_chunk_duration_ms: int = DEFAULT_TARGET_MAX_CHUNK_DURATION_MS,
                            supported_extensions=(".flac",".m4a"),
                            num_workers: int | None = None
                            ) -> tuple[str | None, dict]:
    """
    Esegue pipeline di splitting: analisi dinamica soglie, poi splitting/raggruppamento.
    """
    # Stampa i valori EFFETTIVI che verranno usati (i default o quelli passati)
    print(f"\n--- Avvio Pipeline di Splitting Parallelo (con Raggruppamento Aggressivo) ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Soglia durata per nome '_partXXX': {split_naming_threshold_seconds / 60:.1f} minuti")
    print(f"Parametri Silenzio di Default/Fallback: min_len={default_min_silence_len_ms}ms, threshold={default_silence_thresh_dbfs}dBFS")
    print(f"Padding Silenzio: keep_silence={keep_silence_ms}ms")
    print(f"Parametri Raggruppamento: max_silence_between={max_silence_between_ms}ms, max_chunk_duration={target_max_chunk_duration_ms / 1000 / 60:.1f}min")
    print(f"Estensioni supportate: {supported_extensions}")

    # ... (Logica num_workers, check dir, creazione dir principale) ...
    if num_workers is None:
        try: num_workers = os.cpu_count(); num_workers = 4 if num_workers is None else num_workers
        except NotImplementedError: num_workers = 4
        print(f"Numero workers: {num_workers}")
    if not os.path.isdir(original_audio_dir): return None, {}
    try: os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e: print(f"Errore creazione directory split: {e}"); return None, {}

    # --- FASE 1: Analisi Parallela (Passa i default globali) ---
    custom_params_dict = _run_parallel_audio_analysis(
        original_audio_dir, supported_extensions, num_workers,
        default_silence_thresh_dbfs, default_min_silence_len_ms # Passa i default
    )

    # --- FASE 2: Splitting/Copia Parallela (Passa parametri specifici e di raggruppamento) ---
    print(f"\n--- Avvio Splitting/Copia/Raggruppamento Parallelo (Workers: {num_workers}) ---")
    split_manifest = {
        "split_method": "silence_detection_dynamic_threshold_aggressively_grouped", # Nuovo nome metodo
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

    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    with context.Pool(processes=num_workers) as pool:
        tasks_args = []
        for original_path in original_files:
            params = custom_params_dict[original_path] # Recupera soglie dinamiche
            tasks_args.append(
                (original_path, split_audio_dir,
                 params['threshold'], params['min_len'], # Soglie dinamiche
                 keep_silence_ms, split_naming_threshold_seconds, # Parametri globali
                 max_silence_between_ms, target_max_chunk_duration_ms # PARAMETRI RAGGRUPPAMENTO
                 )
            )
        split_results = []
        try:
            split_results = pool.starmap(_process_single_audio_file_for_split, tasks_args)
            print(f"\nCompletati {len(split_results)} task di splitting/copy/grouping dalla pool.")
        except Exception as pool_error: print(f"!!! Errore Pool Splitting: {pool_error}"); return None, {}

        for result_tuple in split_results:
            try:
                file_manifest_entries, success_flag, chunks_exported, _ = result_tuple
                if success_flag:
                    total_files_processed_success += 1
                    split_manifest["files"].update(file_manifest_entries)
                    total_chunks_exported += chunks_exported
                else: total_errors += 1
            except Exception as e: total_errors += 1; print(f"!!! Errore processando risultato split: {e}")

    # --- Riepilogo Finale ---
    print(f"\n--- Pipeline di Splitting (Raggruppamento Aggressivo) Completata ---")
    print(f"File originali processati: {total_files_processed_success} (con errori: {total_errors})")
    print(f"  - Totale chunk finali esportati: {total_chunks_exported}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")

    if not split_manifest["files"]: return None, {}

    # Salva manifest
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e: print(f"!!! Errore salvataggio manifest: {e}"); return None, {}

# --- END OF transcriptionUtils/splitAudio.py (MODIFICATO per Raggruppamento Aggressivo) ---