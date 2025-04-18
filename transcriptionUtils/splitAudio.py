# --- START OF transcriptionUtils/splitAudio.py (MODIFICATO per Raggruppamento Chunk) ---

import os
import json
import math
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
from pydub.silence import detect_nonsilent # type: ignore
import shutil
import numpy as np
import platform
import multiprocessing as mp

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
ANALYSIS_CHUNK_MS = 50

# --- Variabili Globali per i Pool ---
# Verranno popolate quando i pool sono attivi
analysis_pool_global_ref = None
splitting_pool_global_ref = None

# --- Worker Analisi Soglie (INVARIATO rispetto a prima) ---
def _analyze_audio_for_thresholds(original_path: str,
                                  default_silence_thresh_dbfs: float,
                                  default_min_silence_len_ms: int
                                  ) -> tuple[str, float | None, int | None]:
    # ... (Codice completo del worker di analisi qui, esattamente come nella risposta precedente) ...
    filename = os.path.basename(original_path)
    calculated_threshold = None
    calculated_min_len = None
    try:
        audio = AudioSegment.from_file(original_path)
        if len(audio) == 0: return original_path, None, None
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
                silence_ranges = detect_silence(audio, 100, calculated_threshold, 1)
                silence_durations_ms = [(end - start) for start, end in silence_ranges]
                meaningful_silences_ms = [d for d in silence_durations_ms if d >= 150]
                if meaningful_silences_ms:
                    potential_min_len = np.percentile(meaningful_silences_ms, 65)
                    calculated_min_len = int(max(250, min(2000, potential_min_len)))
                else: calculated_min_len = default_min_silence_len_ms
            except Exception: calculated_min_len = default_min_silence_len_ms
        else: calculated_min_len = default_min_silence_len_ms
    except Exception as e: print(f"  !!! [AnalysisWorker] ERROR analyzing {filename}: {e}")
    return original_path, calculated_threshold, calculated_min_len

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


# --- WORKER per lo SPLITTING (MODIFICATO per RAGGRUPPARE segmenti) ---
def _process_single_audio_file_for_split(original_path: str,
                                         split_audio_dir: str,
                                         silence_thresh_dbfs: float,
                                         min_silence_len_ms: int,
                                         keep_silence_ms: int,
                                         split_naming_threshold_seconds: float,
                                         # NUOVI PARAMETRI PER RAGGRUPPAMENTO
                                         max_silence_between_ms: int,
                                         target_max_chunk_duration_ms: int
                                         ) -> tuple[dict, int, int, int]:
    """
    Worker per dividere/copiare usando soglie specifiche e RAGGRUPPANDO
    segmenti di parlato vicini.
    """
    filename = os.path.basename(original_path)
    # print(f"  [SplitWorker {os.getpid()}] Processing: {filename} with Threshold={silence_thresh_dbfs:.1f}, MinLen={min_silence_len_ms}") # Meno verboso
    file_manifest_entries = {}
    chunks_exported_count = 0
    success_flag = 0

    try:
        base_name, ext = os.path.splitext(filename)
        audio = AudioSegment.from_file(original_path)
        duration_seconds = len(audio) / 1000.0
        if duration_seconds == 0: raise ValueError("Audio file is empty.")

        # --- 1. Rileva segmenti NON silenziosi ---
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len_ms, # Usa min_len dinamico qui
            silence_thresh=silence_thresh_dbfs,
            seek_step=1
        )

        if not nonsilent_ranges:
             print(f"  [SplitWorker {os.getpid()}] WARN: detect_nonsilent non ha trovato segmenti per {filename}. Skipping.")
             return {}, 0, 0, 0

        # --- 2. Logica di Raggruppamento ---
        print(f"  [SplitWorker {os.getpid()}] Found {len(nonsilent_ranges)} raw non-silent ranges for {filename}. Grouping...")
        grouped_ranges = [] # Lista di tuple: (start_ms_agg, end_ms_agg)
        if not nonsilent_ranges: # Check di sicurezza
            pass # Nessun range, la lista rimane vuota
        else:
            # Inizia il primo gruppo con il primo segmento
            current_group_start, current_group_end = nonsilent_ranges[0]
            current_group_duration = current_group_end - current_group_start

            for i in range(len(nonsilent_ranges) - 1):
                # Silenzio tra la fine del segmento corrente e l'inizio del successivo
                silence_between = nonsilent_ranges[i+1][0] - nonsilent_ranges[i][1]
                next_segment_duration = nonsilent_ranges[i+1][1] - nonsilent_ranges[i+1][0]

                # Condizioni per aggiungere il segmento successivo al gruppo corrente:
                # 1. Il silenzio tra i due è abbastanza corto
                # 2. Aggiungere il prossimo segmento non supera la durata massima target
                if silence_between < max_silence_between_ms and \
                   (current_group_duration + silence_between + next_segment_duration) <= target_max_chunk_duration_ms:
                    # Aggiungi il segmento successivo al gruppo corrente estendendo la fine
                    current_group_end = nonsilent_ranges[i+1][1]
                    # Aggiorna la durata del gruppo corrente (approssimata includendo il silenzio interno)
                    current_group_duration = current_group_end - current_group_start
                else:
                    # Il segmento successivo non può essere aggiunto (silenzio troppo lungo o chunk troppo lungo)
                    # Finalizza il gruppo corrente
                    grouped_ranges.append((current_group_start, current_group_end))
                    # Inizia un nuovo gruppo con il segmento successivo
                    current_group_start, current_group_end = nonsilent_ranges[i+1]
                    current_group_duration = current_group_end - current_group_start

            # Aggiungi l'ultimo gruppo formato
            grouped_ranges.append((current_group_start, current_group_end))

        print(f"  [SplitWorker {os.getpid()}] Grouped into {len(grouped_ranges)} final chunks for {filename}.")

        # --- 3. Esportazione dei Chunk Raggruppati ---
        chunks_exported_count = 0
        is_considered_long = duration_seconds > split_naming_threshold_seconds

        for i, (start_ms, end_ms) in enumerate(grouped_ranges):
             # Applica padding al gruppo aggregato
             padded_start_ms = max(0, start_ms - keep_silence_ms)
             padded_end_ms = min(len(audio), end_ms + keep_silence_ms)
             if padded_start_ms >= padded_end_ms: continue

             chunk = audio[padded_start_ms:padded_end_ms]

             # Nomenclatura: usa _partXXX se l'originale era lungo O se ci sono >1 chunk finali
             if is_considered_long or len(grouped_ranges) > 1:
                  chunk_filename = f"{base_name}_part{i:03d}{ext}"
                  is_chunk_flag = True
                  chunk_index = i
             else: # Solo un chunk finale da un file originale corto
                  chunk_filename = f"{base_name}{ext}"
                  is_chunk_flag = False
                  chunk_index = 0

             chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
             start_time_abs = padded_start_ms / 1000.0
             end_time_abs = padded_end_ms / 1000.0

             if not os.path.exists(chunk_output_path):
                 try:
                     # print(f"    Exporting: {chunk_filename} [{start_time_abs:.2f}s - {end_time_abs:.2f}s]") # Verboso
                     chunk.export(chunk_output_path, format=ext.lstrip('.'))
                 except Exception as export_err:
                      print(f"  !!! [SplitWorker {os.getpid()}] ERROR exporting {chunk_filename}: {export_err}. Skipping chunk.")
                      continue

             file_manifest_entries[chunk_output_path] = {
                 "original_file": original_path, "is_chunk": is_chunk_flag,
                 "chunk_index": chunk_index, "start_time_abs": start_time_abs,
                 "end_time_abs": end_time_abs, "original_duration_seconds": duration_seconds
             }
             chunks_exported_count += 1

        if chunks_exported_count > 0: success_flag = 1

    # --- Gestione Errori (invariata) ---
    except pydub_exceptions.CouldntDecodeError as e: print(f"  !!! [SplitWorker {os.getpid()}] ERROR decoding {filename}: {e}. Skipping.")
    except FileNotFoundError: print(f"  !!! [SplitWorker {os.getpid()}] ERROR File not found: {original_path}. Skipping.")
    except ValueError as ve: print(f"  !!! [SplitWorker {os.getpid()}] ERROR processing {filename}: {ve}. Skipping.")
    except Exception as e:
        print(f"  !!! [SplitWorker {os.getpid()}] UNEXPECTED ERROR processing {filename}: {e}")
        import traceback; traceback.print_exc()

    return file_manifest_entries, success_flag, chunks_exported_count, 0


# --- Funzione Principale split_large_audio_files (MODIFICATA per pool globale) ---
def split_large_audio_files(original_audio_dir: str, split_audio_dir: str, split_naming_threshold_seconds: float = 45 * 60, default_min_silence_len_ms: int = 700, default_silence_thresh_dbfs: float = -35.0, keep_silence_ms: int = 150, max_silence_between_ms: int = 1500, target_max_chunk_duration_ms: int = 10 * 60 * 1000, supported_extensions=(".flac",".m4a"), num_workers: int | None = None) -> tuple[str | None, dict]:
    global splitting_pool_global_ref # Usa la variabile globale
    """
    Esegue analisi dinamica soglie, poi divide/copia file audio in parallelo
    raggruppando segmenti vicini, e crea un manifest JSON.
    """
    # ... (Stampa parametri iniziali, inclusi i nuovi default di raggruppamento) ...
    print(f"\n--- Avvio Pipeline di Splitting Parallelo (con Raggruppamento) ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Soglia durata per nome '_partXXX': {split_naming_threshold_seconds / 60:.1f} minuti")
    print(f"Parametri Silenzio di Default: min_len={default_min_silence_len_ms}ms, threshold={default_silence_thresh_dbfs}dBFS")
    print(f"Padding Silenzio: keep_silence={keep_silence_ms}ms")
    print(f"Parametri Raggruppamento: max_silence_between={max_silence_between_ms}ms, max_chunk_duration={target_max_chunk_duration_ms / 1000 / 60:.1f}min")
    print(f"Estensioni supportate: {supported_extensions}")

    # ... (Logica num_workers, check dir, creazione dir principale) ...
    if num_workers is None:
        try: num_workers = os.cpu_count(); num_workers = 4 if num_workers is None else num_workers
        except NotImplementedError: num_workers = 4
        print(f"Numero workers: {num_workers} (default o specificato)")
    if not os.path.isdir(original_audio_dir): return None, {}
    try: os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e: print(f"Errore creazione directory split: {e}"); return None, {}


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