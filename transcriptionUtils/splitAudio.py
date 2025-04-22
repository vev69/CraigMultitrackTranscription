# --- START OF transcriptionUtils/splitAudio.py (v6 - Analisi Completa + Splitting Ibrido) ---

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
import warnings

# Import per LUFS
try: import pyloudnorm as pyln; PYLOUDNORM_AVAILABLE = True # type: ignore
except ImportError: print("WARN: pyloudnorm non trovato!"); PYLOUDNORM_AVAILABLE = False

# Ignora warning comuni
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
ANALYSIS_CHUNK_MS = 50

# --- PARAMETRI DI DEFAULT ---
DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC: int = 700
DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC: float = -35.0
TARGET_CHUNK_DURATION_SECONDS: float = 10 * 60  # 10 minuti
MIN_DURATION_FOR_SPLIT_SECONDS: float = TARGET_CHUNK_DURATION_SECONDS + (1 * 60)
DEFAULT_KEEP_SILENCE_MS: int = 250
SILENCE_SEARCH_RANGE_MS: int = 10 * 1000
MAX_TARGET_AVG_DBFS_LIMIT: float = -6.0 # Limite per target basato sul max RMS/LUFS
FALLBACK_TARGET_LUFS: float = -19.0 # Target LUFS se il calcolo fallisce o è unico file


# --- WORKER: Analisi Audio (Calcola TUTTE le metriche necessarie) ---
def _analyze_audio_for_params(original_path: str,
                              default_silence_thresh_dbfs: float,
                              default_min_silence_len_ms: int
                              ) -> tuple[str, dict]: # Restituisce dict con tutte le metriche
    """
    Worker per analizzare un file audio e determinare parametri completi.
    """
    filename = os.path.basename(original_path)
    metrics = { # Inizializza con valori di default/errore
        'threshold': default_silence_thresh_dbfs,
        'min_len': default_min_silence_len_ms,
        'avg_dbfs': -999.0,
        'loudness_lufs': None, # Cruciale per normalizzazione target
        'peak_dbfs': -np.inf,
        'noise_floor_15p_dbfs': None,
        'est_snr_80_15_db': None
    }
    try:
        audio = AudioSegment.from_file(original_path)
        if len(audio) == 0: return original_path, metrics # Ritorna default se vuoto

        # Calcola avg_dbfs
        if audio.rms > 0: metrics['avg_dbfs'] = audio.dBFS
        else: metrics['avg_dbfs'] = -np.inf

        # Calcola Peak dBFS
        samples = np.array(audio.get_array_of_samples())
        float_samples = None
        if samples.size > 0:
            bits = audio.sample_width * 8
            if bits > 0:
                norm_factor = 1 << (bits - 1)
                float_samples = samples.astype(np.float32) / norm_factor
                peak_linear = np.max(np.abs(float_samples));
                if peak_linear > 1e-9: metrics['peak_dbfs'] = 20 * np.log10(peak_linear)

        # Calcola LUFS (se possibile e ci sono campioni float)
        if PYLOUDNORM_AVAILABLE and float_samples is not None and float_samples.size > 0:
             try:
                 # Assicurati sia mono per pyloudnorm
                 mono_float = np.mean(float_samples.reshape(-1, audio.channels), axis=1) if audio.channels > 1 else float_samples
                 meter = pyln.Meter(audio.frame_rate); loudness = meter.integrated_loudness(mono_float)
                 if np.isfinite(loudness): metrics['loudness_lufs'] = loudness
             except Exception: pass # Ignora errori LUFS

        # Calcola Soglia Dinamica e Min Len (come prima)
        dbfs_values = [chunk.dBFS for i in range(0, len(audio), ANALYSIS_CHUNK_MS) if len(chunk := audio[i:i+ANALYSIS_CHUNK_MS]) > 0]
        valid_dbfs = [db for db in dbfs_values if np.isfinite(db)]
        if valid_dbfs:
            try: # Istogramma per Soglia
                hist, bin_edges = np.histogram(valid_dbfs, bins='auto'); bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
                n15=np.percentile(valid_dbfs,15); s80=np.percentile(valid_dbfs,80); n_idx=np.argmin(np.abs(bin_centers-n15)); s_idx=np.argmin(np.abs(bin_centers-s80))
                start_idx, end_idx = min(n_idx,s_idx), max(n_idx,s_idx); valley_thresh=metrics['threshold']
                if start_idx < end_idx:
                    try: v_idx_rel=np.argmin(hist[start_idx:end_idx+1]); v_idx_abs=start_idx+v_idx_rel; pot_thresh=bin_edges[v_idx_abs+1]; valley_thresh=max(-60.0,min(-15.0,pot_thresh))
                    except: pass
                else: valley_thresh=max(-60.0,min(-15.0,n15+10.0))
                metrics['threshold'] = valley_thresh
                # Calcola Min Len
                try:
                    sil_ranges=detect_silence(audio,100,metrics['threshold'],1); sil_durs=[e-s for s,e in sil_ranges]; mean_sil=[d for d in sil_durs if d>=150]
                    if mean_sil: pot_min=np.percentile(mean_sil,65); metrics['min_len']=int(max(250,min(2000,pot_min)))
                except: pass # Usa default se fallisce
                # Calcola SNR stimato
                metrics['noise_floor_15p_dbfs'] = n15
                if np.isfinite(n15) and np.isfinite(s80): metrics['est_snr_80_15_db'] = s80 - n15
            except Exception: pass # Usa default se fallisce

    except Exception as e: print(f"  !!! [AnalysisWorker] ERROR analyzing {filename}: {e}")
    return original_path, metrics


# --- Funzione Orchestratrice Analisi (Restituisce dizionario e target LUFS) ---
def _run_parallel_audio_analysis(original_audio_dir: str, supported_extensions: tuple, num_workers: int,
                                 default_silence_thresh_dbfs: float, default_min_silence_len_ms: int
                                 ) -> tuple[dict[str, dict], float]:
    print(f"\n--- Avvio Analisi Audio Parallela (Params + Target LUFS) ---")
    custom_params_dict: dict[str, dict] = {}
    effective_target_lufs = FALLBACK_TARGET_LUFS # Fallback
    # ... (Trova original_files) ...
    original_files = sorted([f for f in os.listdir(original_audio_dir) if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)])
    if not original_files: return custom_params_dict, effective_target_lufs

    # ... (Setup Pool) ...
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    global analysis_pool_global_ref; analysis_pool_global_ref = None
    all_valid_lufs = [] # Raccogli LUFS validi

    try:
        with context.Pool(processes=num_workers) as pool:
            analysis_pool_global_ref = pool
            tasks_args = [(os.path.abspath(os.path.join(original_audio_dir, filename)), default_silence_thresh_dbfs, default_min_silence_len_ms) for filename in original_files]
            analysis_results = []; pool_error_occurred = False
            try: analysis_results = pool.starmap(_analyze_audio_for_params, tasks_args); print(f"Completata analisi per {len(analysis_results)} file.")
            except Exception as pool_error: print(f"!!! Errore Pool Analisi: {pool_error}"); pool_error_occurred = True
            if pool_error_occurred: return {}, effective_target_lufs

            # Popola dizionario e raccogli LUFS
            for original_path, params_dict in analysis_results:
                 custom_params_dict[original_path] = params_dict
                 lufs_val = params_dict.get('loudness_lufs')
                 if lufs_val is not None and np.isfinite(lufs_val):
                      all_valid_lufs.append(lufs_val)

            # Calcola Target LUFS Effettivo basato sul massimo
            if all_valid_lufs:
                max_lufs = max(all_valid_lufs)
                # Limita il target LUFS a un massimo ragionevole (es. -14 LUFS) e non più basso del fallback
                effective_target_lufs = max(FALLBACK_TARGET_LUFS, min(-14.0, max_lufs)) # Usa -14 come cap
                print(f"Loudness max rilevato: {max_lufs:.1f} LUFS. Target effettivo impostato a: {effective_target_lufs:.1f} LUFS.")
            else: print(f"WARN: Impossibile determinare Loudness max, uso default: {FALLBACK_TARGET_LUFS:.1f} LUFS.")

    finally: analysis_pool_global_ref = None
    print("--- Fine Analisi Audio Parallela ---")
    return custom_params_dict, effective_target_lufs


# --- WORKER Splitting (Riceve dict parametri e salva metriche rilevanti) ---
def _process_single_audio_file_hybrid_split(original_path: str, split_audio_dir: str,
                                            # Riceve l'intero dizionario dei parametri calcolati
                                            original_file_params: dict,
                                            # Parametri globali/configurabili
                                            keep_silence_ms: int,
                                            min_duration_for_split_seconds: float,
                                            target_chunk_duration_seconds: float,
                                            silence_search_range_ms: int
                                            ) -> tuple[dict, int, int, int]:
    filename = os.path.basename(original_path); file_manifest_entries = {}; chunks_exported_count = 0; success_flag = 0
    try:
        base_name, ext = os.path.splitext(filename); audio = AudioSegment.from_file(original_path)
        duration_ms = len(audio); duration_seconds = duration_ms / 1000.0
        if duration_seconds == 0: raise ValueError("Audio file is empty.")

        # Estrai parametri specifici per questo file dal dizionario
        silence_thresh_dbfs = original_file_params['threshold']
        min_silence_len_ms = original_file_params['min_len']
        original_avg_dbfs = original_file_params['avg_dbfs'] # Sarà salvato nel manifest
        # Potremmo salvare anche altre metriche se utili dopo
        original_lufs = original_file_params['loudness_lufs']
        original_snr = original_file_params['est_snr_80_15_db']

        if duration_seconds <= min_duration_for_split_seconds: # File corto
            output_filename = f"{base_name}{ext}"; output_path = os.path.abspath(os.path.join(split_audio_dir, output_filename))
            if not os.path.exists(output_path):
                 try: audio.export(output_path, format=ext.lstrip('.'))
                 except Exception as export_err: print(f"  !!! ERROR exporting short {output_filename}: {export_err}"); raise
            file_manifest_entries[output_path] = { # Salva metriche rilevanti
                "original_file": original_path, "is_chunk": False, "chunk_index": 0, "start_time_abs": 0.0,
                "end_time_abs": duration_seconds, "original_duration_seconds": duration_seconds,
                "original_avg_dbfs": original_avg_dbfs, "original_lufs": original_lufs, "original_snr": original_snr }
            chunks_exported_count = 1; success_flag = 1
        else: # File lungo (Logica splitting ibrido invariata, usa soglie dinamiche)
            target_chunk_duration_ms = int(target_chunk_duration_seconds * 1000)
            num_ideal_chunks = math.ceil(duration_ms / target_chunk_duration_ms)
            cut_points_ms = [0]; last_cut_ms = 0
            for i in range(1, num_ideal_chunks): # Trova punti taglio
                ideal_cut_point_ms = min(duration_ms, last_cut_ms + target_chunk_duration_ms)
                search_start_ms = max(last_cut_ms + min_silence_len_ms, ideal_cut_point_ms - silence_search_range_ms)
                search_end_ms = min(duration_ms - min_silence_len_ms, ideal_cut_point_ms + silence_search_range_ms)
                actual_cut_point_ms = ideal_cut_point_ms
                if search_start_ms < search_end_ms:
                    silences = detect_silence(audio[search_start_ms:search_end_ms], min_silence_len_ms, silence_thresh_dbfs, 1)
                    if silences:
                        best_silence_center_ms = -1; min_dist_to_ideal = float('inf')
                        for s_start, s_end in silences: s_center=search_start_ms+s_start+(s_end-s_start)/2; dist=abs(s_center-ideal_cut_point_ms);
                        if dist < min_dist_to_ideal: min_dist_to_ideal = dist; best_silence_center_ms = int(s_center)
                        actual_cut_point_ms = best_silence_center_ms
                actual_cut_point_ms = int(round(max(last_cut_ms + 1000, min(duration_ms - 1000, actual_cut_point_ms))))
                if actual_cut_point_ms > last_cut_ms: cut_points_ms.append(actual_cut_point_ms); last_cut_ms = actual_cut_point_ms
                else: break
            cut_points_ms.append(duration_ms)
            # Esporta chunk
            chunks_exported_count = 0
            for i in range(len(cut_points_ms) - 1):
                 start_ms=cut_points_ms[i]; end_ms=cut_points_ms[i+1]; padded_start_ms=max(0, start_ms-keep_silence_ms); padded_end_ms=min(duration_ms, end_ms+keep_silence_ms)
                 if i > 0: padded_start_ms = max(padded_start_ms, cut_points_ms[i])
                 if i < len(cut_points_ms) - 2: padded_end_ms = min(padded_end_ms, cut_points_ms[i+1])
                 if padded_start_ms >= padded_end_ms: continue
                 chunk = audio[padded_start_ms:padded_end_ms]; chunk_filename = f"{base_name}_part{i:03d}{ext}"
                 chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
                 start_time_abs = padded_start_ms/1000.0; end_time_abs = padded_end_ms/1000.0
                 if not os.path.exists(chunk_output_path):
                     try: chunk.export(chunk_output_path, format=ext.lstrip('.'))
                     except Exception as export_err: print(f"  !!! ERROR exporting {chunk_filename}: {export_err}"); continue
                 file_manifest_entries[chunk_output_path] = { # Salva metriche rilevanti
                     "original_file": original_path, "is_chunk": True, "chunk_index": i, "start_time_abs": start_time_abs,
                     "end_time_abs": end_time_abs, "original_duration_seconds": duration_seconds,
                     "original_avg_dbfs": original_avg_dbfs, "original_lufs": original_lufs, "original_snr": original_snr }
                 chunks_exported_count += 1
            if chunks_exported_count > 0: success_flag = 1
    except Exception as e: print(f"  !!! [SplitWorkerHybrid] ERROR processing {filename}: {e}"); import traceback; traceback.print_exc()
    return file_manifest_entries, success_flag, chunks_exported_count, 0


# --- Funzione Principale (Orchestra Analisi + Splitting, salva target LUFS nel manifest) ---
def split_large_audio_files(original_audio_dir: str, split_audio_dir: str,
                            target_chunk_duration_seconds: float = TARGET_CHUNK_DURATION_SECONDS,
                            min_duration_for_split_seconds: float = MIN_DURATION_FOR_SPLIT_SECONDS,
                            default_min_silence_len_ms: int = DEFAULT_MIN_SILENCE_LEN_MS_DYNAMIC,
                            default_silence_thresh_dbfs: float = DEFAULT_SILENCE_THRESH_DBFS_DYNAMIC,
                            keep_silence_ms: int = DEFAULT_KEEP_SILENCE_MS,
                            silence_search_range_ms: int = SILENCE_SEARCH_RANGE_MS,
                            supported_extensions=(".flac",".m4a"),
                            num_workers: int | None = None
                            ) -> tuple[str | None, dict]:
    # ... (Stampa parametri iniziali) ...
    # ... (Logica num_workers, check dirs...) ...

    # --- FASE 1: Analisi Parallela -> Ottiene custom_params E effective_target_lufs ---
    custom_params_dict, effective_target_lufs = _run_parallel_audio_analysis(
        original_audio_dir, supported_extensions, num_workers if num_workers is not None else os.cpu_count() or 4,
        default_silence_thresh_dbfs, default_min_silence_len_ms
    )

    # --- FASE 2: Splitting/Copia Parallela ---
    print(f"\n--- Avvio Splitting/Copia Ibrida Parallela (Workers: {num_workers}) ---")
    split_manifest = { # Salva parametri e target LUFS
        "split_method": "hybrid_silence_dynamic_params_lufs_target",
        "effective_target_lufs_for_norm": effective_target_lufs, # <-- TARGET CALCOLATO
        # ... (salva altri parametri come prima) ...
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
            tasks_args = [] # Prepara args per worker splitting
            for original_path in original_files:
                params = custom_params_dict[original_path] # Recupera T, L, A, Lufs, SNR, etc.
                tasks_args.append(
                    (original_path, split_audio_dir,
                     params['threshold'], params['min_len'], params['avg_dbfs'], # Passa valori specifici
                     keep_silence_ms, min_duration_for_split_seconds, # Passa globali/config
                     target_chunk_duration_seconds, silence_search_range_ms) )
            split_results = []
            try: split_results = pool.starmap(_process_single_audio_file_hybrid_split, tasks_args); print(f"\nCompletati {len(split_results)} task splitting.")
            except Exception as pool_error: pool_error_occurred = True; print(f"!!! Errore Pool Splitting: {pool_error}")
            if not pool_error_occurred: # Processa risultati
                for result_tuple in split_results:
                    try:
                        file_manifest_entries, success_flag, chunks_exported, _ = result_tuple
                        if success_flag: total_files_processed_success += 1; split_manifest["files"].update(file_manifest_entries); total_chunks_exported += chunks_exported
                        else: total_errors += 1
                    except Exception as e: total_errors += 1; print(f"!!! Errore processando risultato split: {e}")
    finally: splitting_pool_global_ref = None

    if pool_error_occurred: print("Pipeline Splitting terminata con errori."); return None, {}
    # ... (Stampa riepilogo finale) ...
    print(f"\n--- Pipeline di Splitting (Analisi + Ibrido) Completata ---")
    print(f"File processati: {total_files_processed_success} (errori: {total_errors})")
    print(f"  - Totale chunk esportati: {total_chunks_exported}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")
    if not split_manifest["files"]: return None, {}
    # ... (Salvataggio manifest) ...
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path} (Target LUFS: {effective_target_lufs:.1f})")
        return manifest_path, split_manifest # Ritorna manifest aggiornato
    except Exception as e: print(f"!!! Errore salvataggio manifest: {e}"); return None, {}

# Riferimenti globali per signal handler
analysis_pool_global_ref = None
splitting_pool_global_ref = None

# --- END OF transcriptionUtils/splitAudio.py (CORRETTO - Analisi + Splitting Ibrido + Salva AvgDBFS) ---