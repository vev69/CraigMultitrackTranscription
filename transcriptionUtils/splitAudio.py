# --- START OF transcriptionUtils/splitAudio.py (MODIFICATO per Analisi Dinamica Soglie) ---

import os
import json
import math
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
# Importa funzioni silence necessarie
from pydub.silence import detect_nonsilent, detect_silence # type: ignore
import shutil
import numpy as np # Necessario per istogramma e percentili
import platform
import multiprocessing as mp

SPLIT_MANIFEST_FILENAME = "split_manifest.json"
# Costanti per l'analisi
ANALYSIS_CHUNK_MS = 50 # Analizza audio in chunk da 50ms

# --- NUOVO WORKER: Analisi Audio per Soglie Dinamiche ---
def _analyze_audio_for_thresholds(original_path: str,
                                  default_silence_thresh_dbfs: float,
                                  default_min_silence_len_ms: int
                                  ) -> tuple[str, float | None, int | None]:
    """
    Worker per analizzare un file audio e determinare soglie di silenzio dinamiche.

    Args:
        original_path: Percorso del file audio.
        default_silence_thresh_dbfs: Valore di fallback per la soglia dBFS.
        default_min_silence_len_ms: Valore di fallback per la lunghezza minima del silenzio.

    Returns:
        Una tupla: (original_path, calculated_threshold_dbfs, calculated_min_silence_len_ms).
        Restituisce None per i valori calcolati se l'analisi fallisce.
    """
    filename = os.path.basename(original_path)
    # print(f"  [AnalysisWorker {os.getpid()}] Analyzing: {filename}") # Verboso
    calculated_threshold = None
    calculated_min_len = None

    try:
        audio = AudioSegment.from_file(original_path)
        if len(audio) == 0:
             print(f"  [AnalysisWorker {os.getpid()}] WARN: Audio file is empty: {filename}"); return original_path, None, None

        # --- 1. Calcolo Soglia dBFS (Istogramma) ---
        # Calcola dBFS per piccoli chunk
        dbfs_values = []
        for i in range(0, len(audio), ANALYSIS_CHUNK_MS):
            chunk = audio[i:i+ANALYSIS_CHUNK_MS]
            if len(chunk) > 0: # Assicurati che il chunk non sia vuoto
                 # chunk.dBFS può essere -inf se il chunk è silenzio digitale
                 # Lo gestiamo dopo filtrando
                 dbfs_values.append(chunk.dBFS)

        # Filtra valori infiniti e NaN se presenti
        valid_dbfs = [db for db in dbfs_values if np.isfinite(db)]

        if not valid_dbfs:
            print(f"  [AnalysisWorker {os.getpid()}] WARN: No valid dBFS values found for {filename}. Using default threshold.")
            calculated_threshold = default_silence_thresh_dbfs # Usa default se non ci sono dati validi
        else:
            try:
                hist, bin_edges = np.histogram(valid_dbfs, bins='auto') # 'auto' sceglie un buon numero di bin
                bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

                # Stima euristica dei picchi (rumore e parlato)
                # Usiamo percentili come guida robusta
                noise_peak_estimate = np.percentile(valid_dbfs, 15) # Stima del livello di rumore
                speech_peak_estimate = np.percentile(valid_dbfs, 80) # Stima del livello di parlato

                # Trova l'indice del bin più vicino alle stime
                noise_bin_idx = np.argmin(np.abs(bin_centers - noise_peak_estimate))
                speech_bin_idx = np.argmin(np.abs(bin_centers - speech_peak_estimate))

                # Assicurati che gli indici siano ordinati e validi
                start_idx = min(noise_bin_idx, speech_bin_idx)
                end_idx = max(noise_bin_idx, speech_bin_idx)

                valley_threshold = default_silence_thresh_dbfs # Inizia con default
                if start_idx < end_idx: # Se i picchi stimati sono distinti
                    # Trova l'indice del minimo nell'istogramma TRA i picchi stimati
                    try:
                        valley_idx_rel = np.argmin(hist[start_idx:end_idx+1])
                        valley_idx_abs = start_idx + valley_idx_rel
                        # La soglia è il bordo destro del bin minimo (o il centro?)
                        # Usiamo il bordo destro per essere più conservativi (taglia meno)
                        potential_threshold = bin_edges[valley_idx_abs + 1]

                        # Applica dei limiti ragionevoli alla soglia calcolata
                        # Evita soglie troppo alte (es > -15dBFS) o troppo basse (es < -60dBFS)
                        valley_threshold = max(-60.0, min(-15.0, potential_threshold))
                        # print(f"  [AnalysisWorker {os.getpid()}] Histogram analysis for {filename}: NoiseEst={noise_peak_estimate:.1f}, SpeechEst={speech_peak_estimate:.1f}, ValleyFoundAt={potential_threshold:.1f}, AdjustedThreshold={valley_threshold:.1f}")

                    except (ValueError, IndexError):
                         print(f"  [AnalysisWorker {os.getpid()}] WARN: Could not find valley in histogram for {filename}. Using default threshold.")
                         valley_threshold = default_silence_thresh_dbfs
                else:
                    # Se i picchi coincidono o sono invertiti, l'istogramma è probabilmente unimodale
                    # Usa una soglia basata sulla percentile bassa + offset
                    print(f"  [AnalysisWorker {os.getpid()}] WARN: Histogram likely unimodal for {filename}. Using percentile-based threshold.")
                    valley_threshold = max(-60.0, min(-15.0, noise_peak_estimate + 10.0)) # Rumore + 10dB

                calculated_threshold = valley_threshold

            except Exception as hist_err:
                 print(f"  [AnalysisWorker {os.getpid()}] ERROR during histogram analysis for {filename}: {hist_err}. Using default threshold.")
                 calculated_threshold = default_silence_thresh_dbfs

        # --- 2. Calcolo Lunghezza Minima Silenzio (Basato sulla Soglia Calcolata) ---
        if calculated_threshold is not None:
            try:
                # Usa detect_silence con la soglia trovata
                # Usiamo una min_silence_len bassa qui (es. 100ms) per trovare *tutti* i silenzi
                # poi analizzeremo le loro durate
                silence_ranges = detect_silence(
                    audio,
                    min_silence_len=100, # Trova anche silenzi brevi
                    silence_thresh=calculated_threshold,
                    seek_step=1
                )

                silence_durations_ms = [(end - start) for start, end in silence_ranges]

                if silence_durations_ms:
                    # Filtra silenzi estremamente brevi (potrebbero essere rumore intra-parola)
                    meaningful_silences_ms = [d for d in silence_durations_ms if d >= 150] # Minimo 150ms

                    if meaningful_silences_ms:
                        # Calcola una percentile delle durate dei silenzi significativi
                        # La mediana (50°) o una percentile leggermente più alta (60°-70°)
                        # può essere un buon candidato per min_silence_len
                        potential_min_len = np.percentile(meaningful_silences_ms, 65) # Es. 65° percentile

                        # Applica limiti ragionevoli
                        calculated_min_len = int(max(250, min(2000, potential_min_len))) # Es. tra 250ms e 2000ms
                        # print(f"  [AnalysisWorker {os.getpid()}] Silence duration analysis for {filename}: Found {len(meaningful_silences_ms)} silences >= 150ms. PotentialMinLen={potential_min_len:.0f}, AdjustedMinLen={calculated_min_len}")
                    else:
                         print(f"  [AnalysisWorker {os.getpid()}] WARN: No meaningful silences (>=150ms) found for {filename} at threshold {calculated_threshold:.1f}dBFS. Using default min_len.")
                         calculated_min_len = default_min_silence_len_ms
                else:
                    print(f"  [AnalysisWorker {os.getpid()}] WARN: No silences detected for {filename} at threshold {calculated_threshold:.1f}dBFS. Using default min_len.")
                    calculated_min_len = default_min_silence_len_ms

            except Exception as silence_err:
                 print(f"  [AnalysisWorker {os.getpid()}] ERROR during silence duration analysis for {filename}: {silence_err}. Using default min_len.")
                 calculated_min_len = default_min_silence_len_ms
        else:
            # Se il calcolo della soglia è fallito, usa anche min_len di default
            calculated_min_len = default_min_silence_len_ms

    except pydub_exceptions.CouldntDecodeError as e:
        print(f"  !!! [AnalysisWorker {os.getpid()}] ERROR decoding {filename}: {e}. Cannot analyze.")
    except FileNotFoundError:
         print(f"  !!! [AnalysisWorker {os.getpid()}] ERROR File not found: {original_path}. Cannot analyze.")
    except Exception as e:
        print(f"  !!! [AnalysisWorker {os.getpid()}] UNEXPECTED ERROR analyzing {filename}: {e}")
        import traceback
        traceback.print_exc()

    # Ritorna il path originale e le soglie calcolate (o None se fallite)
    return original_path, calculated_threshold, calculated_min_len


# --- NUOVA FUNZIONE: Orchestra l'Analisi Parallela ---
def _run_parallel_audio_analysis(original_audio_dir: str,
                                 supported_extensions: tuple,
                                 num_workers: int,
                                 default_silence_thresh_dbfs: float,
                                 default_min_silence_len_ms: int
                                 ) -> dict[str, dict]:
    """
    Esegue l'analisi preliminare dei file audio in parallelo per determinare
    le soglie di silenzio dinamiche.

    Returns:
        Un dizionario: {original_path: {'threshold': float, 'min_len': int}}
        dove i valori sono quelli calcolati o i default se l'analisi fallisce.
    """
    print(f"\n--- Avvio Analisi Audio Parallela per Soglie Dinamiche (Workers: {num_workers}) ---")
    custom_params_dict: dict[str, dict] = {}

    original_files = sorted([
        f for f in os.listdir(original_audio_dir)
        if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)
    ])

    if not original_files:
        print("Nessun file trovato per l'analisi.")
        return custom_params_dict

    # Setup Pool
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)

    with context.Pool(processes=num_workers) as pool:
        tasks_args = [
            (os.path.abspath(os.path.join(original_audio_dir, filename)),
             default_silence_thresh_dbfs,
             default_min_silence_len_ms)
            for filename in original_files
        ]
        analysis_results = []
        try:
            # starmap blocca fino al completamento
            analysis_results = pool.starmap(_analyze_audio_for_thresholds, tasks_args)
            print(f"Completata analisi preliminare per {len(analysis_results)} file.")
        except Exception as pool_error:
             print(f"!!! Errore durante l'esecuzione del Pool di Analisi: {pool_error}")
             # In caso di errore grave nel pool, ritorna dizionario vuoto
             # Lo splitting userà i default globali
             return custom_params_dict

        # Costruisci il dizionario dei risultati
        for original_path, calc_thresh, calc_min_len in analysis_results:
            custom_params_dict[original_path] = {
                'threshold': calc_thresh if calc_thresh is not None else default_silence_thresh_dbfs,
                'min_len': calc_min_len if calc_min_len is not None else default_min_silence_len_ms
            }
            # Log opzionale dei parametri calcolati/usati
            # print(f"  Params for {os.path.basename(original_path)}: Threshold={custom_params_dict[original_path]['threshold']:.1f} dBFS, MinLen={custom_params_dict[original_path]['min_len']} ms")


    print("--- Fine Analisi Audio Parallela ---")
    return custom_params_dict


# --- WORKER per lo SPLITTING (MODIFICATO per accettare parametri dinamici) ---
def _process_single_audio_file_for_split(original_path: str,
                                         split_audio_dir: str,
                                         # Parametri specifici per questo file
                                         silence_thresh_dbfs: float,
                                         min_silence_len_ms: int,
                                         # Parametri globali/configurabili
                                         keep_silence_ms: int,
                                         split_naming_threshold_seconds: float
                                         ) -> tuple[dict, int, int, int]:
    """
    Worker per dividere/copiare usando soglie specifiche per il file.
    """
    filename = os.path.basename(original_path)
    # Usa i parametri specifici ricevuti
    # print(f"  [SplitWorker {os.getpid()}] Processing: {filename} with Threshold={silence_thresh_dbfs:.1f}, MinLen={min_silence_len_ms}") # Verboso
    file_manifest_entries = {}
    chunks_exported_count = 0
    success_flag = 0

    try:
        base_name, ext = os.path.splitext(filename)
        audio = AudioSegment.from_file(original_path)
        duration_seconds = len(audio) / 1000.0
        if duration_seconds == 0: raise ValueError("Audio file is empty.")

        # --- Logica Splitting con detect_nonsilent (usa parametri specifici) ---
        nonsilent_ranges = detect_nonsilent(
            audio,
            # Usa min_silence_len_ms specifico per questo file (aumentato un po' per robustezza taglio)
            min_silence_len=min_silence_len_ms + int(keep_silence_ms * 0.5),
            # Usa silence_thresh_dbfs specifico per questo file
            silence_thresh=silence_thresh_dbfs,
            seek_step=1
        )

        if not nonsilent_ranges:
             print(f"  [SplitWorker {os.getpid()}] WARN: detect_nonsilent non ha trovato segmenti per {filename} (Thresh={silence_thresh_dbfs:.1f}, MinLen={min_silence_len_ms}). Skipping.")
             return {}, 0, 0, 0

        # print(f"  [SplitWorker {os.getpid()}] Found {len(nonsilent_ranges)} non-silent ranges for {filename}.") # Meno verboso
        chunks_exported_count = 0

        is_considered_long = duration_seconds > split_naming_threshold_seconds

        for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
             padded_start_ms = max(0, start_ms - keep_silence_ms)
             padded_end_ms = min(len(audio), end_ms + keep_silence_ms)
             if padded_start_ms >= padded_end_ms: continue

             chunk = audio[padded_start_ms:padded_end_ms]

             if len(nonsilent_ranges) == 1 and not is_considered_long:
                  chunk_filename = f"{base_name}{ext}"
                  is_chunk_flag = False
                  chunk_index = 0
             else:
                  chunk_filename = f"{base_name}_part{i:03d}{ext}"
                  is_chunk_flag = True
                  chunk_index = i

             chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
             start_time_abs = padded_start_ms / 1000.0
             end_time_abs = padded_end_ms / 1000.0

             if not os.path.exists(chunk_output_path):
                 try:
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
    except ValueError as ve: print(f"  !!! [SplitWorker {os.getpid()}] ERROR processing {filename}: {ve}. Skipping.") # Es. file vuoto
    except Exception as e:
        print(f"  !!! [SplitWorker {os.getpid()}] UNEXPECTED ERROR processing {filename}: {e}")
        import traceback; traceback.print_exc()

    return file_manifest_entries, success_flag, chunks_exported_count, 0


# --- Funzione Principale (MODIFICATA per integrare analisi) ---
def split_large_audio_files(original_audio_dir: str,
                            split_audio_dir: str,
                            split_naming_threshold_seconds: float = 45 * 60,
                            # Valori di DEFAULT per analisi (usati anche come fallback)
                            default_min_silence_len_ms: int = 700,
                            default_silence_thresh_dbfs: float = -35.0,
                            # Parametro globale per padding
                            keep_silence_ms: int = 150,
                            supported_extensions=(".flac",".m4a"),
                            num_workers: int | None = None
                            ) -> tuple[str | None, dict]:
    """
    Esegue analisi dinamica soglie, poi divide/copia file audio in parallelo
    e crea un manifest JSON.
    """
    # ... (Stampa parametri iniziali, setup num_workers, check dir original_audio_dir) ...
    # ... (Stampa parametri iniziali invariate) ...
    print(f"\n--- Avvio Pipeline di Splitting Parallelo ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Soglia durata per nome '_partXXX': {split_naming_threshold_seconds / 60:.1f} minuti")
    print(f"Parametri Silenzio di Default: min_len={default_min_silence_len_ms}ms, threshold={default_silence_thresh_dbfs}dBFS")
    print(f"Padding Silenzio: keep_silence={keep_silence_ms}ms")
    print(f"Estensioni supportate: {supported_extensions}")

    if num_workers is None:
        try: num_workers = os.cpu_count(); num_workers = 4 if num_workers is None else num_workers
        except NotImplementedError: num_workers = 4
        print(f"Numero workers non specificato, uso default: {num_workers}")
    else: print(f"Numero workers specificato: {num_workers}")
    if not os.path.isdir(original_audio_dir): return None, {}
    try: os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e: print(f"Errore creazione directory split: {e}"); return None, {}

    # --- FASE 1: Analisi Parallela per Soglie Dinamiche ---
    custom_params_dict = _run_parallel_audio_analysis(
        original_audio_dir,
        supported_extensions,
        num_workers,
        default_silence_thresh_dbfs,
        default_min_silence_len_ms
    )

    # --- FASE 2: Splitting/Copia Parallela usando Soglie Calcolate ---
    print(f"\n--- Avvio Splitting/Copia Parallela (Workers: {num_workers}) ---")
    split_manifest = {
        "split_method": "silence_detection_dynamic_threshold",
        "default_min_silence_len_ms": default_min_silence_len_ms, # Salva i default usati
        "default_silence_thresh_dbfs": default_silence_thresh_dbfs,
        "keep_silence_ms": keep_silence_ms,
        "split_naming_threshold_seconds": split_naming_threshold_seconds,
        "files": {} # Verrà popolato
    }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    total_files_processed_success = 0
    total_chunks_exported = 0
    total_errors = 0

    original_files = sorted(custom_params_dict.keys()) # Usa le chiavi del dizionario params
    if not original_files:
         print("Nessun file analizzato con successo nella fase precedente.")
         return None, {}

    # Setup Pool per lo splitting
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    with context.Pool(processes=num_workers) as pool:
        # Prepara argomenti includendo le soglie specifiche per file
        tasks_args = []
        for original_path in original_files:
            params = custom_params_dict[original_path] # Recupera soglie calcolate
            tasks_args.append(
                (original_path,
                 split_audio_dir,
                 params['threshold'], # Passa soglia specifica
                 params['min_len'],   # Passa min_len specifico
                 keep_silence_ms,     # Passa parametro globale
                 split_naming_threshold_seconds # Passa parametro globale
                 )
            )

        split_results = []
        try:
            split_results = pool.starmap(_process_single_audio_file_for_split, tasks_args)
            print(f"\nCompletati {len(split_results)} task di splitting/copy dalla pool.")
        except Exception as pool_error:
             print(f"!!! Errore durante l'esecuzione del Pool di Splitting: {pool_error}")
             return None, {}

        # Processa i risultati dello splitting
        for result_tuple in split_results:
            try:
                file_manifest_entries, success_flag, chunks_exported, _ = result_tuple
                if success_flag:
                    total_files_processed_success += 1
                    split_manifest["files"].update(file_manifest_entries)
                    total_chunks_exported += chunks_exported
                else:
                    total_errors += 1
            except Exception as e:
                total_errors += 1
                print(f"!!! Errore processando risultato task splitting: {e}")

    # --- Riepilogo Finale (invariato) ---
    print(f"\n--- Pipeline di Splitting Completata ---")
    print(f"File originali processati con successo (analisi + split/copy): {total_files_processed_success}")
    print(f"  - Totale chunk esportati: {total_chunks_exported}")
    print(f"Errori riscontrati (analisi o split/copy falliti): {total_errors}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")

    if not split_manifest["files"]: return None, {}

    # Salva il manifest JSON finale
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e:
        print(f"!!! Errore critico durante salvataggio manifest: {e}")
        return None, {}

# --- END OF transcriptionUtils/splitAudio.py (MODIFICATO per Analisi Dinamica Soglie) ---