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
# import numpy as np
import platform
import multiprocessing as mp

SPLIT_MANIFEST_FILENAME = "split_manifest.json"

# --- PARAMETRI CONFIGURABILI PER TAGLIO IBRIDO ---
# Durata target approssimativa per ogni chunk
TARGET_CHUNK_DURATION_SECONDS: float = 15 * 60  # 15 minuti
# Soglia minima per considerare un file "lungo" e necessitare di splitting
MIN_DURATION_FOR_SPLIT_SECONDS: float = TARGET_CHUNK_DURATION_SECONDS + (1 * 60) # Es: splitta solo se > 16 min
# Parametri per trovare il punto di taglio silenzioso
SILENCE_SEARCH_RANGE_MS: int = 10 * 1000 # Cerca silenzio in +/- 10 secondi attorno al taglio ideale
MIN_SILENCE_LEN_FOR_CUT_MS: int = 700     # Silenzio deve essere lungo almeno 700ms per tagliare
SILENCE_THRESH_DBFS_FOR_CUT: float = -40.0 # Soglia dBFS per considerare silenzio (può essere fissa qui)
# Padding da mantenere attorno ai chunk tagliati
KEEP_SILENCE_MS: int = 250


# --- WORKER per Splitting Ibrido ---
def _process_single_audio_file_hybrid_split(original_path: str,
                                            split_audio_dir: str
                                            ) -> tuple[dict, int, int, int]:
    """
    Worker per splittare un file audio usando l'approccio ibrido:
    mira a una durata target ma taglia su un silenzio vicino.
    Copia i file più corti della soglia minima.

    Args:
        original_path: Percorso assoluto del file audio originale.
        split_audio_dir: Directory dove salvare i chunk/file esportati.

    Returns:
        Una tupla:
        - dict: Voci del manifest generate per questo file (path_output -> metadata).
        - int: Flag di successo (1 se successo, 0 se errore).
        - int: Numero di chunk esportati (>=1 se successo).
        - int: 0 (non usato).
    """
    filename = os.path.basename(original_path)
    print(f"  [SplitWorkerHybrid {os.getpid()}] Processing: {filename}")
    file_manifest_entries = {}
    chunks_exported_count = 0
    success_flag = 0

    try:
        base_name, ext = os.path.splitext(filename)
        audio = AudioSegment.from_file(original_path)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0
        print(f"  [SplitWorkerHybrid {os.getpid()}] Duration: {duration_seconds:.2f}s")

        if duration_seconds <= MIN_DURATION_FOR_SPLIT_SECONDS:
            # File corto: copia/esporta come singolo file
            print(f"  [SplitWorkerHybrid {os.getpid()}] File is short. Exporting as single chunk.")
            output_filename = f"{base_name}{ext}" # Nome originale
            output_path = os.path.abspath(os.path.join(split_audio_dir, output_filename))

            if not os.path.exists(output_path):
                 try: audio.export(output_path, format=ext.lstrip('.'))
                 except Exception as export_err: print(f"  !!! ERROR exporting short file {output_filename}: {export_err}"); raise # Rilancia errore per bloccare questo file

            file_manifest_entries[output_path] = {
                "original_file": original_path, "is_chunk": False, "chunk_index": 0,
                "start_time_abs": 0.0, "end_time_abs": duration_seconds,
                "original_duration_seconds": duration_seconds
            }
            chunks_exported_count = 1
            success_flag = 1

        else:
            # File lungo: splitta in modo ibrido
            target_chunk_duration_ms = int(TARGET_CHUNK_DURATION_SECONDS * 1000)
            num_ideal_chunks = math.ceil(duration_ms / target_chunk_duration_ms)
            print(f"  [SplitWorkerHybrid {os.getpid()}] File is long. Aiming for ~{num_ideal_chunks} chunks of ~{TARGET_CHUNK_DURATION_SECONDS/60:.1f} min.")

            cut_points_ms = [0] # Inizia sempre da 0
            last_cut_ms = 0

            for i in range(1, num_ideal_chunks):
                ideal_cut_point_ms = i * target_chunk_duration_ms

                # Definisci l'intervallo di ricerca attorno al punto di taglio ideale
                search_start_ms = max(last_cut_ms + MIN_SILENCE_LEN_FOR_CUT_MS, # Non cercare prima dell'ultimo taglio
                                     ideal_cut_point_ms - SILENCE_SEARCH_RANGE_MS)
                search_end_ms = min(duration_ms - MIN_SILENCE_LEN_FOR_CUT_MS, # Non cercare troppo vicino alla fine
                                    ideal_cut_point_ms + SILENCE_SEARCH_RANGE_MS)

                if search_start_ms >= search_end_ms:
                    # Intervallo di ricerca non valido, usa il taglio ideale (meno ottimale)
                    actual_cut_point_ms = ideal_cut_point_ms
                    print(f"    WARN: Invalid search range around {ideal_cut_point_ms/1000:.1f}s. Using ideal cut.")
                else:
                    # Cerca silenzi nell'intervallo definito
                    audio_slice_for_search = audio[search_start_ms:search_end_ms]
                    silences = detect_silence(
                        audio_slice_for_search,
                        min_silence_len=MIN_SILENCE_LEN_FOR_CUT_MS,
                        silence_thresh=SILENCE_THRESH_DBFS_FOR_CUT,
                        seek_step=1
                    )

                    if silences:
                        # Trovato almeno un silenzio valido! Scegli il migliore.
                        # Opzione 1: Scegli il centro del silenzio più lungo
                        # Opzione 2: Scegli il centro del silenzio più vicino al taglio ideale
                        best_silence_center_ms = -1
                        min_dist_to_ideal = float('inf')

                        for silence_start_rel, silence_end_rel in silences:
                             silence_center_rel = silence_start_rel + (silence_end_rel - silence_start_rel) / 2
                             # Converte il centro relativo in assoluto (rispetto all'inizio dell'audio)
                             silence_center_abs = search_start_ms + silence_center_rel
                             dist = abs(silence_center_abs - ideal_cut_point_ms)
                             if dist < min_dist_to_ideal:
                                 min_dist_to_ideal = dist
                                 best_silence_center_ms = int(silence_center_abs)

                        actual_cut_point_ms = best_silence_center_ms
                        print(f"    Found silence near {ideal_cut_point_ms/1000:.1f}s. Cutting at {actual_cut_point_ms/1000:.1f}s.")
                    else:
                        # Nessun silenzio valido trovato nell'intervallo, usa il taglio ideale
                        actual_cut_point_ms = ideal_cut_point_ms
                        print(f"    No suitable silence found near {ideal_cut_point_ms/1000:.1f}s. Using ideal cut.")

                # Assicurati che il punto di taglio non sia troppo vicino al precedente
                if actual_cut_point_ms <= last_cut_ms + 1000: # Minimo 1 secondo tra tagli?
                    print(f"    WARN: Calculated cut point {actual_cut_point_ms/1000:.1f}s too close to previous {last_cut_ms/1000:.1f}s. Adjusting.")
                    actual_cut_point_ms = last_cut_ms + target_chunk_duration_ms # Forza avanzamento
                    # Assicura non superi la durata totale
                    actual_cut_point_ms = min(duration_ms - 1000, actual_cut_point_ms) # Lascia almeno 1s alla fine


                # Aggiungi punto di taglio e aggiorna l'ultimo taglio
                # Arrotonda all'intero per lo slicing
                actual_cut_point_ms = int(round(actual_cut_point_ms))
                if actual_cut_point_ms > last_cut_ms: # Assicura avanzamento
                     cut_points_ms.append(actual_cut_point_ms)
                     last_cut_ms = actual_cut_point_ms
                else:
                    print(f"    ERROR: Failed to advance cut point for chunk {i+1}. Stopping split for this file.")
                    break # Esce dal loop for i

            # Aggiungi il punto finale (fine dell'audio)
            cut_points_ms.append(duration_ms)

            print(f"  [SplitWorkerHybrid {os.getpid()}] Final cut points (ms): {cut_points_ms}")

            # --- Esporta i chunk basati sui punti di taglio calcolati ---
            chunks_exported_count = 0
            for i in range(len(cut_points_ms) - 1):
                 start_ms = cut_points_ms[i]
                 end_ms = cut_points_ms[i+1]

                 # Applica padding (ma senza sovrapporre i chunk!)
                 padded_start_ms = max(0, start_ms - KEEP_SILENCE_MS)
                 padded_end_ms = min(duration_ms, end_ms + KEEP_SILENCE_MS)

                 # Assicura che i chunk con padding non si sovrappongano troppo
                 # Se non è il primo chunk, assicurati che il suo inizio paddato
                 # non sia *prima* della *fine non paddata* del chunk precedente.
                 if i > 0:
                      previous_end_ms = cut_points_ms[i]
                      padded_start_ms = max(padded_start_ms, previous_end_ms)

                 # Se non è l'ultimo chunk, assicurati che la sua fine paddata
                 # non sia *dopo* l'*inizio non paddato* del chunk successivo.
                 if i < len(cut_points_ms) - 2:
                     next_start_ms = cut_points_ms[i+1]
                     padded_end_ms = min(padded_end_ms, next_start_ms)


                 if padded_start_ms >= padded_end_ms: continue # Salta chunk non valido

                 chunk = audio[padded_start_ms:padded_end_ms]
                 chunk_filename = f"{base_name}_part{i:03d}{ext}"
                 chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
                 start_time_abs = padded_start_ms / 1000.0
                 end_time_abs = padded_end_ms / 1000.0

                 if not os.path.exists(chunk_output_path):
                     try:
                         # print(f"    Exporting: {chunk_filename} [{start_time_abs:.2f}s - {end_time_abs:.2f}s]") # Verboso
                         chunk.export(chunk_output_path, format=ext.lstrip('.'))
                     except Exception as export_err:
                          print(f"  !!! ERROR exporting {chunk_filename}: {export_err}. Skipping chunk.")
                          continue

                 file_manifest_entries[chunk_output_path] = {
                     "original_file": original_path, "is_chunk": True, "chunk_index": i,
                     "start_time_abs": start_time_abs, "end_time_abs": end_time_abs,
                     "original_duration_seconds": duration_seconds
                 }
                 chunks_exported_count += 1

            if chunks_exported_count > 0: success_flag = 1

    # --- Gestione Errori (invariata) ---
    except pydub_exceptions.CouldntDecodeError as e: print(f"  !!! ERROR decoding {filename}: {e}. Skipping.")
    except FileNotFoundError: print(f"  !!! ERROR File not found: {original_path}. Skipping.")
    except ValueError as ve: print(f"  !!! ERROR processing {filename}: {ve}. Skipping.")
    except Exception as e:
        print(f"  !!! UNEXPECTED ERROR processing {filename}: {e}")
        import traceback; traceback.print_exc()

    return file_manifest_entries, success_flag, chunks_exported_count, 0


# --- Funzione Principale (MODIFICATA per usare worker ibrido) ---
def split_large_audio_files(original_audio_dir: str,
                            split_audio_dir: str,
                            # Usa i default definiti all'inizio
                            target_chunk_duration_seconds: float = TARGET_CHUNK_DURATION_SECONDS,
                            min_duration_for_split_seconds: float = MIN_DURATION_FOR_SPLIT_SECONDS,
                            silence_search_range_ms: int = SILENCE_SEARCH_RANGE_MS,
                            min_silence_len_for_cut_ms: int = MIN_SILENCE_LEN_FOR_CUT_MS,
                            silence_thresh_dbfs_for_cut: float = SILENCE_THRESH_DBFS_FOR_CUT,
                            keep_silence_ms: int = KEEP_SILENCE_MS,
                            supported_extensions=(".flac",".m4a"),
                            num_workers: int | None = None
                            ) -> tuple[str | None, dict]:
    """
    Esegue splitting ibrido (tempo target + taglio su silenzio vicino) in parallelo
    e crea un manifest JSON.
    """
    # Stampa i parametri EFFETTIVI che verranno usati
    print(f"\n--- Avvio Pipeline di Splitting Parallelo (Ibrido Tempo/Silenzio) ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Durata Target Chunk: ~{target_chunk_duration_seconds / 60:.1f} minuti")
    print(f"Split solo se originale > {min_duration_for_split_seconds / 60:.1f} minuti")
    print(f"Ricerca Silenzio per Taglio: Range=+/-{silence_search_range_ms/1000}s, MinLen={min_silence_len_for_cut_ms}ms, Thresh={silence_thresh_dbfs_for_cut}dBFS")
    print(f"Padding Silenzio: keep_silence={keep_silence_ms}ms")
    print(f"Estensioni supportate: {supported_extensions}")

    # Logica num_workers, check dir, creazione dir principale (invariata)
    if num_workers is None:
        try: num_workers = os.cpu_count(); num_workers = 4 if num_workers is None else num_workers
        except NotImplementedError: num_workers = 4
        print(f"Numero workers: {num_workers}")
    if not os.path.isdir(original_audio_dir): return None, {}
    try: os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e: print(f"Errore creazione directory split: {e}"); return None, {}

    # NON serve più l'analisi preliminare
    # custom_params_dict = _run_parallel_audio_analysis(...)

    # --- FASE UNICA: Splitting/Copia Ibrida Parallela ---
    print(f"\n--- Avvio Splitting/Copia Ibrida Parallela (Workers: {num_workers}) ---")
    split_manifest = {
        "split_method": "hybrid_time_silence", # Nuovo metodo
        "target_chunk_duration_seconds": target_chunk_duration_seconds,
        "min_duration_for_split_seconds": min_duration_for_split_seconds,
        "silence_search_range_ms": silence_search_range_ms,
        "min_silence_len_for_cut_ms": min_silence_len_for_cut_ms,
        "silence_thresh_dbfs_for_cut": silence_thresh_dbfs_for_cut,
        "keep_silence_ms": keep_silence_ms,
        "files": {}
    }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    total_files_processed_success = 0
    total_chunks_exported = 0
    total_errors = 0

    original_files = sorted([
        f for f in os.listdir(original_audio_dir)
        if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)
    ])
    if not original_files: return None, {}

    # Setup Pool
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)
    with context.Pool(processes=num_workers) as pool:
        # Prepara argomenti per il worker ibrido
        tasks_args = [
            (os.path.abspath(os.path.join(original_audio_dir, filename)),
             split_audio_dir)
            for filename in original_files
        ]

        split_results = []
        try:
            # Usa starmap (passa tuple di argomenti)
            split_results = pool.starmap(_process_single_audio_file_hybrid_split, tasks_args)
            print(f"\nCompletati {len(split_results)} task di splitting/copy dalla pool.")
        except Exception as pool_error:
             print(f"!!! Errore durante l'esecuzione del Pool di Splitting: {pool_error}")
             return None, {}

        # Processa i risultati
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

    # --- Riepilogo Finale ---
    print(f"\n--- Pipeline di Splitting Ibrido Completata ---")
    print(f"File originali processati: {total_files_processed_success} (con errori: {total_errors})")
    print(f"  - Totale chunk finali esportati: {total_chunks_exported}") # Questo numero dovrebbe essere molto più basso ora
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")

    if not split_manifest["files"]: return None, {}

    # Salva manifest
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e: print(f"!!! Errore salvataggio manifest: {e}"); return None, {}


# --- END OF transcriptionUtils/splitAudio.py (MODIFICATO per Taglio Ibrido Tempo/Silenzio) ---