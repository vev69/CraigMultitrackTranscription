# --- START OF transcriptionUtils/splitAudio.py (CON num_workers) ---

import os
import json
import math
import time
import concurrent.futures # Aggiunto
from concurrent.futures import ProcessPoolExecutor # Aggiunto
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
import shutil # Spostato import qui

SPLIT_MANIFEST_FILENAME = "split_manifest.json"

# --- WORKER per processare un singolo file originale ---
def _process_single_audio_file_for_split(original_path: str,
                                         split_audio_dir: str,
                                         split_threshold_seconds: float,
                                         target_chunk_duration_seconds: float
                                         ) -> tuple[dict, int, int, int]:
    """
    Worker eseguito in un processo separato per analizzare, dividere o copiare
    un singolo file audio originale e generare le sue voci per il manifest.

    Args:
        original_path: Percorso assoluto del file audio originale.
        split_audio_dir: Directory dove salvare i chunk/file copiati.
        split_threshold_seconds: Soglia per decidere se splittare.
        target_chunk_duration_seconds: Durata target dei chunk.

    Returns:
        Una tupla:
        - dict: Voci del manifest generate per questo file (path_output -> metadata).
        - int: Flag di successo (1 se successo, 0 se errore).
        - int: Numero di chunk creati (0 se copiato o errore).
        - int: Numero di file copiati (1 se copiato, 0 se splittato o errore).
    """
    filename = os.path.basename(original_path)
    print(f"  [SplitWorker {os.getpid()}] Processing: {filename}")
    file_manifest_entries = {}
    chunks_created = 0
    files_copied = 0
    success_flag = 0 # Inizia come fallimento

    try:
        base_name, ext = os.path.splitext(filename)
        # Carica l'audio per ottenere la durata
        # Specifica il formato esplicitamente se possibile, anche se pydub di solito lo deduce
        audio = AudioSegment.from_file(original_path) # Rimuovi format=... pydub è bravo a dedurlo
        duration_seconds = len(audio) / 1000.0
        # print(f"  [SplitWorker {os.getpid()}] Duration for {filename}: {duration_seconds:.2f}s") # Meno verboso

        if duration_seconds > split_threshold_seconds:
            # File lungo -> Dividi in chunk
            num_chunks = math.ceil(duration_seconds / target_chunk_duration_seconds)
            actual_chunk_len_ms = math.ceil(len(audio) / num_chunks)
            print(f"  [SplitWorker {os.getpid()}] Splitting {filename} into {num_chunks} chunks...")

            for i in range(num_chunks):
                start_ms = i * actual_chunk_len_ms
                end_ms = min((i + 1) * actual_chunk_len_ms, len(audio))
                # Assicurati che start_ms sia minore di end_ms (importante per slice pydub)
                if start_ms >= end_ms: continue # Salta chunk di lunghezza zero o negativa

                chunk = audio[start_ms:end_ms]
                chunk_filename = f"{base_name}_part{i:03d}{ext}" # Usa estensione originale
                chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))
                start_time_abs = start_ms / 1000.0
                end_time_abs = end_ms / 1000.0

                if os.path.exists(chunk_output_path):
                    # print(f"    [SplitWorker {os.getpid()}] Skipping existing chunk: {chunk_filename}") # Meno verboso
                    pass
                else:
                    # print(f"    [SplitWorker {os.getpid()}] Creating chunk {i+1}/{num_chunks}: {chunk_filename}") # Meno verboso
                    # Esporta nel formato originale (usa ext)
                    chunk.export(chunk_output_path, format=ext.lstrip('.'))

                # Aggiungi metadati al dizionario locale del worker
                file_manifest_entries[chunk_output_path] = {
                    "original_file": original_path, "is_chunk": True, "chunk_index": i,
                    "start_time_abs": start_time_abs, "end_time_abs": end_time_abs,
                    "original_duration_seconds": duration_seconds
                }
                chunks_created += 1
            success_flag = 1 # Successo se lo split è completato (o non ha generato errori)

        else:
            # File corto -> Copia direttamente
            # print(f"  [SplitWorker {os.getpid()}] Copying {filename}...") # Meno verboso
            output_path = os.path.abspath(os.path.join(split_audio_dir, filename))

            if os.path.exists(output_path):
                # print(f"    [SplitWorker {os.getpid()}] Skipping existing file: {filename}") # Meno verboso
                pass
            else:
                shutil.copy2(original_path, output_path)
                # print(f"    [SplitWorker {os.getpid()}] File copied: {output_path}") # Meno verboso

            # Aggiungi metadati al dizionario locale del worker
            file_manifest_entries[output_path] = {
                "original_file": original_path, "is_chunk": False, "chunk_index": 0,
                "start_time_abs": 0.0, "end_time_abs": duration_seconds,
                "original_duration_seconds": duration_seconds
            }
            files_copied = 1
            success_flag = 1 # Successo se la copia è completata (o non ha generato errori)

    except pydub_exceptions.CouldntDecodeError as e:
        print(f"  !!! [SplitWorker {os.getpid()}] ERROR decoding {filename}: {e}. Assicurati che ffmpeg sia installato e nel PATH. Skipping.")
        # success_flag rimane 0
    except FileNotFoundError:
         print(f"  !!! [SplitWorker {os.getpid()}] ERROR File not found: {original_path}. Skipping.")
         # success_flag rimane 0
    except Exception as e:
        print(f"  !!! [SplitWorker {os.getpid()}] UNEXPECTED ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc() # Stampa più dettagli per errori inattesi
        # success_flag rimane 0

    return file_manifest_entries, success_flag, chunks_created, files_copied


# --- Funzione Principale MODIFICATA per usare il Pool Executor ---
def split_large_audio_files(original_audio_dir: str,
                            split_audio_dir: str,
                            split_threshold_seconds: float = 45 * 60,
                            target_chunk_duration_seconds: float = 10 * 60,
                            supported_extensions=(".flac",".m4a"), # Assicurati includa m4a se necessario
                            num_workers: int | None = None # PARAMETRO DEFINITO QUI
                            ) -> tuple[str | None, dict]:
    """
    Scansiona, divide o copia file audio in parallelo e crea un manifest JSON.
    """
    print(f"\n--- Avvio Divisione File Audio Parallela ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Soglia per divisione: {split_threshold_seconds / 60:.1f} minuti")
    print(f"Durata target chunk: {target_chunk_duration_seconds / 60:.1f} minuti")
    print(f"Estensioni supportate: {supported_extensions}")

    if num_workers is None:
        try:
            num_workers = os.cpu_count()
            if num_workers is None: num_workers = 4 # Fallback se cpu_count() ritorna None
        except NotImplementedError:
            num_workers = 4 # Fallback se cpu_count() non è implementato
        print(f"Numero workers non specificato, uso default: {num_workers}")
    else:
        print(f"Numero workers specificato: {num_workers}")


    if not os.path.isdir(original_audio_dir):
        print(f"Errore: Directory originale non trovata: {original_audio_dir}")
        return None, {}

    try:
        # Crea la directory di output principale qui, prima che i worker partano
        os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e:
        print(f"Errore creazione directory split principale: {e}")
        return None, {}

    split_manifest = {
        "split_threshold_seconds": split_threshold_seconds,
        "target_chunk_duration_seconds": target_chunk_duration_seconds,
        "files": {} # Verrà popolato dai risultati dei worker
    }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    total_files_processed_success = 0
    total_chunks_created = 0
    total_files_copied = 0
    total_errors = 0

    original_files = sorted([
        f for f in os.listdir(original_audio_dir)
        if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)
    ])

    if not original_files:
        print("Nessun file audio con estensioni supportate trovato nella directory originale.")
        return None, {}

    print(f"Trovati {len(original_files)} file audio originali da processare in parallelo...")

    # Usa ProcessPoolExecutor per parallelizzare
    # Utilizza 'spawn' come start method se non su Linux per evitare problemi
    # Questo andrebbe fatto all'inizio dello script principale, ma lo aggiungiamo qui per sicurezza
    # anche se potrebbe dare un warning se già impostato.
    import platform
    import multiprocessing as mp
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)

    with context.Pool(processes=num_workers) as pool: # Usa context.Pool
        # Prepara gli argomenti per ogni chiamata a _process_single_audio_file_for_split
        tasks_args = [
            (os.path.abspath(os.path.join(original_audio_dir, filename)),
             split_audio_dir,
             split_threshold_seconds,
             target_chunk_duration_seconds)
            for filename in original_files
        ]

        results = []
        try:
            # Usa pool.starmap per passare gli argomenti multipli al worker
            # pool.starmap blocca finché tutti i risultati non sono pronti
            results = pool.starmap(_process_single_audio_file_for_split, tasks_args)
            print(f"Completati {len(results)} task di analisi/split/copy dalla pool.")
        except Exception as pool_error:
             print(f"!!! Errore durante l'esecuzione del Pool: {pool_error}")
             # Potrebbe essere necessario gestire meglio questo caso, magari tentando un approccio sequenziale
             return None, {}


        # Processa i risultati raccolti da starmap
        processed_count = 0
        for result_tuple in results:
            processed_count += 1
            try:
                file_manifest_entries, success_flag, chunks_created, files_copied = result_tuple

                if success_flag:
                    total_files_processed_success += 1
                    split_manifest["files"].update(file_manifest_entries)
                    total_chunks_created += chunks_created
                    total_files_copied += files_copied
                else:
                    total_errors += 1
                    # Il worker dovrebbe aver già stampato l'errore specifico
                    # print(f"  ({processed_count}/{len(results)}) Failed processing original file index {processed_count-1}")

            except Exception as e:
                # Errore nell'elaborare la tupla del risultato (improbabile ma possibile)
                total_errors += 1
                print(f"!!! Errore processando risultato task {processed_count}: {e}")


    # --- Riepilogo Finale ---
    print(f"\n--- Divisione/Copia Parallela Completata ---")
    print(f"File originali esaminati: {len(original_files)}")
    print(f"File processati con successo (divisi o copiati): {total_files_processed_success}")
    print(f"  - Totale chunk creati: {total_chunks_created}")
    print(f"  - Totale file copiati direttamente: {total_files_copied}")
    print(f"Errori riscontrati (file saltati): {total_errors}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")

    if not split_manifest["files"]:
        print("Nessun file processato o aggiunto al manifest.")
        return None, {}

    # Salva il manifest JSON finale
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e:
        print(f"!!! Errore critico durante salvataggio manifest: {e}")
        return None, {}

# --- END OF transcriptionUtils/splitAudio.py (CON num_workers) ---