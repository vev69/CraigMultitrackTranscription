# --- START OF transcriptionUtils/splitAudio.py (MODIFICATO per Silence Splitting) ---

import os
import json
import math
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor # Manteniamo per parallelismo per file
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
# Importa la funzione di splitting basata sui silenzi
from pydub.silence import split_on_silence # type: ignore
import shutil

SPLIT_MANIFEST_FILENAME = "split_manifest.json"

# --- WORKER MODIFICATO per processare un singolo file originale con SILENCE SPLITTING ---
def _process_single_audio_file_for_split(original_path: str,
                                         split_audio_dir: str,
                                         # Rimuovi parametri non più usati direttamente qui
                                         # split_threshold_seconds: float,
                                         # target_chunk_duration_seconds: float
                                         # Aggiungi parametri per silence split
                                         min_silence_len_ms: int,
                                         silence_thresh_dbfs: float,
                                         keep_silence_ms: int,
                                         # Parametro soglia per decidere se NOMINARE come chunk
                                         split_naming_threshold_seconds: float
                                         ) -> tuple[dict, int, int, int]:
    """
    Worker eseguito in un processo separato. Analizza l'audio originale,
    lo divide basandosi sui silenzi (se supera la soglia di NOME),
    o lo copia/esporta come singolo file, e genera le voci per il manifest.

    Args:
        original_path: Percorso assoluto del file audio originale.
        split_audio_dir: Directory dove salvare i chunk/file esportati.
        min_silence_len_ms: Durata minima (ms) del silenzio per splittare.
        silence_thresh_dbfs: Soglia (dBFS) sotto cui è considerato silenzio.
        keep_silence_ms: Quanto silenzio (ms) mantenere all'inizio/fine dei chunk.
        split_naming_threshold_seconds: Durata originale sopra cui i file (anche se non divisi)
                                         vengono nominati con _part000.

    Returns:
        Una tupla:
        - dict: Voci del manifest generate per questo file (path_output -> metadata).
        - int: Flag di successo (1 se successo, 0 se errore).
        - int: Numero di chunk esportati (>=1 se successo).
        - int: Deprecato (era files_copied). Restituisce 0.
    """
    filename = os.path.basename(original_path)
    print(f"  [SplitWorker {os.getpid()}] Processing: {filename}")
    file_manifest_entries = {}
    chunks_exported_count = 0
    # files_copied = 0 # Non più usato direttamente
    success_flag = 0 # Inizia come fallimento

    try:
        base_name, ext = os.path.splitext(filename)
        # Carica l'audio
        audio = AudioSegment.from_file(original_path)
        duration_seconds = len(audio) / 1000.0
        print(f"  [SplitWorker {os.getpid()}] Duration for {filename}: {duration_seconds:.2f}s")

        # Esegui lo splitting basato sui silenzi
        # Nota: keep_silence può essere un bool o un int (ms)
        # Usiamo un valore per avere un po' di contesto
        audio_chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh_dbfs,
            keep_silence=keep_silence_ms,
            seek_step=1 # Controlla ogni ms
        )

        num_chunks_found = len(audio_chunks)
        print(f"  [SplitWorker {os.getpid()}] Found {num_chunks_found} non-silent segments for {filename}.")

        if num_chunks_found == 0:
            print(f"  [SplitWorker {os.getpid()}] WARN: No audio found above silence threshold for {filename}. Skipping.")
            # Non genera voci manifest, success_flag rimane 0
            return {}, 0, 0, 0

        # Determina se il file originale era "lungo" per decidere la nomenclatura
        # e il flag is_chunk nel manifest
        is_considered_long = duration_seconds > split_naming_threshold_seconds

        current_pos_ms = 0 # Tiene traccia della posizione nel file originale
        actual_exported_chunks = 0

        for i, chunk in enumerate(audio_chunks):
            # Calcola start/end assoluti APPROSSIMATIVI.
            # split_on_silence non restituisce i timestamp esatti del silenzio,
            # quindi stimiamo basandoci sulla posizione corrente.
            # Questo è un limite di pydub, librerie più avanzate potrebbero fare meglio.
            # Lo start assoluto è dove siamo arrivati nel file originale.
            # L'end assoluto è start + durata del chunk ATTUALE (che include keep_silence).
            # NOTA: Questo ignora la durata esatta del *silenzio* tra i chunk.
            # È un'approssimazione, ma necessaria con pydub.silence.split_on_silence.

            # Trova il punto di inizio effettivo di questo chunk nel file originale
            # Cerca la prima occorrenza del chunk a partire da current_pos_ms
            # Questo è computazionalmente costoso e impreciso con pydub!
            # *** APPROCCIO SEMPLIFICATO MA IMPRECISO ***
            # Assumiamo che i chunk siano contigui con silenzi rimossi/standardizzati
            # Calcoliamo start/end assoluti basandoci sull'indice e sulla durata
            # dei chunk PRECEDENTI.
            # Questo NON è accurato se i silenzi rimossi erano lunghi e variabili.

            # Manteniamo traccia del tempo di fine dell'ultimo chunk esportato
            # per stimare l'inizio del successivo.
            # Iniziamo da 0 per il primo chunk.
            # Per i successivi, lo start è stimato dopo la fine del precedente
            # (questo ignora il silenzio *reale* che c'era tra loro).

            # *** APPROCCIO ALTERNATIVO PIU' SEMPLICE: Calcolo Progressivo ***
            # Calcoliamo start_time basandoci sulla fine del chunk precedente.
            # Questo crea una timeline *relativa* ai soli chunk, ignorando i silenzi.
            # MODIFICHIAMO LA LOGICA DI COMBINAZIONE SUCCESSIVAMENTE? No, è peggio.

            # *** Compromesso: Usiamo la durata dei chunk come stima ***
            # Sebbene impreciso, è meglio di niente. Calcoliamo start/end
            # basandoci sulle durate dei chunk precedenti e sulla stima di `keep_silence`
            # Questo è complicato. Proviamo a mantenere la logica semplice:
            # Lo start assoluto è STIMATO come la fine assoluta del chunk precedente.
            # La fine assoluta è start + durata del chunk corrente.

            # RIASSUMENDO: Con pydub.split_on_silence, ottenere timestamp assoluti
            # accurati dei chunk rispetto all'originale è difficile.
            # L'approccio più robusto sarebbe usare una libreria di diarizzazione/VAD
            # che fornisca timestamp [start, end] per ogni segmento parlato.

            # *** SOLUZIONE PRAGMATICA PER ORA: ***
            # Usiamo un metodo che *preserva* i timestamp originali, se possibile.
            # Invece di usare `split_on_silence`, potremmo identificare i punti di silenzio
            # e poi tagliare manualmente usando `audio[start_ms:end_ms]`.

            # --- NUOVA LOGICA CON IDENTIFICAZIONE SILENZI ---
            from pydub.silence import detect_nonsilent # Usiamo questo per trovare i bordi

            # Trova tutti i segmenti NON silenziosi con i loro timestamp [start_ms, end_ms]
            # Aumenta leggermente min_silence_len per evitare tagli troppo frequenti
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len_ms + int(keep_silence_ms * 1.5), # Richiedi silenzio un po' più lungo per tagliare
                silence_thresh=silence_thresh_dbfs,
                seek_step=1
            )

            if not nonsilent_ranges:
                 print(f"  [SplitWorker {os.getpid()}] WARN: detect_nonsilent non ha trovato segmenti per {filename}. Skipping.")
                 return {}, 0, 0, 0

            # Raggruppa segmenti vicini per creare chunk più lunghi se necessario? No, per ora no.
            # Esporta ogni segmento non silenzioso come chunk.

            print(f"  [SplitWorker {os.getpid()}] Found {len(nonsilent_ranges)} non-silent ranges for {filename}.")
            chunks_exported_count = 0 # Resetta contatore

            for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
                 # Aggiungi il padding 'keep_silence' (ma senza sforare i limiti dell'audio)
                 padded_start_ms = max(0, start_ms - keep_silence_ms)
                 padded_end_ms = min(len(audio), end_ms + keep_silence_ms)

                 # Estrai il chunk con padding
                 chunk = audio[padded_start_ms:padded_end_ms]
                 
                 # Determina nome file e flag 'is_chunk'
                 # Se c'è solo 1 range E il file non era considerato lungo => nome originale
                 # Altrimenti => nome con _partXXX
                 if len(nonsilent_ranges) == 1 and not is_considered_long:
                      chunk_filename = f"{base_name}{ext}"
                      is_chunk_flag = False
                      chunk_index = 0
                 else:
                      chunk_filename = f"{base_name}_part{i:03d}{ext}"
                      is_chunk_flag = True
                      chunk_index = i

                 chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))

                 # I timestamp assoluti ora sono quelli rilevati da detect_nonsilent (con padding)
                 start_time_abs = padded_start_ms / 1000.0
                 end_time_abs = padded_end_ms / 1000.0

                 if os.path.exists(chunk_output_path):
                     # print(f"    [SplitWorker {os.getpid()}] Skipping existing chunk/file: {chunk_filename}")
                     pass
                 else:
                     # print(f"    [SplitWorker {os.getpid()}] Exporting chunk {i+1}/{len(nonsilent_ranges)}: {chunk_filename} [{start_time_abs:.2f}s - {end_time_abs:.2f}s]")
                     try:
                         chunk.export(chunk_output_path, format=ext.lstrip('.'))
                     except Exception as export_err:
                          print(f"  !!! [SplitWorker {os.getpid()}] ERROR exporting {chunk_filename}: {export_err}. Skipping chunk.")
                          continue # Salta questo chunk se l'export fallisce

                 # Aggiungi metadati al dizionario locale
                 file_manifest_entries[chunk_output_path] = {
                     "original_file": original_path,
                     "is_chunk": is_chunk_flag,
                     "chunk_index": chunk_index,
                     "start_time_abs": start_time_abs, # Timestamp con padding
                     "end_time_abs": end_time_abs,     # Timestamp con padding
                     "original_duration_seconds": duration_seconds
                 }
                 chunks_exported_count += 1 # Conta i chunk effettivamente aggiunti al manifest

            # Se abbiamo esportato almeno un chunk, considera successo
            if chunks_exported_count > 0:
                 success_flag = 1


    except pydub_exceptions.CouldntDecodeError as e:
        print(f"  !!! [SplitWorker {os.getpid()}] ERROR decoding {filename}: {e}. Assicurati che ffmpeg sia installato e nel PATH. Skipping.")
    except FileNotFoundError:
         print(f"  !!! [SplitWorker {os.getpid()}] ERROR File not found: {original_path}. Skipping.")
    except Exception as e:
        print(f"  !!! [SplitWorker {os.getpid()}] UNEXPECTED ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()

    # Ritorna le voci del manifest create, il flag di successo, e il numero di chunk creati
    # L'ultimo valore (ex files_copied) è sempre 0 ora
    return file_manifest_entries, success_flag, chunks_exported_count, 0


# --- Funzione Principale MODIFICATA per passare parametri di silenzio ---
def split_large_audio_files(original_audio_dir: str,
                            split_audio_dir: str,
                            # Rinomina soglia per chiarezza: soglia per NOMENCLATURA chunk
                            split_naming_threshold_seconds: float = 45 * 60,
                            # Parametri per il rilevamento silenzi
                            min_silence_len_ms: int = 700, # Default: 700ms
                            silence_thresh_dbfs: float = -35.0, # Default: -35 dBFS
                            keep_silence_ms: int = 150, # Default: 150ms
                            supported_extensions=(".flac",".m4a"),
                            num_workers: int | None = None
                            ) -> tuple[str | None, dict]:
    """
    Scansiona, divide (basandosi sui silenzi) o copia file audio in parallelo
    e crea un manifest JSON.
    """
    print(f"\n--- Avvio Divisione File Audio Parallela (Basata su Silenzi) ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Soglia durata per nome '_partXXX': {split_naming_threshold_seconds / 60:.1f} minuti")
    print(f"Parametri Silenzio: min_len={min_silence_len_ms}ms, threshold={silence_thresh_dbfs}dBFS, keep_silence={keep_silence_ms}ms")
    print(f"Estensioni supportate: {supported_extensions}")

    # ... (Logica num_workers, creazione dir, inizializzazione manifest invariate) ...
    if num_workers is None:
        try:
            num_workers = os.cpu_count(); num_workers = 4 if num_workers is None else num_workers
        except NotImplementedError: num_workers = 4
        print(f"Numero workers non specificato, uso default: {num_workers}")
    else:
        print(f"Numero workers specificato: {num_workers}")

    if not os.path.isdir(original_audio_dir): return None, {}
    try: os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e: print(f"Errore creazione directory split: {e}"); return None, {}

    split_manifest = {
        # Salva anche i parametri usati per lo split nel manifest
        "split_method": "silence_detection",
        "min_silence_len_ms": min_silence_len_ms,
        "silence_thresh_dbfs": silence_thresh_dbfs,
        "keep_silence_ms": keep_silence_ms,
        "split_naming_threshold_seconds": split_naming_threshold_seconds,
        "files": {}
    }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    total_files_processed_success = 0
    total_chunks_exported = 0 # Cambia nome contatore
    # total_files_copied = 0 # Non più tracciato separatamente
    total_errors = 0

    original_files = sorted([
        f for f in os.listdir(original_audio_dir)
        if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)
    ])

    if not original_files: return None, {}

    print(f"Trovati {len(original_files)} file audio originali da processare in parallelo...")

    # --- Logica Pool Executor (leggermente modificata per passare nuovi parametri) ---
    import platform
    import multiprocessing as mp
    start_method = 'spawn' if platform.system() != 'Linux' else None
    context = mp.get_context(start_method)

    with context.Pool(processes=num_workers) as pool:
        tasks_args = [
            (os.path.abspath(os.path.join(original_audio_dir, filename)),
             split_audio_dir,
             # Passa i parametri di silenzio al worker
             min_silence_len_ms,
             silence_thresh_dbfs,
             keep_silence_ms,
             split_naming_threshold_seconds)
            for filename in original_files
        ]
        results = []
        try:
            results = pool.starmap(_process_single_audio_file_for_split, tasks_args)
            print(f"\nCompletati {len(results)} task di analisi/split/copy dalla pool.")
        except Exception as pool_error:
             print(f"!!! Errore durante l'esecuzione del Pool: {pool_error}")
             return None, {}

        # Processa i risultati
        for result_tuple in results:
            try:
                # L'ultimo elemento della tupla non è più 'files_copied'
                file_manifest_entries, success_flag, chunks_exported, _ = result_tuple
                if success_flag:
                    total_files_processed_success += 1
                    split_manifest["files"].update(file_manifest_entries)
                    total_chunks_exported += chunks_exported
                else:
                    total_errors += 1
            except Exception as e:
                total_errors += 1
                print(f"!!! Errore processando risultato task: {e}")

    # --- Riepilogo Finale (aggiornato) ---
    print(f"\n--- Divisione Basata su Silenzi Completata ---")
    print(f"File originali esaminati: {len(original_files)}")
    print(f"File processati con successo (almeno un chunk esportato): {total_files_processed_success}")
    print(f"  - Totale chunk esportati: {total_chunks_exported}")
    print(f"Errori riscontrati (file saltati o senza audio): {total_errors}")
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

# --- END OF transcriptionUtils/splitAudio.py (MODIFICATO per Silence Splitting) ---