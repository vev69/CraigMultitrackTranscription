# --- START OF transcriptionUtils/splitAudio.py ---

import os
import json
import math
import time
from pydub import AudioSegment, exceptions as pydub_exceptions

SPLIT_MANIFEST_FILENAME = "split_manifest.json"

def split_large_audio_files(original_audio_dir: str,
                            split_audio_dir: str,
                            split_threshold_seconds: float = 45 * 60, # 45 minuti
                            target_chunk_duration_seconds: float = 10 * 60, # 10 minuti (o usa 45min?)
                            supported_extensions=(".flac",)
                            ) -> tuple[str | None, dict]:
    """
    Scansiona una directory per file audio, divide quelli più lunghi della soglia
    in chunk di durata target e crea un manifest JSON con i metadati.
    I file più corti vengono copiati direttamente.

    Args:
        original_audio_dir: Directory contenente i file .flac originali.
        split_audio_dir: Directory dove salvare i file divisi (chunk) e il manifest.
        split_threshold_seconds: Durata minima (in secondi) per dividere un file.
        target_chunk_duration_seconds: Durata target approssimativa (in secondi) per ogni chunk.
        supported_extensions: Tuple di estensioni file da processare.

    Returns:
        Una tupla:
        - Percorso assoluto del file manifest JSON creato (o None se fallisce).
        - Un dizionario rappresentante il contenuto del manifest.
        Restituisce (None, {}) se non ci sono file validi o in caso di errore grave.
    """
    print(f"\n--- Avvio Divisione File Audio Lunghi ---")
    print(f"Directory originale: {original_audio_dir}")
    print(f"Directory output (split): {split_audio_dir}")
    print(f"Soglia per divisione: {split_threshold_seconds / 60:.1f} minuti")
    print(f"Durata target chunk: {target_chunk_duration_seconds / 60:.1f} minuti")

    if not os.path.isdir(original_audio_dir):
        print(f"Errore: Directory originale non trovata: {original_audio_dir}")
        return None, {}

    try:
        os.makedirs(split_audio_dir, exist_ok=True)
    except OSError as e:
        print(f"Errore creazione directory split: {e}")
        return None, {}

    split_manifest = {
        "split_threshold_seconds": split_threshold_seconds,
        "target_chunk_duration_seconds": target_chunk_duration_seconds,
        "files": {} # Qui andranno i metadati dei chunk/file copiati
    }
    manifest_path = os.path.join(split_audio_dir, SPLIT_MANIFEST_FILENAME)
    files_processed_count = 0
    chunks_created_count = 0
    files_copied_count = 0
    errors_count = 0

    original_files = sorted([
        f for f in os.listdir(original_audio_dir)
        if os.path.isfile(os.path.join(original_audio_dir, f)) and f.lower().endswith(supported_extensions)
    ])

    if not original_files:
        print("Nessun file audio supportato trovato nella directory originale.")
        # Scriviamo comunque un manifest vuoto? Forse meglio di no.
        return None, {}

    print(f"Trovati {len(original_files)} file audio da esaminare.")

    for filename in original_files:
        original_path = os.path.abspath(os.path.join(original_audio_dir, filename))
        base_name, ext = os.path.splitext(filename)
        print(f"\nEsaminando: {filename}...")

        try:
            audio = AudioSegment.from_file(original_path, format=ext.lstrip('.'))
            duration_seconds = len(audio) / 1000.0
            print(f"  Durata: {duration_seconds:.2f} secondi ({duration_seconds / 60:.1f} minuti)")

            if duration_seconds > split_threshold_seconds:
                # File lungo -> Dividi in chunk
                print(f"  File supera la soglia, verrà diviso.")
                num_chunks = math.ceil(duration_seconds / target_chunk_duration_seconds)
                actual_chunk_len_ms = math.ceil(len(audio) / num_chunks) # Lunghezza effettiva in ms per chunk
                print(f"  Previsti {num_chunks} chunk di circa {actual_chunk_len_ms / 1000.0:.2f} secondi.")

                for i in range(num_chunks):
                    start_ms = i * actual_chunk_len_ms
                    end_ms = min((i + 1) * actual_chunk_len_ms, len(audio)) # Non superare la fine
                    chunk = audio[start_ms:end_ms]

                    # Nome file chunk
                    chunk_filename = f"{base_name}_part{i:03d}{ext}"
                    chunk_output_path = os.path.abspath(os.path.join(split_audio_dir, chunk_filename))

                    # Calcola timestamp assoluti per il manifest
                    start_time_abs = start_ms / 1000.0
                    end_time_abs = end_ms / 1000.0 # Fine assoluta del chunk nell'originale

                    # Salta se il chunk esiste già (permette ripresa parziale della fase di split)
                    if os.path.exists(chunk_output_path):
                         print(f"    Skipping: Chunk {chunk_filename} già esistente.")
                    else:
                         print(f"    Creazione chunk {i+1}/{num_chunks}: {chunk_filename} [{start_time_abs:.2f}s - {end_time_abs:.2f}s]")
                         chunk.export(chunk_output_path, format=ext.lstrip('.'))

                    # Aggiungi metadati al manifest (sovrascrive se già esistente)
                    split_manifest["files"][chunk_output_path] = {
                        "original_file": original_path,
                        "is_chunk": True,
                        "chunk_index": i,
                        "start_time_abs": start_time_abs,
                        "end_time_abs": end_time_abs, # Fine assoluta di questo chunk
                        "original_duration_seconds": duration_seconds
                    }
                    chunks_created_count +=1 # Incrementa anche se skippato? Sì, conta come "gestito".

            else:
                # File corto -> Copia direttamente
                print(f"  File non supera la soglia, verrà copiato.")
                output_path = os.path.abspath(os.path.join(split_audio_dir, filename))

                if os.path.exists(output_path):
                     print(f"    Skipping: File {filename} già esistente in destinazione.")
                else:
                     import shutil
                     shutil.copy2(original_path, output_path)
                     print(f"    File copiato in: {output_path}")

                # Aggiungi metadati al manifest
                split_manifest["files"][output_path] = {
                        "original_file": original_path,
                        "is_chunk": False,
                        "chunk_index": 0, # Indice 0 per file non divisi
                        "start_time_abs": 0.0,
                        "end_time_abs": duration_seconds, # La fine è la durata totale
                        "original_duration_seconds": duration_seconds
                    }
                files_copied_count += 1

            files_processed_count += 1

        except pydub_exceptions.CouldntDecodeError as e:
            print(f"  ERRORE: Impossibile decodificare {filename}. Potrebbe mancare ffmpeg o file corrotto. {e}")
            errors_count += 1
        except Exception as e:
            print(f"  ERRORE inaspettato durante processamento di {filename}: {e}")
            errors_count += 1

    print(f"\n--- Divisione File Audio Completata ---")
    print(f"File originali esaminati: {len(original_files)}")
    print(f"File processati con successo (divisi o copiati): {files_processed_count}")
    print(f"  - File divisi in chunk: {chunks_created_count} chunk totali")
    print(f"  - File copiati direttamente: {files_copied_count}")
    print(f"Errori riscontrati: {errors_count}")
    print(f"Totale voci nel manifest: {len(split_manifest['files'])}")

    if not split_manifest["files"]:
         print("Nessun file processato o aggiunto al manifest.")
         return None, {}

    # Salva il manifest JSON
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(split_manifest, f, ensure_ascii=False, indent=4)
        print(f"Manifest salvato in: {manifest_path}")
        return manifest_path, split_manifest
    except Exception as e:
        print(f"!!! Errore critico durante salvataggio manifest: {e}")
        return None, {}

# --- END OF transcriptionUtils/splitAudio.py ---