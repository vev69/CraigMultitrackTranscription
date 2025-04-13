# --- START OF transcriptionUtils/combineSpeakerTexts.py (MODIFICATO) ---

from dataclasses import dataclass, field
import os
import re
import json # Necessario per caricare il manifest

SUPPORTED_FILE_EXTENSIONS = (".txt")
# Manifest globale caricato una sola volta
split_manifest_data = None

@dataclass
class TranscribedLine:
    timeStampStart: float # Ora rappresenta il timestamp ASSOLUTO
    timeStampEnd: float   # Ora rappresenta il timestamp ASSOLUTO
    lineText: str
    sourceChunkFile: str | None = None # Path assoluto del file chunk da cui deriva
    sourceOriginalFile: str | None = None # Path assoluto del file originale

@dataclass
class TranscibedLineWithSpeaker: # Manteniamo il typo per coerenza interna
    speaker: str
    timeStampStart: float # Assoluto
    timeStampEnd: float   # Assoluto
    lineText: str
    sourceChunkFile: str | None = field(default=None) # Assoluto
    sourceOriginalFile: str | None = field(default=None) # Assoluto


# --- Funzione __readFile__ MODIFICATA per calcolare timestamp assoluti ---
def __readFile__(pathToFile: str) -> list[TranscribedLine]:
    """
    Legge un file di trascrizione (.txt), estrae le linee e calcola
    i timestamp assoluti usando il manifest globale caricato.
    """
    global split_manifest_data
    arrLinesAndTimes = []
    if split_manifest_data is None:
        print(f"ERRORE: Manifest di split non caricato in __readFile__ per {pathToFile}. Salto.")
        return []

    try:
        if not os.path.exists(pathToFile) or os.path.getsize(pathToFile) == 0: return []

        # Determina il path del file chunk *originale* da cui deriva questo .txt
        # Assumiamo che il nome del file .txt segua lo schema: {chunk_basename}-TranscribedAudio.txt
        # Es: 'audio_split/speaker1_part001.flac' -> 'output/model/speaker1_part001-TranscribedAudio.txt'
        transcription_basename = os.path.basename(pathToFile)
        if not transcription_basename.endswith("-TranscribedAudio.txt"):
             print(f"Warn: Formato nome file trascrizione inatteso: {transcription_basename}. Impossibile derivare chunk originale."); return []

        chunk_basename = transcription_basename.replace("-TranscribedAudio.txt", "")
        chunk_abs_path_found = None

        # Cerca il path assoluto del chunk nel manifest usando il basename
        # Questo è un punto potenzialmente fragile se ci sono basename duplicati in diverse sub-dir
        # Sarebbe meglio se `transcribeAudioFile` potesse restituire/salvare il path del chunk usato.
        # Soluzione alternativa: Cercare il chunk_basename tra le chiavi del manifest.
        possible_matches = [p for p in split_manifest_data['files'].keys() if os.path.basename(p) == chunk_basename]

        if len(possible_matches) == 1:
             chunk_abs_path_found = possible_matches[0]
        elif len(possible_matches) > 1:
             print(f"Warn: Trovati multipli chunk corrispondenti a {chunk_basename} nel manifest. Impossibile determinare univocamente. Salto {pathToFile}.")
             return []
        else:
             print(f"Warn: Nessun chunk trovato nel manifest per {chunk_basename} (da {pathToFile}). Salto.")
             return []

        # Ottieni i metadati del chunk dal manifest
        chunk_metadata = split_manifest_data['files'].get(chunk_abs_path_found)
        if not chunk_metadata:
             print(f"Errore: Metadati non trovati per chunk {chunk_abs_path_found} nel manifest. Salto {pathToFile}.")
             return []

        chunk_start_time_abs = chunk_metadata.get('start_time_abs', 0.0)
        original_file_path = chunk_metadata.get('original_file', 'unknown_original')

        # Ora leggi il file di trascrizione riga per riga
        with open(pathToFile, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("[ERROR]") or line_stripped.startswith("[INFO]"): continue

                # Estrai timestamp RELATIVI e testo dalla riga
                lineComponentRel = __extractLineComponents__(line_stripped, line_num + 1, pathToFile, is_relative=True)
                if lineComponentRel is not None:
                    # Calcola timestamp ASSOLUTI
                    start_abs = chunk_start_time_abs + lineComponentRel.timeStampStart
                    end_abs = chunk_start_time_abs + lineComponentRel.timeStampEnd

                    # Crea l'oggetto TranscribedLine con timestamp assoluti e info sorgente
                    absoluteLine = TranscribedLine(
                        timeStampStart=start_abs,
                        timeStampEnd=end_abs,
                        lineText=lineComponentRel.lineText,
                        sourceChunkFile=chunk_abs_path_found,
                        sourceOriginalFile=original_file_path
                    )
                    arrLinesAndTimes.append(absoluteLine)

    except Exception as e: print(f"Errore lettura/processing file {pathToFile}: {e}")
    return arrLinesAndTimes

# --- Funzione __extractLineComponents__ MODIFICATA per gestire timestamp relativi ---
def __extractLineComponents__(line, line_num, filename, is_relative=False) -> TranscribedLine | None:
     """Estrae componenti, ritorna timestamp relativi se is_relative=True"""
     pattern = r'\[\s*([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)\s*\]\s*(.*)'
     match = re.match(pattern, line)
     if not match: print(f"Warn: Formato riga non valido (File: {os.path.basename(filename)}, Line: {line_num}): {line}"); return None
     start_str, end_str, text = match.groups()
     try:
         start_time = float(start_str); end_time = float(end_str)
         if start_time < 0 or end_time < 0: return None # Timestamp negativi non validi
         # Nota: Non aggiustiamo end_time < start_time qui, lasciamo che il post-proc lo faccia
     except ValueError as e: print(f"Errore conversione ts (File: {os.path.basename(filename)}, Line: {line_num}): '{start_str}', '{end_str}' - {e}"); return None
     clean_text = text.strip()
     # Ritorna un oggetto TranscribedLine temporaneo con timestamp relativi
     # I campi source verranno aggiunti dopo se necessario
     return TranscribedLine(start_time, end_time, clean_text)


# --- Funzione __processWhisperTranscribedAudio__ INVARIATA ---
# (La pulizia opera ancora sulle singole linee, ora con timestamp assoluti)
def __processWhisperTranscribedAudio__(transcribedFile: list[TranscribedLine], filename: str) -> list[TranscribedLine]:
    # ... (codice invariato) ...
    if not transcribedFile: return []
    # La pulizia si basa solo su testo vuoto o end<=start (assoluti ora)
    cleaned = [line for line in transcribedFile if line.lineText or line.timeStampEnd > line.timeStampStart]
    # Potrebbe essere necessario ri-validare end > start dopo l'aggiunta dell'offset?
    # No, se erano validi relativi (end_rel >= start_rel), rimangono validi assoluti.
    return cleaned

# --- Definizioni ALTRE funzioni di pulizia (INVARIATE e non usate) ---
# ...

# --- Funzione __getSpeaker__ MODIFICATA per usare il file originale ---
def __getSpeaker__(chunk_or_file_path: str) -> str:
    """
    Estrae lo speaker basandosi sul nome del file *originale* ottenuto dal manifest.
    """
    global split_manifest_data
    if split_manifest_data is None: return "unknown_speaker_no_manifest"

    # Ottieni i metadati per questo chunk/file
    metadata = split_manifest_data['files'].get(os.path.abspath(chunk_or_file_path))
    if not metadata:
        # Fallback: prova ad estrarre dal nome del chunk stesso (meno robusto)
        base_name_chunk = os.path.basename(chunk_or_file_path)
        # Rimuovi _partXXX e estensione
        name_part = re.sub(r'_part\d{3,}\..*$', '', base_name_chunk)
        # Rimuovi prefisso numerico se presente
        name_part = re.sub(r'^\d+-', '', name_part)
        print(f"Warn: Metadati non trovati per {chunk_or_file_path}. Fallback speaker: '{name_part}'")
        return name_part if name_part else "unknown_speaker_fallback"

    original_filename = os.path.basename(metadata.get('original_file', ''))
    if not original_filename: return "unknown_speaker_no_original"

    # Usa la logica originale sul nome del file originale
    base_name_original = os.path.splitext(original_filename)[0]

    # Logica di estrazione speaker (come prima ma su base_name_original)
    match_prefix = re.match(r"^\d+-(.*)", base_name_original)
    if match_prefix:
        name_part = match_prefix.group(1)
        match_suffix = re.match(r"(.*)_\d+$", name_part)
        speaker_name = match_suffix.group(1) if match_suffix else name_part
        if speaker_name: return speaker_name.strip() # Non serve loggare qui

    # Se non matcha il formato numerico, usa il nome base originale
    print(f"Info: Formato nome originale non standard ('{original_filename}'). Uso '{base_name_original}' come speaker.")
    return base_name_original.strip() if base_name_original else "unknown_speaker_direct"


# --- Funzione __preprocessFiles__ MODIFICATA ---
def __preprocessFiles__(transcription_output_dir: str, files: list[str]) -> dict[str, list[TranscribedLine]]:
    """ Preprocessa file di trascrizione, ricalcola timestamp e raggruppa per speaker. """
    allFilesBySpeaker = {}
    print(f"\nPreprocessing files di trascrizione in: {transcription_output_dir}")
    # Filtra i file .txt finali (non InProgress, non FAILED)
    transcription_files = sorted([
        f for f in files if f.endswith("-TranscribedAudio.txt")
    ])
    print(f"Found {len(transcription_files)} potential transcription files to process.")

    if not transcription_files: return {}

    for file in transcription_files:
        filePath = os.path.join(transcription_output_dir, file)
        try:
            # __readFile__ ora calcola timestamp assoluti e richiede il manifest globale
            processedLines = __readFile__(filePath) # Lista di TranscribedLine con ts assoluti
            if processedLines:
                # Determina lo speaker usando il path del chunk sorgente (dal primo elemento)
                # Assumiamo che tutte le linee in un file vengano dallo stesso chunk/file
                if processedLines[0].sourceChunkFile:
                     # Ottieni lo speaker dal file chunk/originale usando il manifest
                     speaker = __getSpeaker__(processedLines[0].sourceChunkFile)

                     # Applica pulizia minimale (es. rimuovi righe vuote o con end<=start)
                     cleanedUpLines = __processWhisperTranscribedAudio__(processedLines, file)
                     if not cleanedUpLines: continue

                     if speaker and not speaker.startswith("unknown"):
                         if speaker in allFilesBySpeaker:
                             allFilesBySpeaker[speaker].extend(cleanedUpLines)
                         else:
                             allFilesBySpeaker[speaker] = cleanedUpLines
                         # L'ordinamento finale avverrà globalmente dopo
                         print(f"  Processed file {file} for speaker '{speaker}' - {len(cleanedUpLines)} lines added (absolute ts).")
                     else:
                         print(f"  Warn: Could not determine valid speaker for file {file}. Skipping.")
                else:
                     print(f"  Warn: Source chunk file not found in processed lines for {file}. Skipping.")

        except Exception as e: print(f"Errore nel processare il file {file}: {e}")

    total_lines_read = sum(len(lines) for lines in allFilesBySpeaker.values())
    print(f"Finished preprocessing transcriptions. Read data for {len(allFilesBySpeaker)} speakers, total {total_lines_read} lines.")
    if not allFilesBySpeaker: print(f"Warning: No valid speaker data extracted from transcriptions in {transcription_output_dir}")

    # Ordina le linee per ogni speaker (anche se l'ordinamento globale è più importante)
    for speaker in allFilesBySpeaker:
         allFilesBySpeaker[speaker].sort(key=lambda x: x.timeStampStart)

    return allFilesBySpeaker


# --- Funzione __combineSpeakers__ MODIFICATA (leggermente, usa nuovi campi dataclass) ---
def __combineSpeakers__(speakerLines: dict[str, list[TranscribedLine]]) -> list[TranscibedLineWithSpeaker]:
    if not speakerLines: print("Nessun dato speaker valido da combinare."); return []
    allLinesWithSpeaker = []
    print("\nCombining speaker lines...")
    for speaker, lines in speakerLines.items():
        for line in lines:
            # Controllo validità timestamp (ora assoluti)
            if not isinstance(line.timeStampStart, (int, float)) or \
               not isinstance(line.timeStampEnd, (int, float)) or \
               line.timeStampStart < 0 or line.timeStampEnd < line.timeStampStart:
                 print(f"    DEBUG - Skipping invalid line for speaker '{speaker}': AbsStart={line.timeStampStart}, AbsEnd={line.timeStampEnd}, Chunk={line.sourceChunkFile}")
                 continue
            # Crea l'oggetto TranscibedLineWithSpeaker
            allLinesWithSpeaker.append(TranscibedLineWithSpeaker(
                speaker=speaker,
                timeStampStart=line.timeStampStart, # Assoluto
                timeStampEnd=line.timeStampEnd,     # Assoluto
                lineText=line.lineText,
                sourceChunkFile=line.sourceChunkFile, # Aggiunto
                sourceOriginalFile=line.sourceOriginalFile # Aggiunto
            ))
    initial_count = len(allLinesWithSpeaker)
    print(f"Total lines read before global sorting: {initial_count}")
    if not allLinesWithSpeaker: return []

    # ORDINAMENTO GLOBALE SUI TIMESTAMP ASSOLUTI
    try:
        allLinesWithSpeaker.sort(key=lambda x: x.timeStampStart)
        print(f"Global sorting by absolute timestamp complete. Final line count: {len(allLinesWithSpeaker)}")
    except Exception as e_sort:
        print(f"!!! ERRORE DURANTE L'ORDINAMENTO GLOBALE: {e_sort}"); return []

    # DEBUG (mostra timestamp assoluti)
    print("\nDEBUG - First 10 globally sorted lines (absolute timestamps):")
    for i, line in enumerate(allLinesWithSpeaker[:10]):
        print(f"  {i+1}: [{line.timeStampStart:.2f}-{line.timeStampEnd:.2f}] {line.speaker}: {line.lineText[:50]}... (Chunk: {os.path.basename(line.sourceChunkFile or '')})")

    return allLinesWithSpeaker


# --- Funzione __writeTranscript__ INVARIATA ---
# (Scrive ancora speaker: testo, basandosi sulle linee ordinate globalmente)
def __writeTranscript__(filePathAndName: str, lines: list[TranscibedLineWithSpeaker]):
    # ... (codice invariato) ...
    if not lines: print("Nessuna linea da scrivere."); return
    try:
        os.makedirs(os.path.dirname(filePathAndName), exist_ok=True)
        with open(filePathAndName, 'w', encoding='utf-8') as file:
            print(f"Writing combined transcript (NO TIMESTAMPS) to: {filePathAndName}")
            line_count = 0; last_speaker = None; lines_written_no_ts = 0
            for line_index, line in enumerate(lines):
                try:
                     current_speaker = line.speaker
                     if not current_speaker or current_speaker.startswith("unknown"): continue
                     if line.lineText is None: continue

                     # Scrivi NomeSpeaker: solo se cambia rispetto alla riga precedente
                     # if current_speaker != last_speaker:
                     #      file.write(f'\n{current_speaker}:\n') # Aggiungi spazio e nome
                     #      last_speaker = current_speaker
                     # file.write(f'  {line.lineText.strip()}\n') # Indenta il testo

                     # Formato semplice NomeSpeaker: Testo per ogni riga
                     file.write(f'{current_speaker}: {line.lineText.strip()}\n')
                     lines_written_no_ts += 1

                except Exception as e_write_line: print(f"Errore scrittura linea {line_index+1}: {e_write_line}")
            print(f"Trascrizione combinata (senza timestamp) salvata ({lines_written_no_ts}/{len(lines)} righe valide scritte).")
    except Exception as e: print(f"Errore scrittura file combinato {filePathAndName}: {e}")

# --- Funzione combineTranscribedSpeakerFiles MODIFICATA per accettare manifest ---
def combineTranscribedSpeakerFiles(transcription_output_dir: str, split_manifest_path: str):
    """
    Funzione principale per combinare le trascrizioni. Carica il manifest
    necessario per il ricalcolo dei timestamp.
    """
    global split_manifest_data # Usa la variabile globale
    print(f"\n--- Combinazione Trascrizioni per: {transcription_output_dir} ---")
    print(f"Utilizzando manifest: {split_manifest_path}")

    if not os.path.isdir(transcription_output_dir):
        print(f"Errore: Directory trascrizioni non trovata: {transcription_output_dir}"); return
    if not os.path.isfile(split_manifest_path):
        print(f"Errore: File manifest non trovato: {split_manifest_path}"); return

    # Carica il manifest una sola volta
    try:
        with open(split_manifest_path, 'r', encoding='utf-8') as f:
            split_manifest_data = json.load(f)
        if not split_manifest_data or 'files' not in split_manifest_data:
            print("Errore: Manifest caricato non valido o vuoto.")
            split_manifest_data = None # Resetta se non valido
            return
        print(f"Manifest caricato con successo ({len(split_manifest_data['files'])} voci).")
    except Exception as e:
        print(f"Errore caricamento manifest JSON '{split_manifest_path}': {e}")
        split_manifest_data = None # Resetta in caso di errore
        return

    try:
        directoryListDir = os.listdir(transcription_output_dir)
        # __preprocessFiles__ ora usa il manifest globale caricato
        preProcessedFilesBySpeaker = __preprocessFiles__(transcription_output_dir, directoryListDir)

        if not preProcessedFilesBySpeaker:
            print(f"Nessun dato valido estratto dai file di trascrizione in {transcription_output_dir}. Salto combinazione."); return

        # __combineSpeakers__ ora ordina usando timestamp assoluti
        combinedTranscripts = __combineSpeakers__(preProcessedFilesBySpeaker)

        if not combinedTranscripts:
            print("Nessuna trascrizione combinata generata dopo l'ordinamento."); return

        model_name = os.path.basename(transcription_output_dir);
        if not model_name: model_name = "combined"
        output_filename = f'{model_name}_AllAudio_Combined.txt' # Nome file leggermente diverso
        full_output_path = os.path.join(transcription_output_dir, output_filename)

        # __writeTranscript__ scrive l'output finale ordinato
        __writeTranscript__(full_output_path, combinedTranscripts)

    except FileNotFoundError: print(f"Errore: Directory non trovata durante combinazione: {transcription_output_dir}")
    except Exception as e: print(f"Errore generico durante combinazione per {transcription_output_dir}: {e}")
    finally:
        split_manifest_data = None # Pulisci il manifest globale dopo l'uso

# --- END OF transcriptionUtils/combineSpeakerTexts.py (MODIFICATO) ---