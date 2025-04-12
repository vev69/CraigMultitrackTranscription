# --- START OF transcriptionUtils/combineSpeakerTexts.py ---

from dataclasses import dataclass, field # Importa 'field'
import os
import re

SUPPORTED_FILE_EXTENSIONS = (".txt")

@dataclass
class TranscribedLine:
    # Definisci prima gli argomenti senza default
    timeStampStart: float
    timeStampEnd: float
    lineText: str
    # Poi quelli con default
    sourceFile: str | None = None

@dataclass
class TranscibedLineWithSpeaker:
    # Definisci prima i campi richiesti specifici di questa classe
    speaker: str
    # Poi i campi ereditati richiesti da TranscribedLine
    timeStampStart: float
    timeStampEnd: float
    lineText: str
    # Infine, i campi con default (ereditato o specifico)
    # Usiamo field(default=None) per essere espliciti
    sourceFile: str | None = field(default=None)

# --- Funzioni __readFile__ e __extractLineComponents__ INVARIATE ---
# (Leggono ancora i timestamp dai file individuali per permettere l'ordinamento)
def __readFile__(pathToFile) -> list[TranscribedLine]:
    # ... (codice invariato) ...
    arrLinesAndTimes = []
    try:
        if not os.path.exists(pathToFile) or os.path.getsize(pathToFile) == 0: return []
        with open(f'{pathToFile}', 'r', encoding='utf-8') as file:
            filename_only = os.path.basename(pathToFile)
            for line_num, line in enumerate(file):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("[ERROR]") or line_stripped.startswith("[INFO]"): continue
                lineComponent = __extractLineComponents__(line_stripped, line_num + 1, pathToFile)
                if lineComponent is not None:
                    lineComponent.sourceFile = filename_only
                    arrLinesAndTimes.append(lineComponent)
    except Exception as e: print(f"Errore lettura file {pathToFile}: {e}")
    return arrLinesAndTimes

def __extractLineComponents__(line, line_num, filename) -> TranscribedLine | None:
     # ... (codice invariato) ...
    pattern = r'\[\s*([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)\s*\]\s*(.*)'
    match = re.match(pattern, line)
    if not match: print(f"Warn: Formato riga non valido (File: {os.path.basename(filename)}, Line: {line_num}): {line}"); return None
    start_str, end_str, text = match.groups()
    try:
        start_time = float(start_str); end_time = float(end_str)
        if start_time < 0 or end_time < 0: return None
    except ValueError as e: print(f"Errore conversione ts (File: {os.path.basename(filename)}, Line: {line_num}): '{start_str}', '{end_str}' - {e}"); return None
    clean_text = text.strip()
    return TranscribedLine(start_time, end_time, clean_text)


# --- Funzione __processWhisperTranscribedAudio__ INVARIATA (pulizia minimale) ---
def __processWhisperTranscribedAudio__(transcribedFile: list[TranscribedLine], filename: str) -> list[TranscribedLine]:
    # ... (codice invariato) ...
    if not transcribedFile: return []
    cleaned = [line for line in transcribedFile if line.lineText or line.timeStampEnd > line.timeStampStart]
    return cleaned

# --- Definizioni ALTRE funzioni di pulizia (NON USATE - codice invariato) ---
def __removeEmptyOrShortSegments__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]: pass
def __combineContiguousSegments__(transcribedFile: list[TranscribedLine], max_gap_s: float = 1.0) -> list[TranscribedLine]: pass
def __removeRepeatHallucinations__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]: pass
def __removeRepeatedSequences__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]: pass
def __sequencesAreSame__(seq1: list[TranscribedLine], seq2: list[TranscribedLine]) -> bool: pass

# --- Funzione __getSpeaker__ INVARIATA (estrae nome completo) ---
def __getSpeaker__(fileName: str) -> str:
    # ... (codice invariato dall'ultima versione) ...
    base_name = os.path.basename(fileName).replace("-TranscribedAudio.txt", "").replace(".flac", "")
    match_prefix = re.match(r"^\d+-(.*)", base_name)
    if match_prefix:
        name_part = match_prefix.group(1)
        match_suffix = re.match(r"(.*)_\d+$", name_part)
        speaker_name = match_suffix.group(1) if match_suffix else name_part
        if speaker_name: print(f"  Speaker extracted from '{fileName}': '{speaker_name}'"); return speaker_name.strip()
    print(f"Warn: Could not extract full speaker name from: {fileName}. Using fallback.")
    fallback_name = os.path.basename(fileName).replace("-TranscribedAudio.txt", "")
    return fallback_name if fallback_name else "unknown"

# --- Funzione __preprocessFiles__ INVARIATA ---
def __preprocessFiles__(transcription_output_dir: str, files: list[str]) -> dict[str, list[TranscribedLine]]:
    # ... (codice invariato) ...
    allFiles = {}
    print(f"Preprocessing files in directory: {transcription_output_dir}")
    sorted_files = sorted([f for f in files if f.endswith(SUPPORTED_FILE_EXTENSIONS) and "-TranscribedAudio.txt" in f and "-InProgress" not in f and ".failed" not in f])
    print(f"Found {len(sorted_files)} potential transcription files to process.")
    for file in sorted_files:
        filePath = os.path.join(transcription_output_dir, file)
        try:
            transcribedFile = __readFile__(filePath)
            if transcribedFile:
                cleanedUpFile = __processWhisperTranscribedAudio__(transcribedFile, file)
                if not cleanedUpFile: continue
                speaker = __getSpeaker__(file)
                if speaker != "unknown":
                    if speaker in allFiles: allFiles[speaker].extend(cleanedUpFile)
                    else: allFiles[speaker] = cleanedUpFile
                    allFiles[speaker].sort(key=lambda x: x.timeStampStart)
                    print(f"  Processed file {file} for speaker '{speaker}' - {len(cleanedUpFile)} lines.")
        except Exception as e: print(f"Errore nel processare il file {file}: {e}")
    total_lines_read = sum(len(lines) for lines in allFiles.values())
    print(f"Finished preprocessing. Read data for {len(allFiles)} speakers, total {total_lines_read} lines.")
    if not allFiles: print(f"Warning: No valid speaker files found/processed in {transcription_output_dir}")
    return allFiles

# --- Funzione __combineSpeakers__ INVARIATA (usa nuovo dataclass) ---
# La logica interna non cambia, ma ora opera su oggetti TranscibedLineWithSpeaker
# con l'ordine dei campi corretto.
def __combineSpeakers__(speakerLines: dict[str, list[TranscribedLine]]) -> list[TranscibedLineWithSpeaker]:
    if not speakerLines: print("Nessun dato speaker valido da combinare."); return []
    allLinesWithSpeaker = []
    print("\nCombining speaker lines...")
    for speaker, lines in speakerLines.items():
        for line in lines:
            # Controllo validità dati
            if not isinstance(line.timeStampStart, (int, float)) or \
               not isinstance(line.timeStampEnd, (int, float)) or \
               line.timeStampStart < 0 or line.timeStampEnd < line.timeStampStart:
                 print(f"    DEBUG - Skipping invalid line for speaker '{speaker}': Start={line.timeStampStart}, End={line.timeStampEnd}, File={line.sourceFile}")
                 continue
            # Crea l'oggetto TranscibedLineWithSpeaker nel nuovo ordine
            allLinesWithSpeaker.append(TranscibedLineWithSpeaker(
                speaker=speaker, # Prima speaker
                timeStampStart=line.timeStampStart,
                timeStampEnd=line.timeStampEnd,
                lineText=line.lineText,
                sourceFile=line.sourceFile # sourceFile (con default) per ultimo
            ))
    initial_count = len(allLinesWithSpeaker)
    print(f"Total lines read before sorting: {initial_count}")
    if not allLinesWithSpeaker: return []
    try:
        allLinesWithSpeaker.sort(key=lambda x: x.timeStampStart) # Ordina per timestamp
        print(f"Sorting by timestamp complete. Final line count: {len(allLinesWithSpeaker)}")
    except Exception as e_sort:
        print(f"!!! ERRORE DURANTE L'ORDINAMENTO: {e_sort}"); return []
    # DEBUG
    print("\nDEBUG - First 10 sorted lines (timestamps used for sorting):")
    for i, line in enumerate(allLinesWithSpeaker[:10]): print(f"  {i+1}: [{line.timeStampStart:.2f}-{line.timeStampEnd:.2f}] {line.speaker}: {line.lineText[:50]}... (From: {line.sourceFile})")
    return allLinesWithSpeaker


# --- Funzione __writeTranscript__ INVARIATA (usa nuovo dataclass) ---
def __writeTranscript__(filePathAndName: str, lines: list[TranscibedLineWithSpeaker]):
    # ... (codice invariato - scrive speaker: testo) ...
    if not lines: print("Nessuna linea da scrivere."); return
    try:
        os.makedirs(os.path.dirname(filePathAndName), exist_ok=True)
        with open(filePathAndName, 'w', encoding='utf-8') as file:
            print(f"Writing combined transcript (NO TIMESTAMPS) to: {filePathAndName}")
            line_count = 0
            for line_index, line in enumerate(lines):
                try:
                     if not line.speaker or line.speaker == "unknown": continue
                     if line.lineText is None: continue
                     # Scrive nel formato NomeSpeaker: Testo
                     file.write(f'{line.speaker}: {line.lineText.lstrip()}\n')
                     line_count += 1
                except Exception as e_write_line: print(f"Errore scrittura linea {line_index+1}: {e_write_line} - Dati: Speaker='{getattr(line, 'speaker', 'N/A')}', TextType={type(getattr(line, 'lineText', None))}")
            print(f"Trascrizione combinata salvata ({line_count}/{len(lines)} righe scritte).")
    except Exception as e: print(f"Errore scrittura file combinato {filePathAndName}: {e}")


# --- Funzione combineTranscribedSpeakerFiles INVARIATA ---
def combineTranscribedSpeakerFiles(transcription_output_dir: str):
    # ... (codice invariato) ...
    print(f"\n--- Combinazione Trascrizioni per: {transcription_output_dir} ---")
    if not os.path.isdir(transcription_output_dir): print(f"Errore: Directory non trovata: {transcription_output_dir}"); return
    try:
        directoryListDir = os.listdir(transcription_output_dir)
        preProcessedFiles = __preprocessFiles__(transcription_output_dir, directoryListDir)
        if not preProcessedFiles: print(f"Nessun file valido trovato in {transcription_output_dir}."); return
        combinedTranscripts = __combineSpeakers__(preProcessedFiles)
        if not combinedTranscripts: print("Nessuna trascrizione combinata generata."); return
        model_name = os.path.basename(transcription_output_dir);
        if not model_name: model_name = "combined"
        output_filename = f'{model_name}_AllAudio.txt'
        full_output_path = os.path.join(transcription_output_dir, output_filename)
        __writeTranscript__(full_output_path, combinedTranscripts)
    except FileNotFoundError: print(f"Errore: Directory non trovata durante combinazione: {transcription_output_dir}")
    except Exception as e: print(f"Errore generico durante combinazione per {transcription_output_dir}: {e}")

# --- END OF transcriptionUtils/combineSpeakerTexts.py ---