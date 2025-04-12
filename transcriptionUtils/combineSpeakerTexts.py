# --- START OF transcriptionUtils/combineSpeakerTexts.py ---

from dataclasses import dataclass, field # field non è più necessario qui
import os
import re

SUPPORTED_FILE_EXTENSIONS = (".txt")

# --- Definizione Dataclass MODIFICATA ---

@dataclass
class TranscribedLine:
    """Rappresenta una linea letta dal file di trascrizione individuale."""
    timeStampStart: float
    timeStampEnd: float
    lineText: str
    sourceFile: str | None = None # Per debug

@dataclass
class TranscibedLineWithSpeaker:
    """Rappresenta una linea combinata con lo speaker (SENZA ereditarietà)."""
    # Campi senza valore predefinito PRIMA
    timeStampStart: float
    timeStampEnd: float
    lineText: str
    speaker: str
    # Campi con valore predefinito DOPO
    sourceFile: str | None = None

# --- Il resto del file combineSpeakerTexts.py rimane INVARIATO ---
# ... (copia tutto il resto del file da __readFile__ in poi dalla versione precedente) ...

# Funzioni (assicurati che il resto sia corretto)
def __readFile__(pathToFile) -> list[TranscribedLine]:
    arrLinesAndTimes = []
    try:
        if not os.path.exists(pathToFile) or os.path.getsize(pathToFile) == 0: return []
        with open(f'{pathToFile}', 'r', encoding='utf-8') as file:
            filename_only = os.path.basename(pathToFile)
            for line_num, line in enumerate(file):
                line_stripped = line.strip()
                # Usa tupla per startswith per maggiore leggibilità/efficienza
                if not line_stripped or line_stripped.startswith(("[ERROR]", "[INFO]")):
                    continue
                lineComponent = __extractLineComponents__(line_stripped, line_num + 1, pathToFile)
                if lineComponent is not None:
                    # Assegna sourceFile all'oggetto TranscribedLine dopo la creazione
                    lineComponent.sourceFile = filename_only
                    arrLinesAndTimes.append(lineComponent)
    except Exception as e: print(f"Errore lettura file {pathToFile}: {e}")
    return arrLinesAndTimes

def __extractLineComponents__(line, line_num, filename) -> TranscribedLine | None:
    pattern = r'\[\s*([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)\s*\]\s*(.*)'
    match = re.match(pattern, line)
    if not match:
        print(f"Warn: Formato riga non valido (File: {os.path.basename(filename)}, Line: {line_num}): {line}")
        return None
    start_str, end_str, text = match.groups()
    try:
        start_time = float(start_str); end_time = float(end_str)
        if start_time < 0 or end_time < 0:
             print(f"Warn: Ignored negative ts (File: {os.path.basename(filename)}, Line: {line_num}): {line}")
             return None
        # Non correggere end<start qui, lo fa post_process_transcription
    except ValueError as e:
        print(f"Errore conversione ts (File: {os.path.basename(filename)}, Line: {line_num}): '{start_str}', '{end_str}' - {e}")
        return None
    clean_text = text.strip()
    # Crea l'oggetto base TranscribedLine; sourceFile viene aggiunto da __readFile__
    return TranscribedLine(start_time, end_time, clean_text)

def __processWhisperTranscribedAudio__(transcribedFile: list[TranscribedLine], filename: str) -> list[TranscribedLine]:
    """Applica solo pulizie MINIME e SICURE."""
    if not transcribedFile: return []
    # Rimuovi solo segmenti senza testo E con durata <= 0
    cleaned = [line for line in transcribedFile if line.lineText or line.timeStampEnd > line.timeStampStart]
    return cleaned

# --- Definizioni funzioni non usate ---
def __removeEmptyOrShortSegments__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]: pass
def __combineContiguousSegments__(transcribedFile: list[TranscribedLine], max_gap_s: float = 1.0) -> list[TranscribedLine]: pass
def __removeRepeatHallucinations__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]: pass
def __removeRepeatedSequences__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]: pass
def __sequencesAreSame__(seq1: list[TranscribedLine], seq2: list[TranscribedLine]) -> bool: pass

# VERSIONE MODIFICATA CHE ESTRAE IL NOME

import os # Assicurati che os e re siano importati all'inizio del file
import re

def __getSpeaker__(fileName: str) -> str:
    """Estrae l'identificatore NOME del relatore dal nome del file"""
    # Assumes format like '1-SpeakerName_partXXX-TranscribedAudio.txt'
    # or '1-SpeakerName-TranscribedAudio.txt'
    base_name = os.path.basename(fileName)
    # Rimuovi suffissi noti in modo più robusto
    suffixes_to_remove = ["-TranscribedAudio.txt", ".flac", ".wav"] # Aggiungi altri se necessario
    for suffix in suffixes_to_remove:
         if base_name.endswith(suffix):
              base_name = base_name[:-len(suffix)]
              break # Rimuovi solo il primo suffisso trovato

    # Rimuovi la parte "_partXXX" se presente (per l'approccio con divisione)
    base_name = re.sub(r'_part\d+$', '', base_name)

    # Match pattern: Cerca "Numero-NomeQualsiasi" all'inizio
    match = re.match(r"^\d+-(.+)", base_name) # Assicura che inizi con numero-trattino
    if match:
        # **** MODIFICA CHIAVE ****
        # Ritorna il secondo gruppo catturato (il nome dopo il trattino)
        speaker_name = match.group(1)
        # Potrebbe essere utile rimuovere eventuali estensioni residue se il cleanup sopra non le ha prese
        # Esempio ulteriore cleanup se il nome contenesse punti:
        # speaker_name = os.path.splitext(speaker_name)[0]
        return speaker_name.strip() # Rimuovi spazi extra
        # --------------------------

    # Fallback (se il formato non è Numero-Nome) - Potrebbe ritornare il nome file senza estensione
    print(f"Warn: Formato nome file non riconosciuto per estrarre nome speaker ({fileName}). Uso base_name: '{base_name}'")
    return os.path.splitext(base_name)[0].strip() if base_name else "unknown"

def __preprocessFiles__(transcription_output_dir: str, files: list[str]) -> dict[str, list[TranscribedLine]]:
    """Preprocessa (con pulizia minima) i file di trascrizione."""
    allFiles = {}
    print(f"Preprocessing files in directory: {transcription_output_dir}")
    sorted_files = sorted([f for f in files if f.endswith(SUPPORTED_FILE_EXTENSIONS) and "-TranscribedAudio.txt" in f and "-InProgress" not in f and ".failed" not in f])
    print(f"Found {len(sorted_files)} potential transcription files.")
    for file in sorted_files:
        filePath = os.path.join(transcription_output_dir, file)
        try:
            transcribedFile = __readFile__(filePath) # Legge come lista di TranscribedLine
            if transcribedFile:
                cleanedUpFile = __processWhisperTranscribedAudio__(transcribedFile, file) # Pulisce lista di TranscribedLine
                if not cleanedUpFile: continue
                speaker = __getSpeaker__(file)
                if speaker != "unknown":
                    if speaker in allFiles: allFiles[speaker].extend(cleanedUpFile)
                    else: allFiles[speaker] = cleanedUpFile
                    allFiles[speaker].sort(key=lambda x: x.timeStampStart) # Ordina le linee per questo speaker
        except Exception as e: print(f"Errore processing file {file}: {e}")
    print(f"Finished preprocessing. Read data for {len(allFiles)} speakers.")
    return allFiles

def __combineSpeakers__(speakerLines: dict[str, list[TranscribedLine]]) -> list[TranscibedLineWithSpeaker]:
    """Combina le trascrizioni e ordina."""
    if not speakerLines: print("Nessun dato speaker da combinare."); return []
    allLinesWithSpeaker = []
    print("\nCombining speaker lines...")
    for speaker, lines in speakerLines.items():
        for line in lines: # 'line' qui è un oggetto TranscribedLine
            # Controllo validità dati
            if not isinstance(line.timeStampStart, (int, float)) or \
               not isinstance(line.timeStampEnd, (int, float)) or \
               line.timeStampStart < 0 or line.timeStampEnd < line.timeStampStart:
                 print(f"    DEBUG - Skipping invalid line for speaker {speaker}: Start={line.timeStampStart}, End={line.timeStampEnd}, File={line.sourceFile}")
                 continue
            # Crea l'oggetto TranscibedLineWithSpeaker usando i campi di TranscribedLine
            allLinesWithSpeaker.append(TranscibedLineWithSpeaker(
                timeStampStart=line.timeStampStart, # Campo da TranscribedLine
                timeStampEnd=line.timeStampEnd,     # Campo da TranscribedLine
                lineText=line.lineText,         # Campo da TranscribedLine
                speaker=speaker,                # Variabile speaker corrente
                sourceFile=line.sourceFile      # Campo da TranscribedLine (può essere None)
            ))
    initial_count = len(allLinesWithSpeaker)
    print(f"Total lines read before sorting: {initial_count}")
    if not allLinesWithSpeaker: return []
    try:
        allLinesWithSpeaker.sort(key=lambda x: x.timeStampStart)
        print(f"Sorting complete. Final line count: {len(allLinesWithSpeaker)}")
    except Exception as e_sort: print(f"!!! ERRORE SORTING: {e_sort}"); return []

    # DEBUG
    print("\nDEBUG - First 10 sorted lines:")
    for i, line in enumerate(allLinesWithSpeaker[:10]): print(f"  {i+1}: [{line.timeStampStart:.2f}-{line.timeStampEnd:.2f}] {line.speaker}: {line.lineText[:50]}... (From: {line.sourceFile})")
    print("DEBUG - Last 10 sorted lines:")
    for i, line in enumerate(allLinesWithSpeaker[-10:]): print(f"  {initial_count-10+i+1}: [{line.timeStampStart:.2f}-{line.timeStampEnd:.2f}] {line.speaker}: {line.lineText[:50]}... (From: {line.sourceFile})")

    return allLinesWithSpeaker

def __writeTranscript__(filePathAndName: str, lines: list[TranscibedLineWithSpeaker]):
    """Scrive la trascrizione combinata."""
    if not lines: print("Nessuna linea da scrivere."); return
    try:
        os.makedirs(os.path.dirname(filePathAndName), exist_ok=True)
        with open(filePathAndName, 'w', encoding='utf-8') as file:
            print(f"Writing combined transcript to: {filePathAndName}")
            line_count = 0
            for line_index, line in enumerate(lines):
                try:
                     start_f = float(line.timeStampStart); end_f = float(line.timeStampEnd)
                     if start_f < 0 or end_f < start_f: continue
                     file.write(f'[{start_f:.2f}-{end_f:.2f}] {line.speaker}: {line.lineText.lstrip()}\n')
                     line_count += 1
                except Exception as e_write_line: print(f"Errore scrittura linea {line_index+1}: {e_write_line} - Dati: {line}")
            print(f"Trascrizione combinata salvata ({line_count}/{len(lines)} righe scritte).")
    except Exception as e: print(f"Errore scrittura file combinato {filePathAndName}: {e}")

def combineTranscribedSpeakerFiles(transcription_output_dir: str):
    """Funzione principale che combina i file."""
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