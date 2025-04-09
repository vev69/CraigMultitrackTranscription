from dataclasses import dataclass
import os
import re

SUPPORTED_FILE_EXTENSIONS = (".txt")

@dataclass
class TranscribedLine:
    timeStampStart: float
    timeStampEnd: float
    lineText: str

@dataclass
class TranscibedLineWithSpeaker(TranscribedLine):
    speaker: str

@dataclass
class SpeakerAndLines:
    speaker: str
    lines: list[TranscribedLine]

def __readFile__(pathToFile) -> list[TranscribedLine]:
    arrLinesAndTimes = []
    try:
        with open(f'{pathToFile}', 'r', encoding='utf-8') as file:
            while line := file.readline():
                lineComponent = __extractLineComponents__(line)
                if lineComponent is not None:
                    arrLinesAndTimes.append(lineComponent)
    except Exception as e:
        print(f"Errore nella lettura del file {pathToFile}: {e}")
    return arrLinesAndTimes

def __extractLineComponents__(line) -> TranscribedLine:
    # Pattern più robusto per estrarre timestamp e testo
    pattern = r'\[([^,]+),\s*([^\]]+)\]\s*(.*)'
    match = re.match(pattern, line.strip())
    
    if not match:
        print(f"Formato riga non valido: {line}")
        return None
        
    start_str, end_str, text = match.groups()
    
    try:
        start_time = None if start_str.lower() == 'none' else float(start_str)
        end_time = None if end_str.lower() == 'none' else float(end_str)
        
        # Se uno dei timestamp è None, gestiscilo meglio
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = start_time + 3.0  # Imposta una durata predefinita
            
        # Verifica che end_time sia maggiore di start_time
        if end_time <= start_time:
            end_time = start_time + 3.0
            
    except ValueError as e:
        print(f"Errore di conversione dei timestamp: {start_str}, {end_str} - {e}")
        return None
        
    # Pulisci il testo da caratteri indesiderati e spazi extra
    clean_text = text.strip()
    
    return TranscribedLine(start_time, end_time, clean_text)

def __processWhisperTranscribedAudio__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]:
    """
    Processa il file trascritto per migliorare la qualità rimuovendo allucinazioni
    e combinando segmenti simili.
    """
    # Step 1: Rimuovi ripetizioni immediate
    dehallucinatedFile = __removeRepeatHallucinations__(transcribedFile)
    
    # Step 2: Rimuovi sequenze ripetute (allucinazioni più complesse)
    dehallucinatedFile = __removeRepeatedSequences__(dehallucinatedFile)
    
    # Step 3: Combina segmenti contigui con lo stesso testo
    dehallucinatedFile = __combineContiguousSegments__(dehallucinatedFile)
    
    # Step 4: Rimuovi segmenti vuoti o troppo brevi
    dehallucinatedFile = __removeEmptyOrShortSegments__(dehallucinatedFile)
    
    return dehallucinatedFile

def __removeEmptyOrShortSegments__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]:
    """Rimuove segmenti vuoti o con testo troppo breve (es. solo punteggiatura)"""
    return [line for line in transcribedFile if line.lineText.strip() and len(line.lineText.strip()) > 1]

def __combineContiguousSegments__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]:
    """Combina segmenti contigui con lo stesso testo"""
    if not transcribedFile:
        return []
        
    result = []
    current = transcribedFile[0]
    
    for i in range(1, len(transcribedFile)):
        next_line = transcribedFile[i]
        # Se i testi sono uguali e i timestamp sono vicini, combinali
        if (current.lineText == next_line.lineText and 
            abs(current.timeStampEnd - next_line.timeStampStart) < 1.0):
            current.timeStampEnd = next_line.timeStampEnd
        else:
            result.append(current)
            current = next_line
    
    # Aggiungi l'ultimo segmento
    result.append(current)
    return result

def __removeRepeatHallucinations__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]:
    """Rimuove ripetizioni immediate dello stesso testo"""
    if not transcribedFile:
        return []
        
    result = [transcribedFile[0]]
    
    for i in range(1, len(transcribedFile)):
        if transcribedFile[i].lineText != result[-1].lineText:
            result.append(transcribedFile[i])
    
    return result

def __removeRepeatedSequences__(transcribedFile: list[TranscribedLine]) -> list[TranscribedLine]:
    """
    Rimuove sequenze ripetute di testo - una forma di allucinazione più complessa.
    """
    if len(transcribedFile) <= 1:
        return transcribedFile.copy()
        
    cloneList = transcribedFile.copy()
    maxSequenceLength = 10  # Limita la lunghezza massima della sequenza da cercare
    
    outerIndex = 0
    while outerIndex < len(cloneList) - 1:
        for seqLength in range(1, min(maxSequenceLength, len(cloneList) - outerIndex)):
            # Estrai la sequenza corrente
            currentSeq = cloneList[outerIndex:outerIndex + seqLength]
            
            # Cerca questa sequenza nel resto del testo
            searchIndex = outerIndex + seqLength
            while searchIndex + seqLength <= len(cloneList):
                # Estrai la sequenza da confrontare
                compareSeq = cloneList[searchIndex:searchIndex + seqLength]
                
                # Verifica se le sequenze sono uguali
                if __sequencesAreSame__(currentSeq, compareSeq):
                    # Rimuovi la sequenza ripetuta
                    del cloneList[searchIndex:searchIndex + seqLength]
                else:
                    searchIndex += 1
                    
        outerIndex += 1
    
    return cloneList

def __sequencesAreSame__(baseSequence: list[TranscribedLine], comparedToSequnece:list[TranscribedLine]) -> bool:
    """Verifica se due sequenze di TranscribedLine hanno lo stesso testo"""
    if len(baseSequence) != len(comparedToSequnece):
        return False
        
    for baseLine, comparedToline in zip(baseSequence, comparedToSequnece):
        if baseLine.lineText.strip() != comparedToline.lineText.strip():
            return False
    
    return True

def __getSpeaker__(fileName: str) -> str:
    """Estrae l'identificatore del relatore dal nome del file"""
    # Cerca il nome del relatore utilizzando pattern più robusto
    match = re.search(r'(\d+)\.flac', fileName)
    if match:
        return match.group(1)
    
    # Usa la logica originale come fallback
    endIndex = fileName.find('.flac')
    if endIndex == -1:
        endIndex = fileName.find('-TranscribedAudio.txt')
    
    return fileName[2:endIndex] if endIndex > 2 else "unknown"

def __preprocessFiles__(path: str, files: list[str]) -> dict[str, list[TranscribedLine]]:
    """Preprocessa tutti i file di trascrizione nella directory"""
    allFiles = {}
    for file in files:
        if file.endswith(SUPPORTED_FILE_EXTENSIONS) and "TranscribedAudio.txt" in file:
            try:
                transcribedFile = __readFile__(f'{path}{os.sep}{file}')
                if transcribedFile:
                    cleanedUpFile = __processWhisperTranscribedAudio__(transcribedFile)
                    speaker = __getSpeaker__(file)
                    allFiles[speaker] = cleanedUpFile
                    print(f"Processato file {file} - {len(cleanedUpFile)} segmenti")
                else:
                    print(f"Nessun contenuto valido nel file {file}")
            except Exception as e:
                print(f"Errore nel processare il file {file}: {e}")
    
    return allFiles

def __combineSpeakers__(speakerLines: dict[str, list[TranscribedLine]]) -> list[TranscibedLineWithSpeaker]:
    """Combina le trascrizioni di tutti i relatori in una singola lista ordinata per timestamp"""
    if not speakerLines:
        print("Nessun file di trascrizione valido trovato")
        return []
        
    indices = {speaker: 0 for speaker in speakerLines}
    allLines = []
    
    # Identifica quali speaker hanno ancora linee da processare
    iterableIndices = [speaker for speaker, lines in speakerLines.items() if indices[speaker] < len(lines)]
    
    while iterableIndices:
        # Mappa ogni speaker alla sua prossima linea
        speaker_lines = []
        for speaker in iterableIndices:
            if indices[speaker] < len(speakerLines[speaker]):
                speaker_lines.append((speaker, speakerLines[speaker][indices[speaker]]))
        
        # Ordina per timestamp di inizio
        if speaker_lines:
            speaker_lines.sort(key=lambda x: x[1].timeStampStart)
            speaker, line = speaker_lines[0]
            
            # Aggiungi la linea alla lista combinata
            allLines.append(TranscibedLineWithSpeaker(
                speaker=speaker,
                timeStampStart=line.timeStampStart,
                timeStampEnd=line.timeStampEnd,
                lineText=line.lineText
            ))
            
            # Avanza all'indice successivo per questo speaker
            indices[speaker] += 1
        
        # Aggiorna la lista degli speaker che hanno ancora linee da processare
        iterableIndices = [speaker for speaker, lines in speakerLines.items() if indices[speaker] < len(lines)]
    
    return allLines

def __writeTranscript__(filePathAndName: str, lines: list[TranscibedLineWithSpeaker]):
    """Scrive la trascrizione combinata in un file"""
    try:
        with open(filePathAndName, 'w', encoding='utf-8') as file:
            for line in lines:
                # Aggiungi timestamp al file di output per riferimento
                file.write(f'[{line.timeStampStart:.2f}-{line.timeStampEnd:.2f}] {line.speaker}: {line.lineText.lstrip()}\n')
        print(f"Trascrizione combinata salvata in: {filePathAndName}")
    except Exception as e:
        print(f"Errore nella scrittura del file {filePathAndName}: {e}")

def combineTranscribedSpeakerFiles(directoryOfFiles):
    """Funzione principale che combina i file di trascrizione in un unico documento"""
    print(f"Elaborazione dei file di trascrizione nella directory: {directoryOfFiles}")
    
    try:
        directoryListDir = os.listdir(directoryOfFiles)
        
        # Preprocessa i file
        preProcessedFiles = __preprocessFiles__(directoryOfFiles, directoryListDir)
        
        if not preProcessedFiles:
            print("Nessun file di trascrizione valido trovato nella directory")
            return
            
        # Combina le trascrizioni
        combinedTranscripts = __combineSpeakers__(preProcessedFiles)
        
        if not combinedTranscripts:
            print("Nessuna trascrizione combinata generata")
            return
            
        # Determina il nome del file di output
        lastSeparator = directoryOfFiles.rfind(os.sep)
        baseName = directoryOfFiles[lastSeparator+1:] if lastSeparator >= 0 else directoryOfFiles
        name = f'{directoryOfFiles}{os.sep}{baseName}AllAudio.txt'
        
        # Scrivi il file combinato
        __writeTranscript__(name, combinedTranscripts)
        
    except Exception as e:
        print(f"Errore durante la combinazione dei file: {e}")
