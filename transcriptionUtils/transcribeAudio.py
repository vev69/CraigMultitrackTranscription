import whisper
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoFeatureExtractor, GenerationConfig
import torch
import sys
import os

# Funzione per caricare il modello selezionato
def load_model(model_choice):
    if model_choice == "whisper":
        MODEL_SIZE = "medium"  # Puoi cambiare a "small", "large", ecc.
        print(f"Caricamento modello Whisper di OpenAI ({MODEL_SIZE})...")
        return whisper.load_model(MODEL_SIZE)
    elif model_choice == "whispy_italian":
        print("Caricamento modello Whisper Italiano da Hugging Face...")
        model_path = "whispy/whisper_italian"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        config = GenerationConfig(
            max_length=448,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            num_beams=5,
            early_stopping=True,
            return_timestamps=True,
            no_timestamps_token_id=tokenizer.eos_token_id,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="it", task="transcribe"),
            suppress_tokens=[173, 50364],
            decoder_start_token_id=processor.tokenizer.bos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
        )
        model.generation_config = config

        # Creiamo la pipeline
        asr_pipeline = pipeline("automatic-speech-recognition",
                                model=model,
                                processor=processor,
                                tokenizer=tokenizer,
                                feature_extractor=feature_extractor,
                                return_timestamps=True,
                                chunk_length_s=30)  # Aggiunto per migliorare il timing

        return asr_pipeline

    else:
        raise ValueError("Modello non riconosciuto. Usa 'whisper' o 'whispy_italian'.")

# Funzione per trascrivere un file audio
def transcribeAudioFile(fileWithPath: str, model, model_choice):
    fileName = f'{fileWithPath}-TranscribedAudio'
    inProgressFileName = f'{fileName}-InProgress.txt'
    completedFileName = f'{fileName}.txt'
    
    if model_choice == "whisper":
        transcribedAudio = model.transcribe(fileWithPath, language="it", task='transcribe', fp16=False)
        with open(inProgressFileName, 'w+', encoding='utf-8') as file:
            for segment in transcribedAudio['segments']:
                file.write(f'[{segment["start"]}, {segment["end"]}] {segment["text"]}\n')
    
    elif model_choice == "whispy_italian":
        # Aumenta la qualità del transcoding usando più parametri
        result = model(
            fileWithPath,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": "it",
            }
        )
        
        with open(inProgressFileName, 'w+', encoding='utf-8') as file:
            # Verifica se il risultato è nel nuovo formato o nel vecchio formato
            if 'chunks' in result:
                segments = result['chunks']
                for segment in segments:
                    timestamp = segment['timestamp']
                    # Gestisci meglio i timestamp per garantire consistenza con il formato whisper
                    start_time = timestamp[0] if timestamp and timestamp[0] is not None else 0.0
                    end_time = timestamp[1] if timestamp and len(timestamp) > 1 and timestamp[1] is not None else start_time + 3.0
                    # Elimina eventuali spazi extra all'inizio e alla fine del testo
                    text = segment['text'].strip()
                    file.write(f'[{start_time}, {end_time}] {text}\n')
            else:
                # Nuovo formato di output (potrebbe cambiare nelle versioni future)
                if isinstance(result, dict) and 'text' in result:
                    # Se non ci sono timestamp, crea un singolo segmento
                    if not 'timestamps' in result or not result['timestamps']:
                        file.write(f'[0.0, 0.0] {result["text"]}\n')
                    else:
                        # Altrimenti, processa i timestamp esistenti
                        for i, (start, end, text) in enumerate(result['timestamps']):
                            file.write(f'[{start}, {end}] {text.strip()}\n')
                elif isinstance(result, list):
                    # Gestisci output in formato lista
                    for item in result:
                        if isinstance(item, dict) and 'text' in item:
                            start = item.get('start', 0.0)
                            end = item.get('end', start + 3.0)
                            file.write(f'[{start}, {end}] {item["text"].strip()}\n')
    
    # Applica post-processing per rimuovere allucinazioni
    post_process_transcription(inProgressFileName, completedFileName)
    
    return fileWithPath

# Funzione per post-processing del file trascritto
def post_process_transcription(input_file, output_file):
    """
    Applica tecniche di post-processing per migliorare la qualità della trascrizione
    e rimuovere allucinazioni comuni.
    """
    lines = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Errore nella lettura del file: {e}")
        return
    
    processed_lines = []
    prev_line_text = ""
    
    for line in lines:
        # Verifica se la riga è nel formato corretto
        if not line.startswith('[') or ']' not in line:
            continue
            
        # Estrai timestamp e testo
        try:
            bracket_end = line.find(']')
            timestamp_part = line[1:bracket_end]
            text_part = line[bracket_end+1:].strip()
            
            # Rimuovi ripetizioni immediate (una forma comune di allucinazione)
            if text_part == prev_line_text:
                continue
                
            # Verifica timestamp validi
            parts = timestamp_part.split(',')
            if len(parts) == 2:
                start = parts[0].strip()
                end = parts[1].strip()
                
                # Assicurati che i timestamp siano validi
                if start.lower() == 'none':
                    start = '0.0'
                if end.lower() == 'none':
                    # Se end è None, usa start + una durata ragionevole
                    end = str(float(start) + 3.0)
                
                # Ricrea la riga con timestamp corretti
                processed_line = f'[{start}, {end}] {text_part}\n'
                processed_lines.append(processed_line)
                prev_line_text = text_part
        except Exception as e:
            print(f"Errore nel processare la riga: {line} - {e}")
    
    # Rimuovi frasi duplicate non consecutive (altra forma di allucinazione)
    final_lines = []
    text_seen = set()
    
    for line in processed_lines:
        bracket_end = line.find(']')
        text_part = line[bracket_end+1:].strip()
        
        # Se il testo è molto breve (meno di 5 parole), preservalo anche se duplicato
        words = text_part.split()
        if len(words) <= 4 or text_part not in text_seen:
            final_lines.append(line)
            text_seen.add(text_part)
    
    # Scrivi il file processato
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
        print(f"Post-processing completato: {output_file}")
    except Exception as e:
        print(f"Errore nella scrittura del file: {e}")
