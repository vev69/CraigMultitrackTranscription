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
            decoder_start_token_id=processor.tokenizer.bos_token_id, # Imposta decoder_start_token_id
            bos_token_id=processor.tokenizer.bos_token_id, # Imposta anche bos_token_id per sicurezza
        )
        model.generation_config = config # Imposta la configurazione direttamente sul modello

        # Creiamo la pipeline
        asr_pipeline = pipeline("automatic-speech-recognition",
                                model=model,
                                processor=processor,
                                tokenizer=tokenizer,
                                feature_extractor=feature_extractor,
                                return_timestamps=True)

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
        result = model(fileWithPath) # La pipeline ora dovrebbe restituire i timestamp
        with open(inProgressFileName, 'w+', encoding='utf-8') as file:
            for segment in result['chunks']:
                timestamp = segment['timestamp']
                start_time = timestamp[0] if timestamp else None
                end_time = timestamp[1] if timestamp and len(timestamp) > 1 else None
                file.write(f'[{start_time}, {end_time}] {segment["text"]}\n')