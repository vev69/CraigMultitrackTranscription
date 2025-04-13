# --- START OF transcriptionUtils/transcribeAudio.py ---
import whisper
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig
import torch
import sys
import os
import re
import traceback
import time # Per pause potenziali
import numpy as np # Necessario per noisereduce

# --- Import per preprocessing ---
try:
    import soundfile as sf
    import noisereduce as nr
    SOUNDFILE_AVAILABLE = True
    NOISEREDUCE_AVAILABLE = True
    print("Moduli soundfile e noisereduce importati per preprocessing.")
except ImportError as e:
    print(f"ATTENZIONE: Modulo mancante per preprocessing ({e}). La riduzione del rumore non verrà eseguita.")
    SOUNDFILE_AVAILABLE = False
    NOISEREDUCE_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    PYDUB_AVAILABLE = True
    print("Modulo pydub importato per normalizzazione.")
except ImportError:
    print("ATTENZIONE: Modulo pydub non trovato (pip install pydub). La normalizzazione non verrà eseguita.")
    PYDUB_AVAILABLE = False
# -----------------------------

BATCH_SIZE_HF = 16

try:
    from optimum.bettertransformer import BetterTransformer
    OPTIMUM_AVAILABLE = True
    print("Modulo Optimum BetterTransformer importato.")
except ImportError:
    print("Modulo Optimum BetterTransformer non trovato (pip install optimum[\"bettertransformer\"]).")
    OPTIMUM_AVAILABLE = False

# --- NUOVA FUNZIONE DI PREPROCESSING ---
def preprocess_audio(input_path: str, output_path: str, noise_reduce=True, normalize=True, target_dbfs=-3.0):
    """
    Applica riduzione rumore (opzionale) e normalizzazione (opzionale) all'audio.
    Salva il risultato in output_path.
    Ritorna True se il processing ha avuto successo, False altrimenti.
    """
    print(f"Preprocessing audio: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    processing_done = False

    # --- Riduzione Rumore (con noisereduce) ---
    if noise_reduce and SOUNDFILE_AVAILABLE and NOISEREDUCE_AVAILABLE:
        try:
            data, rate = sf.read(input_path)
            print(f"  Audio caricato ({data.shape}, rate={rate}Hz). Applicazione riduzione rumore...")

            # Converti in mono se necessario (noisereduce lavora meglio su mono)
            if data.ndim > 1 and data.shape[1] > 1:
                 print("  Conversione audio in mono...")
                 data = np.mean(data, axis=1)

            # Applica riduzione rumore (stima rumore dall'intero clip)
            # Parametri da aggiustare: prop_decrease, n_fft, hop_length etc.
            reduced_noise_data = nr.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=0.85)
            print("  Riduzione rumore applicata.")
            # Sovrascrivi i dati originali con quelli puliti per il passo successivo
            data_to_process = reduced_noise_data
            rate_to_process = rate
            processing_done = True # Segna che almeno un passo è stato fatto
            # Nota: non salviamo qui, passiamo i dati al passo successivo o al salvataggio finale

        except Exception as e_nr:
            print(f"  ERRORE durante riduzione rumore: {e_nr}. Si procederà con l'audio originale.")
            # Se fallisce, usa i dati originali per la normalizzazione (se richiesta)
            try:
                data_to_process, rate_to_process = sf.read(input_path)
                if data_to_process.ndim > 1 and data_to_process.shape[1] > 1: # Assicura mono se letto ora
                     data_to_process = np.mean(data_to_process, axis=1)
            except Exception as e_read_fallback:
                 print(f"  ERRORE lettura audio originale dopo fallimento NR: {e_read_fallback}")
                 return False # Fallimento grave

    else:
        # Se la riduzione rumore non è richiesta o non disponibile, leggi i dati originali
        # ma preparali per la normalizzazione
        if normalize and (SOUNDFILE_AVAILABLE or PYDUB_AVAILABLE):
             try:
                 data_to_process, rate_to_process = sf.read(input_path)
                 if data_to_process.ndim > 1 and data_to_process.shape[1] > 1:
                      data_to_process = np.mean(data_to_process, axis=1)
             except Exception as e_read_orig:
                  print(f"  ERRORE lettura audio originale per normalizzazione: {e_read_orig}")
                  return False
        else:
             # Nessun processing richiesto/possibile, non serve leggere/scrivere
             print("  Nessun preprocessing audio eseguito (skip noise reduction e/o normalization).")
             # Potremmo copiare il file originale, ma è più efficiente non farlo
             # e usare direttamente l'originale nella trascrizione.
             # Indichiamo che non serve usare il file di output.
             return False # Indica che l'output_path non contiene dati processati


    # --- Normalizzazione (con Pydub - più robusto per diversi formati/tipi) ---
    # Usiamo Pydub qui perché gestisce meglio la conversione tipi e il salvataggio
    audio_processed_further = False
    if normalize and PYDUB_AVAILABLE:
        temp_input_for_pydub = None
        try:
            # Se abbiamo processato con noisereduce, dobbiamo salvare un file temporaneo
            # perché pydub preferisce leggere da file.
            if processing_done: # processing_done è True solo se noisereduce ha funzionato
                temp_input_for_pydub = output_path + ".nr_temp.wav" # Usa wav per compatibilità
                print(f"  Salvataggio temporaneo audio con riduzione rumore per normalizzazione: {os.path.basename(temp_input_for_pydub)}")
                sf.write(temp_input_for_pydub, data_to_process, rate_to_process)
                input_file_pydub = temp_input_for_pydub
            else:
                # Altrimenti normalizza direttamente l'originale
                input_file_pydub = input_path

            print(f"  Normalizzazione audio (target {target_dbfs} dBFS)...")
            audio = AudioSegment.from_file(input_file_pydub)
            normalized_audio = audio.apply_gain(target_dbfs - audio.dBFS)
            # Salva il file normalizzato nel percorso di output finale
            # Usa un formato lossless come FLAC o WAV di alta qualità
            normalized_audio.export(output_path, format="flac")
            print(f"  Audio normalizzato salvato in: {os.path.basename(output_path)}")
            audio_processed_further = True # Indica che abbiamo scritto in output_path

        except CouldntDecodeError as e_pydub_decode:
             print(f"  ERRORE Pydub: Impossibile decodificare {os.path.basename(input_file_pydub)} - {e_pydub_decode}")
             print("  Potrebbe mancare ffmpeg? (Installalo e assicurati sia nel PATH)")
             # Se la normalizzazione fallisce ma la riduzione rumore era stata fatta,
             # salva almeno il risultato della riduzione rumore.
             if processing_done and not audio_processed_further:
                  try:
                       print(f"  Salvataggio fallback: solo riduzione rumore in {os.path.basename(output_path)}")
                       sf.write(output_path, data_to_process, rate_to_process, format='FLAC')
                       audio_processed_further = True
                  except Exception as e_save_fallback:
                       print(f"  ERRORE salvataggio fallback: {e_save_fallback}")
                       return False # Fallimento
             else:
                  return False # Fallimento se non si può né normalizzare né salvare NR

        except Exception as e_norm:
            print(f"  ERRORE durante normalizzazione: {e_norm}")
            # Come sopra, salva almeno NR se possibile
            if processing_done and not audio_processed_further:
                 try:
                      print(f"  Salvataggio fallback: solo riduzione rumore in {os.path.basename(output_path)}")
                      sf.write(output_path, data_to_process, rate_to_process, format='FLAC')
                      audio_processed_further = True
                 except Exception as e_save_fallback:
                      print(f"  ERRORE salvataggio fallback: {e_save_fallback}")
                      return False
            else:
                 return False
        finally:
             # Rimuovi il file temporaneo di noisereduce se creato
             if temp_input_for_pydub and os.path.exists(temp_input_for_pydub):
                  try: os.remove(temp_input_for_pydub)
                  except OSError: pass

    elif processing_done and not audio_processed_further:
         # Se la riduzione rumore è stata fatta ma la normalizzazione no (o non disponibile)
         # dobbiamo salvare il risultato di NR nel file di output finale.
         try:
              print(f"  Salvataggio audio solo con riduzione rumore in: {os.path.basename(output_path)}")
              sf.write(output_path, data_to_process, rate_to_process, format='FLAC')
              audio_processed_further = True
         except Exception as e_save_nr_only:
              print(f"  ERRORE durante salvataggio audio solo con NR: {e_save_nr_only}")
              return False

    # Ritorna True solo se abbiamo effettivamente scritto un file processato in output_path
    return audio_processed_further


# Funzione per caricare il modello selezionato (MODIFICATA per large-v2)
def load_model(model_choice_with_size: str):
    # Separa la scelta base dall'eventuale dimensione (es. "whisper-large-v2")
    parts = model_choice_with_size.split('-')
    model_base = parts[0] # "whisper" o "hf_whisper" o "whispy_italian"
    model_size = parts[1] if len(parts) > 1 else "medium" # Default a medium se non specificato

    # --- Blocco OpenAI Whisper ---
    if model_base == "whisper":
        # Mappa le dimensioni richieste ai nomi dei modelli OpenAI
        openai_model_name = model_size
        if model_size == "largev2": openai_model_name = "large-v2" # Nome specifico OpenAI
        elif model_size == "largev3": openai_model_name = "large-v3"
        elif model_size not in ["tiny", "base", "small", "medium", "large"]:
             print(f"Warn: Dimensione '{model_size}' non riconosciuta per OpenAI Whisper, uso 'medium'.")
             openai_model_name = "medium"

        print(f"Caricamento modello Whisper di OpenAI ({openai_model_name})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        try:
            model = whisper.load_model(openai_model_name, device=device)
            print(f"Modello OpenAI Whisper '{openai_model_name}' caricato.")
            return model
        except Exception as e:
            print(f"Errore caricamento modello OpenAI Whisper '{openai_model_name}': {e}")
            if device == "cuda":
                print("Tentativo fallback su CPU...")
                try: return whisper.load_model(openai_model_name, device="cpu")
                except Exception as e_cpu: print(f"Errore fallback CPU: {e_cpu}")
            raise

    # --- Blocco Modelli Hugging Face ---
    elif model_base in ["whispy_italian", "hf_whisper"]:
        hf_model_id = ""
        if model_base == "whispy_italian":
            # Ignora la dimensione per questo, usa sempre il modello specifico
            hf_model_id = "whispy/whisper_italian"
            if model_size != "medium": # Avvisa se è stata data una dimensione
                 print(f"Warn: La dimensione '{model_size}' è ignorata per 'whispy_italian'. Uso il modello specifico.")
        elif model_base == "hf_whisper":
            # Costruisci l'ID del modello HF base
            hf_model_suffix = model_size
            if model_size == "largev2": hf_model_suffix = "large-v2"
            elif model_size == "largev3": hf_model_suffix = "large-v3"
            elif model_size not in ["tiny", "base", "small", "medium", "large"]:
                print(f"Warn: Dimensione '{model_size}' non riconosciuta per HF Whisper, uso 'medium'.")
                hf_model_suffix = "medium"
            hf_model_id = f"openai/whisper-{hf_model_suffix}"

        print(f"Caricamento modello {hf_model_id} via Hugging Face...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipeline_dtype = torch.float16 if device != "cpu" else None
        print(f"Device: {device}, Dtype pipeline: {pipeline_dtype}")

        try:
            load_device = "cpu" if OPTIMUM_AVAILABLE else device
            model = AutoModelForSpeechSeq2Seq.from_pretrained(hf_model_id, low_cpu_mem_usage=True)
            processor = AutoProcessor.from_pretrained(hf_model_id)
            print(f"Modello '{hf_model_id}' e processore caricati.")

            if OPTIMUM_AVAILABLE:
                try: model = BetterTransformer.transform(model, keep_original_model=False); print("Modello trasformato con BetterTransformer.")
                except Exception as e_bt: print(f"Warn: Errore BetterTransformer: {e_bt}.")

            if load_device != device: model = model.to(device)
            print(f"Modello pronto su device: {device}")

        except Exception as e: print(f"Errore caricamento modello/processore HF '{hf_model_id}': {e}"); raise

        # Crea GenerationConfig (uguale per tutti i modelli HF qui)
        try:
            bos_token_id = processor.tokenizer.bos_token_id; eos_token_id = processor.tokenizer.eos_token_id
            pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_token_id
            try: decoder_start_token_id = model.config.decoder_start_token_id
            except AttributeError: print("Warn: decoder_start_token_id non trovato, uso bos_token_id."); decoder_start_token_id = bos_token_id
            if None in [bos_token_id, eos_token_id, pad_token_id, decoder_start_token_id]: raise ValueError("Token ID mancante.")
            print(f"Token IDs: BOS={bos_token_id}, EOS={eos_token_id}, PAD={pad_token_id}, DecoderStart={decoder_start_token_id}")

            forced_decoder_ids = processor.get_decoder_prompt_ids(language="it", task="transcribe")
            if not forced_decoder_ids: raise ValueError("forced_decoder_ids mancante.")
            print(f"Forced Decoder IDs (it, transcribe) ottenuti.")

            config = GenerationConfig(
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id, forced_decoder_ids=forced_decoder_ids,
                return_timestamps=True, no_timestamps_token_id=eos_token_id, max_length=448, num_beams=1
            )
            model.generation_config = config; print("GenerationConfig assegnata.")
        except Exception as e: print(f"Errore creazione GenerationConfig: {e}"); raise

        # Crea pipeline
        print("Creazione pipeline HF...")
        try:
            asr_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                                    feature_extractor=processor.feature_extractor, torch_dtype=pipeline_dtype,
                                    device=device, chunk_length_s=30, return_timestamps=True)
            print("Pipeline HF creata."); return asr_pipeline
        except Exception as e: print(f"Errore creazione pipeline HF: {e}"); raise

    else:
        raise ValueError(f"Scelta modello base non valida: '{model_base}'. Usare 'whisper', 'whispy_italian', 'hf_whisper'.")


# Funzione per trascrivere un file audio (legge da input_path, output in output_dir)
def transcribeAudioFile(input_path: str, model, model_choice: str, output_dir: str):
    """
    Trascrive un file audio (da input_path) e salva i risultati in output_dir.
    Nota: model_choice ora può includere la dimensione (es. "whisper-largev2").
    """
    global BATCH_SIZE_HF
    completedFileName = None
    inProgressFileName = None
    audio_basename = os.path.basename(input_path)
    start_time_transcription = time.time()
    final_return_path = None # Variabile per memorizzare il percorso da restituire

    try:
        os.makedirs(output_dir, exist_ok=True)
        fileName = os.path.splitext(audio_basename)[0]
        outputBaseName = os.path.join(output_dir, f'{fileName}-TranscribedAudio')
        inProgressFileName = f'{outputBaseName}-InProgress.txt'
        completedFileName = f'{outputBaseName}.txt'
        # print(f"Output previsto in: {output_dir}") # Meno verboso

        # Determina il modello base per la logica di trascrizione
        model_base = model_choice.split('-')[0]

        # --- Blocco OpenAI Whisper ---
        if model_base == "whisper":
            print(f"Transcribing {audio_basename} with OpenAI Whisper ({model_choice})...")
            fp16_enabled = torch.cuda.is_available(); print(f"  Using fp16: {fp16_enabled}")
            transcribedAudio = model.transcribe(input_path, language="it", task='transcribe', fp16=fp16_enabled, temperature=0.0)
            # Scrittura grezza nel file InProgress
            with open(inProgressFileName, 'w', encoding='utf-8') as file:
                if 'segments' in transcribedAudio and transcribedAudio['segments']:
                    for segment in transcribedAudio['segments']:
                        start = segment.get("start"); end = segment.get("end"); text = segment.get("text", "").strip()
                        start_val = start if start is not None else 0.0
                        end_val = end if end is not None else start_val
                        file.write(f'[{start_val:.2f}, {end_val:.2f}] {text}\n')
                elif 'text' in transcribedAudio:
                    full_text = transcribedAudio.get('text', '').strip(); print(f"  Warn: No 'segments'. Using full text.")
                    if full_text: file.write(f'[0.00, 0.00] {full_text}\n')
                    else: file.write("[INFO] Transcription resulted in empty text.\n")
                else: print(f"  Warn: No segments or text."); file.write("[INFO] No output data.\n")

        # --- Blocco per modelli Hugging Face ---
        elif model_base in ["whispy_italian", "hf_whisper"]:
            print(f"Transcribing {audio_basename} with {model_choice} (HF Pipeline)...")
            result = None
            try:
                 print(f"  Using batch_size={BATCH_SIZE_HF}, temperature=0.0, num_beams=1")
                 result = model( input_path, return_timestamps=True, chunk_length_s=30,
                                 batch_size=BATCH_SIZE_HF, generate_kwargs={ "temperature": 0.0, "num_beams": 1, } )
            except torch.cuda.OutOfMemoryError:
                 oom_msg = f"[ERROR] CUDA OutOfMemoryError batch_size={BATCH_SIZE_HF}. Riduci BATCH_SIZE_HF.\n"
                 print(oom_msg.strip());
                 with open(inProgressFileName, 'w', encoding='utf-8') as file: file.write(oom_msg)
            except Exception as e:
                error_message = f"[ERROR] Transcription failed: {e}\n"
                print(error_message.strip())
                with open(inProgressFileName, 'w', encoding='utf-8') as file:
                    file.write(error_message); traceback.print_exc(file=file)

            # Scrittura grezza nel file InProgress
            if result:
                with open(inProgressFileName, 'w', encoding='utf-8') as file:
                    if 'chunks' in result and isinstance(result['chunks'], list):
                        segments = result['chunks']; print(f"  Received {len(segments)} chunks.")
                        if not segments: print(f"  Warn: 0 chunks."); file.write("[INFO] 0 chunks.\n")
                        else:
                            for segment in segments:
                                timestamp = segment.get('timestamp', (None, None))
                                start_time = timestamp[0] if timestamp and timestamp[0] is not None else 0.0
                                end_time = timestamp[1] if timestamp and len(timestamp) > 1 and timestamp[1] is not None else start_time
                                text = segment.get('text', '').strip()
                                file.write(f'[{start_time:.2f}, {end_time:.2f}] {text}\n')
                    elif 'text' in result:
                        full_text = result['text'].strip(); print(f"  Warn: HF has 'text' not 'chunks'.")
                        if full_text: file.write(f'[0.00, 0.00] {full_text}\n')
                        else: print(f"  Warn: Full text empty."); file.write("[INFO] Empty text.\n")
                    else:
                        print(f"  Warn: Unexpected HF format: {result.keys() if isinstance(result, dict) else type(result)}"); file.write("[ERROR] Unexpected HF format\n")
            elif not os.path.exists(inProgressFileName): # Se errore ma file non creato
                 with open(inProgressFileName, 'w', encoding='utf-8') as f_touch: f_touch.write("[INFO] Transcription error occurred before writing data.\n")

        # --- Post-processing (Sanitizzazione Timestamp) ---
        if os.path.exists(inProgressFileName):
            try:
                should_post_process = True
                # ... (Controllo [ERROR]/[INFO] marker) ...
                with open(inProgressFileName, 'r', encoding='utf-8') as f_check:
                    first_line = f_check.readline()
                    if first_line and (first_line.startswith("[ERROR]") or first_line.startswith("[INFO]")):
                        print(f"  Skipping PP for {inProgressFileName} (marker found).")
                        should_post_process = False
                        try: # Rinomina per conservare il log
                            if os.path.exists(completedFileName): os.remove(completedFileName)
                            os.rename(inProgressFileName, completedFileName); print(f"  Info/Error file saved: {os.path.basename(completedFileName)}")
                            final_return_path = completedFileName # *** Memorizza percorso log ***
                        except OSError as e_mv: print(f"  Warn: Could not rename info/error file: {e_mv}")

                if should_post_process:
                    print(f"  Post-processing / Timestamp Sanitization {os.path.basename(inProgressFileName)}...")
                    post_process_transcription(inProgressFileName, completedFileName)
                    # Dopo il post-processing, il file finale è completedFileName
                    final_return_path = completedFileName # *** Memorizza percorso completato ***
                    try: # Rimuovi InProgress
                        if os.path.exists(completedFileName) and os.path.getsize(completedFileName) > 0:
                            os.remove(inProgressFileName); print(f"  Removed intermediate: {os.path.basename(inProgressFileName)}")
                        # else: Gestito sotto (se completed non esiste o è vuoto)
                    except OSError as e_rm: print(f"  Warn: Could not remove intermediate: {e_rm}")

            except Exception as e_pp_check:
                print(f"  Error during PP check/exec: {e_pp_check}")
                final_return_path = None # Indica fallimento PP
                # (Tentativo salvataggio .failed_pp)
                try:
                    failed_pp_name = completedFileName + ".failed_pp"
                    if os.path.exists(inProgressFileName) and not os.path.exists(failed_pp_name):
                         os.rename(inProgressFileName, failed_pp_name); print(f"  Preserved intermediate as {os.path.basename(failed_pp_name)}")
                except OSError as e_mv_fail: print(f"  Could not preserve intermediate after PP error: {e_mv_fail}")
        else:
            # Se InProgress non esiste (es. errore grave durante trascrizione)
            print(f"  Skipping PP: {os.path.basename(inProgressFileName)} not found.")
            final_return_path = None # Non c'è un file finale da restituire
            # Pulisci eventuale Completed orfano
            if os.path.exists(completedFileName): 
                try: os.remove(completedFileName); 
                except OSError: pass

        # --- Fine Post-processing ---

        # Verifica finale prima del return
        if final_return_path and not os.path.exists(final_return_path):
             print(f"  WARN: Il percorso finale restituito '{final_return_path}' non esiste. Restituisco None.")
             final_return_path = None
        elif final_return_path and os.path.getsize(final_return_path) == 0:
             print(f"  INFO: Il file finale '{os.path.basename(final_return_path)}' è vuoto.")
             # Restituiamo comunque il percorso del file vuoto

        end_time_transcription = time.time()
        print(f"  Transcription+PP completed for {audio_basename} in {end_time_transcription - start_time_transcription:.2f} seconds.")
        return final_return_path # *** RESTITUISCI IL PERCORSO MEMORIZZATO ***

    except Exception as e_outer:
        # ... (Blocco gestione eccezione esterna e scrittura file FAILED.txt INVARIATO) ...
        print(f"!!! Unhandled error in transcribeAudioFile for {input_path}: {e_outer}")
        # ... (Scrittura FAILED.txt) ...
        # Restituisci il percorso del file FAILED.txt se creato, altrimenti None
        error_log_path = os.path.join(output_dir, f"{os.path.splitext(audio_basename)[0]}-TRANSCRIPTION_FAILED.txt")
        return error_log_path if os.path.exists(error_log_path) else None


# Funzione per rimuovere ripetizioni interne a una stringa
def remove_internal_repetitions(text, min_len=5, max_lookback=15):
    """
    Tenta di rimuovere sequenze di parole ripetute all'interno di una stringa.
    Es: "ciao come stai ciao come stai ciao come" -> "ciao come stai ciao come"
    """
    words = text.split()
    if len(words) < min_len * 2: # Non abbastanza parole per avere ripetizioni significative
        return text

    processed_words = []
    i = 0
    while i < len(words):
        found_repeat = False
        # Guarda indietro per possibili ripetizioni
        # Cerca sequenze ripetute di lunghezza da `min_len` fino a `max_lookback`
        for lookback in range(min(max_lookback, i // 2, len(words) // 2), min_len - 1, -1):
             if i + lookback <= len(words): # Assicura che ci sia spazio per confrontare
                 current_segment = tuple(words[i : i + lookback])
                 # Controlla se questo segmento è uguale al segmento precedente di stessa lunghezza
                 if current_segment == tuple(words[i - lookback : i]):
                      # Trovata una ripetizione! Salta la sequenza corrente
                      # print(f"    DEBUG REPEAT: Found internal repeat of '{' '.join(current_segment)}', skipping {lookback} words at index {i}")
                      i += lookback
                      found_repeat = True
                      break # Esci dal ciclo lookback

        if not found_repeat:
            processed_words.append(words[i])
            i += 1

    return " ".join(processed_words)

# --- Funzione post_process_transcription AGGIORNATA ---
# NUOVA Funzione per valutare la ripetitività (sostituisce remove_internal_repetitions)
def is_highly_repetitive(text, threshold=0.6, min_words=10):
    """
    Valuta se una stringa di testo è eccessivamente ripetitiva.
    Calcola il rapporto tra parole uniche e parole totali.
    Ritorna True se il rapporto è sotto la soglia (troppo ripetitivo), False altrimenti.
    """
    words = text.lower().split() # Lavora su minuscolo
    if len(words) < min_words:
        return False # Non valutare testi troppo corti

    unique_words = set(words)
    repetition_ratio = len(unique_words) / len(words)

    # print(f"DEBUG REPETITIVE: Text='{text[:50]}...', Len={len(words)}, Unique={len(unique_words)}, Ratio={repetition_ratio:.2f}") # Log Debug

    return repetition_ratio < threshold

# --- Funzione post_process_transcription AGGIORNATA ---
def post_process_transcription(input_file, output_file):
    print(f"  Starting post-processing & timestamp sanitization for {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    lines = []
    try:
        if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
             print(f"  Warn: Input file for PP empty/missing."); 
             with open(output_file, 'w', encoding='utf-8'): pass; return
        with open(input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except Exception as e:
        print(f"  Errore lettura PP: {e}"); 
        with open(output_file, 'w', encoding='utf-8') as f_err: f_err.write(f"[ERROR] Read fail: {e}\n"); return

    sanitized_lines = []
    last_valid_end_time = 0.0
    MAX_SEGMENT_DURATION = 30.0 # Aggiustabile
    line_errors, line_adjusted_overlap, line_adjusted_duration, line_adjusted_endstart = 0, 0, 0, 0
    lines_written = 0
    lines_removed_repetitive = 0 # Contatore per righe scartate

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("[ERROR]") or line.startswith("[INFO]"):
            if line: sanitized_lines.append(line + '\n'); lines_written+=1
            continue

        match = re.match(r'\[\s*([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)\s*\]\s*(.*)', line)
        if not match: print(f"  Warn (L{line_num+1}): Malformed: '{line}'"); line_errors += 1; continue

        try:
            start_str, end_str, text_part_raw = match.groups()
            start_f = float(start_str); end_f = float(end_str)
            original_start, original_end = start_f, end_f; adjustment_log = ""

            # Controlli Timestamp (come prima)
            if start_f < 0 or end_f < 0: line_errors += 1; continue
            if end_f < start_f: end_f = start_f + 0.1; line_adjusted_endstart += 1; adjustment_log += f"|End<Start"
            elif end_f == start_f: end_f = start_f + 0.05; adjustment_log += f"|Start=End"
            duration = end_f - start_f
            if duration > MAX_SEGMENT_DURATION: end_f = start_f + MAX_SEGMENT_DURATION; line_adjusted_duration += 1; adjustment_log += f"|Dur>Max"
            if start_f < last_valid_end_time - 0.001:
                 adjusted_start = last_valid_end_time; line_adjusted_overlap +=1; adjustment_log += f"|Overlap"
                 start_f = adjusted_start
                 if end_f <= start_f: end_f = start_f + 0.1
            if end_f <= start_f: line_errors += 1; continue

            # --- NUOVA PULIZIA TESTO (SCARTA SE RIPETITIVO) ---
            text_part_stripped = text_part_raw.strip()

            # 1. Salta la linea se il testo è vuoto
            if not text_part_stripped:
                 # print(f"  DEBUG - Line {line_num+1} skipped (empty text).")
                 continue

            # 2. Salta la linea se è altamente ripetitiva
            if is_highly_repetitive(text_part_stripped, threshold=0.5, min_words=10): # Aggiusta threshold/min_words
                 print(f"  Warn (Line {line_num+1}): Skipping highly repetitive line: '{text_part_stripped[:80]}...'")
                 lines_removed_repetitive += 1
                 continue
            # --- Fine Pulizia Testo ---

            # Logga aggiustamenti timestamp se ci sono stati
            # if adjustment_log: print(f"  Adjust (L{line_num+1}): Orig:[{original_start:.2f},{original_end:.2f}] New:[{start_f:.2f},{end_f:.2f}] {adjustment_log}")

            processed_line = f'[{start_f:.2f}, {end_f:.2f}] {text_part_stripped}\n' # Usa testo originale (solo strippato)
            sanitized_lines.append(processed_line)
            last_valid_end_time = end_f
            lines_written += 1

        except ValueError as e: print(f"  Errore valore (L{line_num+1}): '{line}' - {e}"); line_errors += 1
        except Exception as e: print(f"  Errore generico (L{line_num+1}): '{line}' - {e}"); line_errors += 1

    # --- Rimozione Ripetizioni INTERE Righe (Minimale) ---
    final_lines = []; prev_line_text = None; removed_final_repeats = 0
    for line in sanitized_lines:
        if line.startswith("[ERROR]") or line.startswith("[INFO]"): final_lines.append(line); continue
        match = re.match(r'\[.*?\]\s*(.*)', line);
        if not match: continue
        text_part = match.group(1).strip()
        # Aggiungi controllo per non rimuovere linee vuote consecutive se le vogliamo tenere
        if text_part and text_part == prev_line_text: removed_final_repeats += 1; continue
        final_lines.append(line)
        # Aggiorna prev_line_text solo se la linea corrente non era vuota
        # per permettere la rimozione di ripetizioni tipo "A B A" -> "A B"
        if text_part:
             prev_line_text = text_part
        # Se vuoi rimuovere anche "A A" (due vuote di fila), rimuovi l'if sopra

    print(f"  Sanitization summary: Overlap={line_adjusted_overlap}, DurTrunc={line_adjusted_duration}, End<Start={line_adjusted_endstart}, Malformed/Skip={line_errors}, Lines discarded as repetitive={lines_removed_repetitive}, Final immediate repeats removed={removed_final_repeats}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f: f.writelines(final_lines)
        print(f"  Post-processing completato. {len(final_lines)} lines saved to: {os.path.basename(output_file)}")
    except Exception as e: print(f"  Errore scrittura PP: {e}") # Gestione errore scrittura invariata

# --- END OF transcriptionUtils/transcribeAudio.py ---