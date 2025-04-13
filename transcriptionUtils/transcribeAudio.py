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
    Gestisce il caso 'No segments' e cerca di evitare file lock.
    """
    global BATCH_SIZE_HF
    completedFileName = None
    inProgressFileName = None
    audio_basename = os.path.basename(input_path)
    start_time_transcription = time.time()
    final_return_path = None # Variabile per memorizzare il percorso da restituire
    
    # --- Setup Nomi File ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        fileName = os.path.splitext(audio_basename)[0]
        outputBaseName = os.path.join(output_dir, f'{fileName}-TranscribedAudio')
        inProgressFileName = f'{outputBaseName}-InProgress.txt'
        completedFileName = f'{outputBaseName}.txt'
        errorLogFileName = f'{outputBaseName}-TRANSCRIPTION_FAILED.txt' # Per errori gravi

        # Pulisci file vecchi se esistono (potrebbe aiutare con lock, ma usare con cautela)
        # try:
        #     if os.path.exists(inProgressFileName): os.remove(inProgressFileName)
        #     if os.path.exists(completedFileName): os.remove(completedFileName)
        #     if os.path.exists(errorLogFileName): os.remove(errorLogFileName)
        # except OSError as e_clean:
        #     print(f"  Warn: Impossibile pulire file preesistenti per {fileName}: {e_clean}")
    except Exception as e_setup:
         print(f"!!! Errore grave nel setup iniziale per {input_path}: {e_setup}")
         return None # Non possiamo procedere
        # Determina il modello base per la logica di trascrizione
    model_base = model_choice.split('-')[0]

    # --- Blocco Trascrizione (OpenAI Whisper) ---
    if model_base == "whisper":
        transcription_data = None
        try:
            print(f"Transcribing {audio_basename} with OpenAI Whisper ({model_choice})...")
            fp16_enabled = torch.cuda.is_available(); print(f"  Using fp16: {fp16_enabled}")
            # Esegui trascrizione
            transcription_data = model.transcribe(input_path, language="it", task='transcribe', fp16=fp16_enabled, temperature=0.0)

            # Scrivi nel file InProgress SUBITO DOPO la trascrizione
            # Usa 'with open' per assicurare chiusura file
            no_segments_case = False
            with open(inProgressFileName, 'w', encoding='utf-8') as file:
                if transcription_data and 'segments' in transcription_data and transcription_data['segments']:
                    for segment in transcription_data['segments']:
                        start = segment.get("start"); end = segment.get("end"); text = segment.get("text", "").strip()
                        start_val = start if start is not None else 0.0
                        end_val = end if end is not None else start_val
                        file.write(f'[{start_val:.2f}, {end_val:.2f}] {text}\n')
                elif transcription_data and 'text' in transcription_data:
                     full_text = transcription_data.get('text', '').strip()
                     print(f"  Warn: No 'segments'. Using full text.")
                     if full_text:
                         # Scrivi la riga speciale MA SENZA marker per ora
                         # Il post-processing la gestirà (o la scarterà se vuota)
                         file.write(f'[0.00, 0.00] {full_text}\n')
                         no_segments_case = True # Segna che siamo in questo caso
                     else:
                         # Se anche il testo completo è vuoto, scrivi INFO
                         print(f"  Warn: No segments and text is empty.")
                         file.write("[INFO] Transcription resulted in empty text.\n")
                else:
                     print(f"  Warn: Transcription data missing segments and text.")
                     file.write("[INFO] No output data from transcription.\n")

        except Exception as e_transcribe:
            error_message = f"[ERROR] OpenAI Whisper transcription failed: {e_transcribe}\n"
            print(error_message.strip())
            # Scrivi l'errore nel file InProgress (assicurati che sia chiuso)
            try:
                with open(inProgressFileName, 'w', encoding='utf-8') as file:
                    file.write(error_message); traceback.print_exc(file=file)
            except Exception as e_write_err:
                 print(f"  !!! Errore scrivendo log di errore su {inProgressFileName}: {e_write_err}")
            # Non procedere oltre se la trascrizione fallisce qui
            # Rinomina InProgress in FAILED se possibile? O lascia InProgress?
            # Per ora, restituiamo None, il loop principale lo marcherà come fallito.
            return None

    # --- Blocco Trascrizione (Hugging Face) ---
    elif model_base in ["whispy_italian", "hf_whisper"]:
        result = None
        try:
            print(f"Transcribing {audio_basename} with {model_choice} (HF Pipeline)...")
            # ... (chiamata al modello HF, gestione OOM come prima) ...
            print(f"  Using batch_size={BATCH_SIZE_HF}, temperature=0.0, num_beams=1")
            result = model( input_path, return_timestamps=True, chunk_length_s=30,
                            batch_size=BATCH_SIZE_HF, generate_kwargs={ "temperature": 0.0, "num_beams": 1, } )

        except torch.cuda.OutOfMemoryError:
             oom_msg = f"[ERROR] CUDA OutOfMemoryError batch_size={BATCH_SIZE_HF}. Riduci BATCH_SIZE_HF.\n"
             print(oom_msg.strip());
             try: # Scrivi errore e chiudi file
                 with open(inProgressFileName, 'w', encoding='utf-8') as file: file.write(oom_msg)
             except Exception as e_write_err: print(f"  !!! Errore scrivendo log OOM: {e_write_err}")
             return None # Fallimento
        except Exception as e:
            error_message = f"[ERROR] HF Transcription failed: {e}\n"
            print(error_message.strip())
            try: # Scrivi errore e chiudi file
                with open(inProgressFileName, 'w', encoding='utf-8') as file:
                    file.write(error_message); traceback.print_exc(file=file)
            except Exception as e_write_err: print(f"  !!! Errore scrivendo log HF error: {e_write_err}")
            return None # Fallimento

        # Scrittura risultato HF nel file InProgress
        try:
            with open(inProgressFileName, 'w', encoding='utf-8') as file:
                if result and 'chunks' in result and isinstance(result['chunks'], list):
                    # ... (logica scrittura chunks invariata) ...
                    segments = result['chunks']; print(f"  Received {len(segments)} chunks.")
                    if not segments: file.write("[INFO] 0 chunks.\n")
                    else:
                        for segment in segments:
                            timestamp = segment.get('timestamp', (None, None))
                            start_time = timestamp[0] if timestamp and timestamp[0] is not None else 0.0
                            end_time = timestamp[1] if timestamp and len(timestamp) > 1 and timestamp[1] is not None else start_time
                            text = segment.get('text', '').strip()
                            file.write(f'[{start_time:.2f}, {end_time:.2f}] {text}\n')
                elif result and 'text' in result:
                    full_text = result['text'].strip(); print(f"  Warn: HF has 'text' not 'chunks'.")
                    if full_text: file.write(f'[0.00, 0.00] {full_text}\n'); no_segments_case = True
                    else: file.write("[INFO] Empty text.\n")
                elif result: # Altro formato inatteso
                    print(f"  Warn: Unexpected HF format: {result.keys() if isinstance(result, dict) else type(result)}"); file.write("[ERROR] Unexpected HF format\n")
                else: # Nessun risultato o errore non catturato sopra
                     file.write("[INFO] No result from HF transcription.\n")
        except Exception as e_write_hf:
             print(f"!!! Errore scrittura risultato HF su {inProgressFileName}: {e_write_hf}")
             # Se la scrittura fallisce, non possiamo procedere
             return None

    # --- Post-processing (Sanitizzazione Timestamp) ---
    # Ora questa sezione viene eseguita SOLO se inProgressFileName esiste
    # e non ha fallito gravemente prima.
    post_processing_skipped_due_to_marker = False
    post_processing_executed = False

    if os.path.exists(inProgressFileName):
        # Controlla marker [ERROR] / [INFO] PRIMA di aprire per PP
        try:
            with open(inProgressFileName, 'r', encoding='utf-8') as f_check:
                first_line = f_check.readline()
                if first_line and (first_line.startswith("[ERROR]") or first_line.startswith("[INFO]")):
                    print(f"  Skipping PP for {os.path.basename(inProgressFileName)} (marker found).")
                    post_processing_skipped_due_to_marker = True
        except Exception as e_check:
             print(f"  Warn: Errore leggendo {inProgressFileName} per controllo marker: {e_check}")

        # Esegui PP solo se non skippato
        if not post_processing_skipped_due_to_marker:
            print(f"  Starting post-processing / Timestamp Sanitization...")
            try:
                 post_process_transcription(inProgressFileName, completedFileName) # Chiama PP
                 post_processing_executed = True
                 print(f"  Post-processing finished. Output expected: {os.path.basename(completedFileName)}") # Log modificato

                 # --- INIZIO BLOCCO CONTROLLO FILE POST-PP ---
                 # Attendiamo un istante prima di controllare l'esistenza del file
                 time.sleep(0.1) # Breve pausa per il filesystem
                 if os.path.exists(completedFileName):
                      final_return_path = completedFileName
                      # Rimuovi InProgress solo se PP ha successo e output esiste
                      time.sleep(0.1) # Altro piccolo delay
                      try:
                           os.remove(inProgressFileName)
                      except OSError as e_rm:
                           print(f"  Warn: Could not remove intermediate {os.path.basename(inProgressFileName)} after PP: {e_rm}")
                 else:
                      # Questo è il caso del log "Warn: Post-processing ran but did not create..."
                      print(f"  ERROR after PP: {os.path.basename(completedFileName)} not found after waiting!")
                      final_return_path = None # Fallimento PP
                 # --- FINE BLOCCO CONTROLLO FILE POST-PP ---

            except Exception as e_pp:
                 print(f"!!! Error during post-processing execution: {e_pp}")
                 traceback.print_exc()
                 final_return_path = None # Fallimento PP
                 # ... (tentativo rename a .failed_pp come prima) ...
                 try:
                     failed_pp_name = completedFileName + ".failed_pp"
                     if os.path.exists(inProgressFileName) and not os.path.exists(failed_pp_name):
                          os.rename(inProgressFileName, failed_pp_name)
                          print(f"  Preserved intermediate as {os.path.basename(failed_pp_name)}")
                 except OSError as e_mv_fail:
                      print(f"  Could not preserve intermediate after PP error: {e_mv_fail}")


        # Se il PP è stato skippato a causa di un marker
        elif post_processing_skipped_due_to_marker:
             print(f"  Renaming {os.path.basename(inProgressFileName)} to {os.path.basename(completedFileName)} (due to skipped PP)...")
             time.sleep(0.2) # Delay più lungo
             try:
                 if os.path.exists(completedFileName): os.remove(completedFileName)
                 os.rename(inProgressFileName, completedFileName)
                 print(f"  Info/Error file saved: {os.path.basename(completedFileName)}")
                 final_return_path = completedFileName
             except OSError as e_mv:
                  print(f"!!! ERROR renaming info/error file {os.path.basename(inProgressFileName)}: {e_mv}")
                  final_return_path = None
             except Exception as e_mv_generic:
                   print(f"!!! UNEXPECTED ERROR renaming info/error file: {e_mv_generic}")
                   final_return_path = None

    else: # Se inProgressFileName non esisteva all'inizio
        print(f"  Skipping PP: {os.path.basename(inProgressFileName)} not found initially.")
        final_return_path = None

    # --- Verifica Finale (con log migliorato) ---
    final_check_msg = ""
    final_path_to_return = final_return_path # Copia temporanea
    if final_path_to_return and os.path.exists(final_path_to_return):
        if os.path.getsize(final_path_to_return) == 0:
            final_check_msg = f"INFO Final Check: File finale '{os.path.basename(final_path_to_return)}' esiste ma è VUOTO."
        else:
            final_check_msg = f"INFO Final Check: File finale '{os.path.basename(final_path_to_return)}' valido trovato."
    elif final_path_to_return and not os.path.exists(final_path_to_return):
         final_check_msg = f"ERROR Final Check: Percorso finale '{os.path.basename(final_path_to_return)}' indicato ma NON ESISTE."
         final_path_to_return = None # Assicura che sia None
    else: # final_path_to_return era già None
         final_check_msg = f"INFO Final Check: Nessun percorso finale valido generato per {audio_basename}."
         # Tenta creazione file FAILED
         if not os.path.exists(completedFileName) and not os.path.exists(inProgressFileName) and not os.path.exists(errorLogFileName):
             try:
                 with open(errorLogFileName, 'w', encoding='utf-8') as f_err:
                     f_err.write(f"[ERROR] Transcription process failed to produce a valid output file for {audio_basename}.\n")
                     f_err.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                 print(f"  Created final error log: {os.path.basename(errorLogFileName)}")
                 final_path_to_return = errorLogFileName # Restituisci il log di errore
             except Exception as e_final_log:
                 print(f"  !!! Errore creando il file di log finale: {e_final_log}")

    print(f"  {final_check_msg}")
    end_time_transcription = time.time()
    elapsed = end_time_transcription - start_time_transcription
    print(f"  Transcription/PP process for {audio_basename} finished in {elapsed:.2f}s. Returning path: {os.path.basename(final_path_to_return) if final_path_to_return else 'None'}")
    return final_path_to_return

def _remove_repeated_token_sequences(text: str,
                                     min_seq_len: int = 3, # Aumentiamo un po' la soglia minima? O teniamo 2? Proviamo 3.
                                     min_consecutive_repeats: int = 2 # Rimuovi se si ripete almeno 2 volte (lascia 1)
                                     ) -> str:
    """
    Rimuove sequenze di token (parole e punteggiatura) che si ripetono
    almeno `min_consecutive_repeats` volte consecutivamente.
    Lascia solo la prima occorrenza della sequenza.
    Ottimizzato per gestire lunghe ripetizioni (es. allucinazioni ASR).

    Args:
        text: La stringa di input.
        min_seq_len: Lunghezza minima in token della sequenza da considerare.
        min_consecutive_repeats: Numero minimo di occorrenze consecutive
                                  per attivare la rimozione (>= 2).

    Returns:
        La stringa con le sequenze ripetute rimosse.
    """
    if not text or min_consecutive_repeats < 2:
        return text

    # Tokenizza preservando parole e punteggiatura
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)
    if len(tokens) < min_seq_len * min_consecutive_repeats:
        return text # Non abbastanza token per il pattern minimo

    made_change_in_pass = True # Flag per continuare a ciclare finché si trovano ripetizioni
    while made_change_in_pass:
        made_change_in_pass = False
        i = 0
        new_tokens = [] # Costruiamo la lista pulita in ogni passata

        while i < len(tokens):
            found_long_repeat = False
            # Itera sulle possibili lunghezze, dalla più lunga alla più corta
            # La lunghezza massima possibile è limitata dalla posizione corrente 'i'
            # e dal numero minimo di ripetizioni necessarie.
            max_possible_len = (len(tokens) - i) // min_consecutive_repeats
            if max_possible_len < min_seq_len:
                 # Non c'è più spazio per trovare sequenze ripetute abbastanza lunghe
                 # Aggiungi i token rimanenti e esci dal loop interno
                 new_tokens.extend(tokens[i:])
                 break

            for L in range(max_possible_len, min_seq_len - 1, -1):
                # La sequenza base da cercare
                sequence_to_match = tuple(tokens[i : i + L])
                # Controlla quante volte si ripete consecutivamente
                repeat_count = 1
                k = i + L # Indice di inizio della potenziale prossima ripetizione
                while k + L <= len(tokens):
                    if tuple(tokens[k : k + L]) == sequence_to_match:
                        repeat_count += 1
                        k += L # Passa alla prossima potenziale ripetizione
                    else:
                        break # La sequenza non si ripete più

                # Se abbiamo trovato abbastanza ripetizioni consecutive
                if repeat_count >= min_consecutive_repeats:
                    # print(f"  DEBUG REPEAT v3: Found sequence {sequence_to_match} repeating {repeat_count} times starting at {i}")
                    # Aggiungi solo la *prima* occorrenza a new_tokens
                    new_tokens.extend(list(sequence_to_match))
                    # Avanza l'indice 'i' per saltare *tutte* le occorrenze trovate
                    i += repeat_count * L
                    found_long_repeat = True
                    made_change_in_pass = True # Segnala che abbiamo modificato
                    break # Esci dal loop delle lunghezze L, abbiamo gestito la ripetizione per questa 'i'

            # Se NON abbiamo trovato una ripetizione di lunghezza >= min_seq_len
            # che si ripete abbastanza volte partendo da 'i',
            # aggiungi semplicemente il token corrente e vai al prossimo.
            if not found_long_repeat:
                new_tokens.append(tokens[i])
                i += 1

        # Aggiorna la lista dei token per il prossimo ciclo while esterno
        tokens = new_tokens
        # Se non sono state fatte modifiche in questa passata, esci dal loop esterno
        # (Il flag made_change_in_pass verrà resettato a False all'inizio della prossima iterazione)

    # --- Ricostruzione della stringa (logica invariata) ---
    if not tokens:
        return ""
    result = tokens[0]
    for j in range(1, len(tokens)):
        prev_token = tokens[j-1]
        current_token = tokens[j]
        if current_token in ".,!?;:": result += current_token
        elif re.match(r"[\w']+", current_token) and not prev_token.endswith("'"): result += " " + current_token
        else: result += current_token
    return result.strip()


# --- Funzione post_process_transcription AGGIORNATA ---
def post_process_transcription(input_file, output_file):
    # ... (Codice lettura input file con 'with open' invariato) ...
    print(f"  Starting post-processing & timestamp sanitization for {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    lines = []
    try:
        # ... (codice lettura file invariato) ...
        if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
             print(f"  Warn: Input file for PP empty/missing: {os.path.basename(input_file)}.");
             with open(output_file, 'w', encoding='utf-8') as f_out: pass
             return
        with open(input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except Exception as e:
        # ... (gestione errore lettura invariato) ...
        print(f"  Errore lettura PP input {os.path.basename(input_file)}: {e}");
        try:
            with open(output_file, 'w', encoding='utf-8') as f_err: f_err.write(f"[ERROR] PP Read fail: {e}\n")
        except: pass
        return

    sanitized_lines = []
    last_valid_end_time = 0.0
    MAX_SEGMENT_DURATION = 30.0
    line_errors, line_adjusted_overlap, line_adjusted_duration, line_adjusted_endstart = 0, 0, 0, 0
    lines_cleaned_repetitive = 0

    for line_num, line in enumerate(lines):
        line = line.strip()
        # ... (salta marker [INFO]/[ERROR]) ...
        if not line or line.startswith("[ERROR]") or line.startswith("[INFO]"):
            if line: sanitized_lines.append(line + '\n')
            continue

        # ... (match regex timestamp) ...
        match = re.match(r'\[\s*([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)\s*\]\s*(.*)', line)
        if not match: print(f"  Warn PP (L{line_num+1}): Malformed: '{line}'"); line_errors += 1; continue

        try:
            start_str, end_str, text_part_raw = match.groups()
            start_f = float(start_str); end_f = float(end_str)
            # ... (Sanitizzazione Timestamp invariata) ...
            original_start, original_end = start_f, end_f; adjustment_log = ""
            if start_f < 0 or end_f < 0: line_errors += 1; adjustment_log += f"|NegativeTS"; continue
            if end_f < start_f: end_f = start_f + 0.1; line_adjusted_endstart += 1; adjustment_log += f"|End<Start"
            elif end_f == start_f and text_part_raw.strip(): end_f = start_f + 0.05; adjustment_log += f"|Start=End"
            duration = end_f - start_f
            if duration > MAX_SEGMENT_DURATION: end_f = start_f + MAX_SEGMENT_DURATION; line_adjusted_duration += 1; adjustment_log += f"|Dur>Max"
            if start_f < last_valid_end_time - 0.001:
                 adjusted_start = last_valid_end_time; line_adjusted_overlap +=1; adjustment_log += f"|Overlap"
                 start_f = adjusted_start
                 if end_f <= start_f: end_f = start_f + 0.1
            if end_f <= start_f: line_errors += 1; adjustment_log += f"|FinalEnd<=Start"; continue


            # --- PULIZIA TESTO CON NUOVA FUNZIONE v3 ---
            text_part_stripped = text_part_raw.strip()
            if not text_part_stripped: continue # Salta se testo originale è vuoto

            # Chiama la nuova funzione v3
            # min_seq_len=3: cerca ripetizioni di almeno 3 token (es. "è il cibo", non solo "un po'")
            # min_consecutive_repeats=2: rimuovi se si ripete 2 o più volte (lascia la prima)
            cleaned_text = _remove_repeated_token_sequences(
                text_part_stripped,
                min_seq_len=3,
                min_consecutive_repeats=2
            )

            # Salta se testo diventa vuoto dopo pulizia
            if not cleaned_text:
                 print(f"  Info PP (Line {line_num+1}): Line discarded (empty after repetition removal v3). Orig: '{text_part_stripped[:80]}...'")
                 continue

            if cleaned_text != text_part_stripped:
                lines_cleaned_repetitive += 1
                # print(f"  Cleaned Repeat v3 (L{line_num+1}): '{text_part_stripped[:80]}...' -> '{cleaned_text[:80]}...'") # Debug opzionale

            # --- Fine Pulizia Testo ---

            processed_line = f'[{start_f:.2f}, {end_f:.2f}] {cleaned_text}\n'
            sanitized_lines.append(processed_line)
            last_valid_end_time = end_f

        # ... (Gestione eccezioni nel try invariata) ...
        except ValueError as e: print(f"  Errore valore PP (L{line_num+1}): '{line}' - {e}"); line_errors += 1
        except Exception as e: print(f"  Errore generico PP (L{line_num+1}): '{line}' - {e}"); line_errors += 1

    # --- Rimozione Righe Finali Duplicate (invariata) ---
    # ... (Codice rimozione righe finali duplicate) ...
    final_lines = []; prev_line_text = None; removed_final_repeats = 0
    # ...

    # --- Scrittura Output (invariata) ---
    # ... (Stampa summary e scrittura file con 'with open') ...
    print(f"  Post-processing summary: OverlapAdj={line_adjusted_overlap}, DurTrunc={line_adjusted_duration}, End<StartAdj={line_adjusted_endstart}, Malformed/Skip={line_errors}, Lines Cleaned Internally={lines_cleaned_repetitive}, Final Repeat Lines Removed={removed_final_repeats}")
    try:
        # ... (scrittura con with open) ...
        pass
    except Exception as e: print(f"  Errore scrittura PP: {e}")

# --- END OF transcriptionUtils/transcribeAudio.py ---