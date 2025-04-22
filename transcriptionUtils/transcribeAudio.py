# --- START OF transcriptionUtils/transcribeAudio.py (CORRETTO - Preprocessing Adattivo LUFS/SNR) ---
import whisper # type: ignore
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig # type: ignore
import torch # type: ignore
import sys, os, re, traceback, time
import numpy as np
from pydub import AudioSegment, exceptions as pydub_exceptions # type: ignore
try: import noisereduce as nr; NOISEREDUCE_AVAILABLE = True # type: ignore
except ImportError: NOISEREDUCE_AVAILABLE = False
try: import soundfile as sf; SOUNDFILE_AVAILABLE = True # type: ignore
except ImportError: SOUNDFILE_AVAILABLE = False; print("WARN: SoundFile non trovato!")
try: from optimum.bettertransformer import BetterTransformer; OPTIMUM_AVAILABLE = True # type: ignore
except ImportError: OPTIMUM_AVAILABLE = False
# Importa Pyloudnorm se disponibile
try: import pyloudnorm as pyln; PYLOUDNORM_AVAILABLE = True # type: ignore
except ImportError: PYLOUDNORM_AVAILABLE = False

# --- Funzione di Preprocessing (MODIFICATA per logica adattiva LUFS/SNR) ---
def preprocess_audio(input_path: str, output_path: str,
                     noise_reduce=True, normalize=True,
                     # Riceve dict metriche originali e target LUFS
                     original_metrics: dict | None = None,
                     target_norm_lufs: float = -19.0, # Target Loudness finale
                     boost_threshold_db: float = 6.0, # Soglia per decidere se boostare (rispetto a target LUFS)
                     # Parametri NR
                     nr_prop_decrease_low_snr: float = 0.90, # NR per audio rumoroso boostato
                     nr_prop_decrease_mid_snr: float = 0.80, # NR per audio medio o silenzioso boostato
                     nr_prop_decrease_high_snr: float = 0.70,# NR leggera per audio già buono
                     # Soglia SNR per decidere NR
                     low_snr_threshold_db: float = 35.0,
                     high_snr_threshold_db: float = 50.0,
                     # Limite boost iniziale e clipping finale
                     max_initial_boost_db: float = 15.0,
                     final_clip_limit_dbfs: float = -0.5):
    filename = os.path.basename(input_path); processing_done = False; apply_nr = False; nr_prop = nr_prop_decrease_normal = nr_prop_decrease_high_snr # Default NR leggera

    try:
        audio = AudioSegment.from_file(input_path)
        if len(audio) == 0: return False

        # Estrai metriche originali se disponibili
        original_lufs = None; original_snr = None
        if original_metrics:
             original_lufs = original_metrics.get('loudness_lufs')
             original_snr = original_metrics.get('original_snr') # Assicurati sia 'original_snr' nel manifest

        # --- Logica Adattiva Boost e NR ---
        gain_to_apply = 0.0
        if normalize and original_lufs is not None and np.isfinite(original_lufs):
            if original_lufs < (target_norm_lufs - boost_threshold_db):
                # Audio silenzioso -> Applica Boost Limitato
                gain_to_apply = min(target_norm_lufs - original_lufs, max_initial_boost_db)
                apply_nr = True # Applica NR dopo boost
                if original_snr is not None and np.isfinite(original_snr):
                    if original_snr < low_snr_threshold_db: # Silenzioso E Rumoroso
                        print(f"  Boost + Strong NR ({nr_prop_decrease_low_snr:.2f}) for {filename} (Loud={original_lufs:.1f} LUFS, SNR={original_snr:.1f}dB)")
                        nr_prop = nr_prop_decrease_low_snr
                    else: # Silenzioso ma Pulito/Medio
                        print(f"  Boost + Mid NR ({nr_prop_decrease_mid_snr:.2f}) for {filename} (Loud={original_lufs:.1f} LUFS, SNR={original_snr:.1f}dB)")
                        nr_prop = nr_prop_decrease_mid_snr
                else: # SNR non disponibile, usa NR media per sicurezza
                    print(f"  Boost + Mid NR ({nr_prop_decrease_mid_snr:.2f}) for {filename} (Loud={original_lufs:.1f} LUFS, SNR=N/A)")
                    nr_prop = nr_prop_decrease_mid_snr
            elif original_snr is not None and original_snr < high_snr_threshold_db:
                # Volume OK ma Rumoroso -> Applica NR leggera/standard
                print(f"  Standard NR ({nr_prop_decrease_normal:.2f}) for {filename} (Loud={original_lufs:.1f} LUFS, SNR={original_snr:.1f}dB)")
                apply_nr = True
                nr_prop = nr_prop_decrease_normal
            else:
                # Volume OK e Pulito -> Probabilmente non serve NR
                print(f"  Skipping NR for {filename} (Loud={original_lufs:.1f} LUFS, SNR={original_snr if original_snr else 'N/A'}dB)")
                apply_nr = False
        else:
            # Metriche originali non disponibili -> Applica NR standard per sicurezza? O no?
            print(f"  WARN: Original metrics missing for {filename}. Applying standard NR.")
            apply_nr = True # Applica NR standard come fallback
            nr_prop = nr_prop_decrease_normal


        # Applica Boost se calcolato
        if gain_to_apply > 0:
            try: audio = audio + gain_to_apply; processing_done = True
            except Exception as boost_err: print(f"  WARN: Failed boost: {boost_err}")

        # Converti a NumPy Float32
        if audio.channels > 1: audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        norm_factor = 1 << (audio.sample_width * 8 - 1); samples /= norm_factor
        rate = audio.frame_rate; data_to_process = samples

        # Applica Noise Reduction selezionata
        if noise_reduce and apply_nr and NOISEREDUCE_AVAILABLE:
            try:
                reduced = nr.reduce_noise(y=data_to_process, sr=rate, stationary=False, prop_decrease=nr_prop)
                if reduced is not None and len(reduced) > 0:
                    if not np.allclose(data_to_process, reduced, atol=1e-5): processing_done = True
                    data_to_process = reduced
            except Exception as e_nr: print(f"  WARN: NR failed for {filename}: {e_nr}")
        elif noise_reduce and apply_nr: print(f"  WARN: Noisereduce lib missing for {filename}.")

        # --- Normalizzazione Finale a Target LUFS (se Pyloudnorm disponibile) ---
        if normalize and PYLOUDNORM_AVAILABLE:
            try:
                meter = pyln.Meter(rate)
                current_lufs = meter.integrated_loudness(data_to_process)
                if np.isfinite(current_lufs):
                    gain_db = target_norm_lufs - current_lufs
                    # Limita guadagno eccessivo anche qui? Forse non necessario se boost iniziale è limitato.
                    # gain_db = min(gain_db, 12.0) # Limite opzionale
                    gain_factor = 10**(gain_db / 20.0)
                    normalized_data = data_to_process * gain_factor
                    if not np.allclose(data_to_process, normalized_data, atol=1e-4): processing_done = True
                    data_to_process = normalized_data
                    # print(f"  Normalized to {target_norm_lufs:.1f} LUFS (Gain={gain_db:.1f}dB)") # Verboso
            except Exception as e_lufs_norm: print(f"  WARN: LUFS normalization failed: {e_lufs_norm}")
        elif normalize: # Fallback a normalizzazione RMS se pyloudnorm manca
             rms_amplitude = np.sqrt(np.mean(np.square(data_to_process)))
             if rms_amplitude > 1e-9:
                 current_rms_dbfs = 20 * np.log10(rms_amplitude)
                 gain_needed_db = target_norm_lufs - current_rms_dbfs # Usa target LUFS come approssimazione
                 gain_factor = 10**(gain_needed_db / 20.0)
                 normalized_data = data_to_process * gain_factor
                 if not np.allclose(data_to_process, normalized_data, atol=1e-4): processing_done = True
                 data_to_process = normalized_data

        # --- Clipping di Sicurezza Finale ---
        if normalize: # Applica clipping solo se abbiamo normalizzato
             clip_limit_linear = 10**(final_clip_limit_dbfs / 20.0)
             clipped_count = np.sum(np.abs(data_to_process) > clip_limit_linear)
             if clipped_count > 0: print(f"  WARN: Clipping {clipped_count} samples for {filename}."); processing_done = True
             data_to_process = np.clip(data_to_process, -clip_limit_linear, clip_limit_linear)

        # --- Salvataggio ---
        if processing_done:
             if SOUNDFILE_AVAILABLE:
                 try: #... (Sanitizzazione NaN/Inf e sf.write) ...
                     if np.isnan(data_to_process).any() or np.isinf(data_to_process).any():
                         print(f"  WARN: NaN/Inf before saving {filename}. Sanitizing."); data_to_process = np.nan_to_num(data_to_process, nan=0.0, posinf=0.0, neginf=0.0)
                     sf.write(output_path, data_to_process, rate, format='FLAC', subtype='PCM_16'); return True
                 except Exception as e: print(f"!!! ERROR saving {output_path}: {e}")
             else: print(f"ERROR: SoundFile missing, cannot save {output_path}")
        else: return False # Non modificato, non salva
    except Exception as e_main: print(f"!!! ERROR in preprocess_audio for {filename}: {e_main}"); traceback.print_exc()
    return False


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
                return_timestamps=True, no_timestamps_token_id=eos_token_id, max_length=448
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
def transcribeAudioFile(input_path: str,
                        model,
                        model_choice: str,
                        output_dir: str,
                        # NUOVI ARGOMENTI OPZIONALI con default
                        num_beams: int = DEFAULT_NUM_BEAMS_TA,  # Usa i tuoi default qui
                        batch_size_hf: int = DEFAULT_BATCH_SIZE_HF_TA): # Usa i tuoi default qui
    """
    Trascrive un file audio (da input_path) gestisce il caso 'No segments' e salva i risultati in output_dir.
    Nota: model_choice ora può includere la dimensione (es. "whisper-largev2").
    Gestisce il caso 'No segments' e accetta parametri per i modelli HF e cerca di evitare file lock.
    """
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
            print(f"  Using batch_size={batch_size_hf},  temperature=0.0, num_beams={num_beams}")
            result = model(
                input_path,
                return_timestamps=True,
                chunk_length_s=30,
                batch_size=batch_size_hf, # Usa parametro
                generate_kwargs={
                    "temperature": 0.0,
                    "num_beams": num_beams # Usa parametro
                }
            )

        except torch.cuda.OutOfMemoryError:
             oom_msg = f"[ERROR] CUDA OutOfMemoryError batch_size={batch_size_hf}. Riduci BATCH_SIZE_HF.\n"
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
                 # --- VERIFICA ESISTENZA DOPO PP ---
                 time.sleep(0.2) # Aumentato delay post PP call
                 if os.path.exists(completedFileName):
                      #print(f"  CHECK after PP call: {os.path.basename(completedFileName)} FOUND.")
                      final_return_path = completedFileName
                      # Tenta rimozione InProgress solo se PP sembra ok
                      time.sleep(0.1)
                      try: os.remove(inProgressFileName)
                      except OSError as e_rm: print(f"  Warn: Could not remove intermediate {os.path.basename(inProgressFileName)} after PP: {e_rm}")
                 #else:
                      #print(f"  ERROR after PP call: {os.path.basename(completedFileName)} NOT FOUND.")
                      # final_return_path rimane None

            except Exception as e_pp:
                 print(f"!!! Error during post-processing execution: {e_pp}")
                 traceback.print_exc()
                 final_return_path = None # Fallimento PP
                 # ... (tentativo rename a .failed_pp come prima) ...
        elif post_processing_skipped_due_to_marker:
            # ... (Logica rename da InProgress a Completed, con time.sleep(0.2) e gestione errori come prima) ...
            # Imposta final_return_path = completedFileName SE rename ha SUCCESSO
            # altrimenti final_return_path rimane None
            print(f"  Attempting rename {os.path.basename(inProgressFileName)} -> {os.path.basename(completedFileName)} (due to skipped PP)...")
            time.sleep(0.2)                 
            try:
                if os.path.exists(completedFileName): os.remove(completedFileName)
                os.rename(inProgressFileName, completedFileName)
                print(f"  Rename successful. Info/Error file saved: {os.path.basename(completedFileName)}")
                final_return_path = completedFileName # Successo rename
            except OSError as e_mv:
                     print(f"!!! ERROR renaming info/error file {os.path.basename(inProgressFileName)}: {e_mv}")
                     # final_return_path rimane None
            except Exception as e_mv_generic:
                  print(f"!!! UNEXPECTED ERROR renaming info/error file: {e_mv_generic}")
                  # final_return_path rimane None
    else: # Se inProgressFileName non esisteva all'inizio
        print(f"  Skipping PP: {os.path.basename(inProgressFileName)} not found initially.")
        # final_return_path rimane None                  

    # --- Verifica Finale (SEMPLIFICATA E CORRETTA) ---
    final_check_msg = ""
    # Lavoriamo direttamente su final_return_path
    if final_return_path and os.path.exists(final_return_path):
        if os.path.getsize(final_return_path) == 0:
            final_check_msg = f"CHECK Final: Path '{os.path.basename(final_return_path)}' exists but is EMPTY."
            # Potremmo decidere di restituire None se è vuoto? Per ora no.
        else:
            final_check_msg = f"CHECK Final: Path '{os.path.basename(final_return_path)}' exists and is valid."
            # Il path è valido, non fare nulla a final_return_path
    elif final_return_path and not os.path.exists(final_return_path):
         # Se il path era stato impostato ma ORA non esiste (molto strano)
         final_check_msg = f"CHECK Final ERROR: Path '{os.path.basename(final_return_path)}' was set but file NOT FOUND at final check."
         final_return_path = None # Reimposta a None perché non è valido
    else: # final_return_path era già None o non esisteva
         if final_return_path is None:
             final_check_msg = f"CHECK Final: No valid output path was generated for {audio_basename} during transcription/PP."
         else: # Path impostato ma non esiste (coperto sopra, ma per sicurezza)
             final_check_msg = f"CHECK Final ERROR: Path '{os.path.basename(final_return_path)}' invalid."
             final_return_path = None

         # Tenta creazione file FAILED solo se final_return_path è effettivamente None
         if final_return_path is None and \
            not os.path.exists(completedFileName) and \
            not os.path.exists(inProgressFileName.replace(".failed_pp","")) and \
            not os.path.exists(errorLogFileName):
             try:
                 with open(errorLogFileName, 'w', encoding='utf-8') as f_err:
                     f_err.write(f"[ERROR] Transcription process failed to produce a valid output file for {audio_basename}.\n")
                     f_err.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                 print(f"  Created final error log: {os.path.basename(errorLogFileName)}")
                 final_return_path = errorLogFileName # Restituisci il log di errore creato
             except Exception as e_final_log:
                 print(f"  !!! Errore creando il file di log finale: {e_final_log}")
                 final_return_path = None # Fallback a None se anche log fallisce
    #print(f"  {final_check_msg}")
    end_time_transcription = time.time()
    elapsed = end_time_transcription - start_time_transcription
    # --- LOG FINALE PRIMA DEL RETURN ---
    returning_value = os.path.basename(final_return_path) if final_return_path else 'None'
    print(f"  --> transcribeAudioFile finished for {audio_basename} in {elapsed:.2f}s. Returning: {returning_value}")
    # --- FINE LOG FINALE ---
    return final_return_path

def _remove_repeated_token_sequences(text: str,
                                     min_seq_len: int = 2, # MINIMO 2 TOKEN
                                     min_consecutive_repeats: int = 2
                                     ) -> str:
    """
    Rimuove sequenze di token (parole, punteggiatura, parentesi) che si ripetono
    almeno `min_consecutive_repeats` volte consecutivamente. Versione 4.
    """
    if not text or min_consecutive_repeats < 2 or min_seq_len < 1:
        return text
    # Tokenizer Migliorato (v4)
    tokens = re.findall(r"\(.+?\)|\[.+?\]|[\w']+|[^\s]", text)
    if len(tokens) < min_seq_len * min_consecutive_repeats:
        return text
    made_change_in_pass = True
    while made_change_in_pass:
        made_change_in_pass = False
        i = 0
        new_tokens = []
        while i < len(tokens):
            found_long_repeat = False
            max_possible_len = (len(tokens) - i) // min_consecutive_repeats
            if max_possible_len < min_seq_len:
                 new_tokens.extend(tokens[i:])
                 break
            for L in range(max_possible_len, min_seq_len - 1, -1):
                sequence_to_match = tuple(tokens[i : i + L])
                repeat_count = 1
                k = i + L
                while k + L <= len(tokens):
                    if tuple(tokens[k : k + L]) == sequence_to_match:
                        repeat_count += 1
                        k += L
                    else:
                        break
                if repeat_count >= min_consecutive_repeats:
                    new_tokens.extend(list(sequence_to_match)) # Aggiungi prima occorrenza
                    i += repeat_count * L # Salta tutte le occorrenze
                    found_long_repeat = True
                    made_change_in_pass = True
                    break # Esci dal loop L
            if not found_long_repeat:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    # Ricostruzione Stringa (Migliorata v4)
    if not tokens: return ""
    result = " ".join(tokens)
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)
    result = re.sub(r'([(\[])\s+', r'\1', result)
    result = re.sub(r'\s+([)\]])', r'\1', result)
    result = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", result)
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
    line_errors = 0; line_adjusted_overlap = 0; line_adjusted_duration = 0; line_adjusted_endstart = 0
    lines_cleaned_count = 0 # Rinominato per chiarezza
    lines_processed_count = 0

    for line_num, line in enumerate(lines):
        lines_processed_count += 1
        line = line.strip()
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
            if start_f < 0 or end_f < 0: line_errors += 1; continue
            if end_f < start_f: end_f = start_f + 0.1; line_adjusted_endstart += 1
            elif end_f == start_f and text_part_raw.strip(): end_f = start_f + 0.05
            duration = end_f - start_f
            if duration > MAX_SEGMENT_DURATION: end_f = start_f + MAX_SEGMENT_DURATION; line_adjusted_duration += 1
            if start_f < last_valid_end_time - 0.001:
                 adjusted_start = last_valid_end_time; line_adjusted_overlap +=1
                 start_f = adjusted_start
                 if end_f <= start_f: end_f = start_f + 0.1
            if end_f <= start_f: line_errors += 1; continue

            text_part_stripped = text_part_raw.strip()
            if not text_part_stripped: continue


            # --- DEBUG: Stampa testo prima della pulizia ---
            # print(f"  DEBUG PP (L{line_num+1}) BEFORE CLEAN: '{text_part_stripped[:100]}...'")

            cleaned_text = _remove_repeated_token_sequences(
                text_part_stripped,
                min_seq_len=2,
                min_consecutive_repeats=2
            )

            # --- DEBUG: Stampa testo dopo la pulizia ---
            # print(f"  DEBUG PP (L{line_num+1}) AFTER CLEAN:  '{cleaned_text[:100]}...'")

            # Salta se testo diventa vuoto dopo pulizia
            if not cleaned_text:
                 print(f"  Info PP (L{line_num+1}): Line discarded (empty after clean). Orig: '{text_part_stripped[:80]}...'")
                 continue

            # Considera pulito solo se il testo è effettivamente diverso E non solo per spazi bianchi
            text_changed = cleaned_text.replace(" ", "") != text_part_stripped.replace(" ", "")
            if text_changed:
                lines_cleaned_count += 1
            #    print(f"  DEBUG PP (L{line_num+1}) TEXT CHANGED!") # Log esplicito se cambia

            processed_line = f'[{start_f:.2f}, {end_f:.2f}] {cleaned_text}\n'
            sanitized_lines.append(processed_line)
            last_valid_end_time = end_f

        except ValueError as e: print(f"  Errore valore PP (L{line_num+1}): '{line}' - {e}"); line_errors += 1
        except Exception as e: print(f"  Errore generico PP (L{line_num+1}): '{line}' - {e}"); line_errors += 1

    # --- Rimozione Righe Finali Duplicate (invariata) ---
    # ... (Codice rimozione righe finali duplicate) ...
    final_lines = []; prev_line_text = None; removed_final_repeats = 0
    # ... (codice rimozione duplicati finali) ...
    for line in sanitized_lines:
        # ... (logica invariata)
        if line.startswith("[ERROR]") or line.startswith("[INFO]"): final_lines.append(line); continue
        match = re.match(r'\[.*?\]\s*(.*)', line);
        if not match: continue
        text_part = match.group(1).strip()
        if text_part and text_part == prev_line_text: removed_final_repeats += 1; continue
        final_lines.append(line)
        if text_part: prev_line_text = text_part


    # --- DIAGNOSTICA SCRITTURA ---
    num_final_lines = len(final_lines)
    print(f"  PP Summary: InputLines={lines_processed_count}, SanitizedLines={len(sanitized_lines)}, FinalLinesToWrite={num_final_lines}")
    print(f"  PP Stats: OverlapAdj={line_adjusted_overlap}, DurTrunc={line_adjusted_duration}, End<StartAdj={line_adjusted_endstart}, Malformed/Skip={line_errors}, LinesCleaned={lines_cleaned_count}, FinalRepeatsRemoved={removed_final_repeats}")

    write_successful = False
    try:
        #print(f"  PP WRITE: Attempting to write {num_final_lines} lines to {os.path.basename(output_file)}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
        #print(f"  PP WRITE: writelines() completed.")
        # Verifica IMMEDIATAMENTE dopo la scrittura
        time.sleep(0.2) # Aumenta leggermente il delay post-scrittura
        if os.path.exists(output_file):
        #     print(f"  PP WRITE CHECK: File '{os.path.basename(output_file)}' EXISTS immediately after write.")
             if os.path.getsize(output_file) > 0 or num_final_lines == 0: # OK se vuoto E doveva essere vuoto
        #         print(f"  PP WRITE CHECK: File size is > 0 (or expected empty). Write seems successful.")
                 write_successful = True
        #     else:
        #         print(f"  PP WRITE CHECK ERROR: File exists BUT IS EMPTY (and {num_final_lines} lines were expected).")
        #else:
        #     print(f"  PP WRITE CHECK ERROR: File '{os.path.basename(output_file)}' DOES NOT EXIST immediately after write attempt.")

    except Exception as e_write:
         #print(f"  PP WRITE FAILED with exception: {e_write}")
         # Stampa traceback completo dell'errore di scrittura
         traceback.print_exc()

    # Stampa messaggio finale basato sul successo della scrittura
    #if write_successful:
    #     print(f"  PP Finished Successfully for: {os.path.basename(output_file)}")
    #else:
    #     print(f"  PP Finished WITH ERRORS writing: {os.path.basename(output_file)}")

# --- END OF transcriptionUtils/transcribeAudio.py ---