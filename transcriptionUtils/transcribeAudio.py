# --- IN transcriptionUtils/transcribeAudio.py ---

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
def post_process_transcription(input_file, output_file):
    print(f"  Starting post-processing & timestamp sanitization for {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    lines = []
    try:
        if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
             print(f"  Warn: Input file for PP empty/missing."); with open(output_file, 'w', encoding='utf-8'): pass; return
        with open(input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except Exception as e:
        print(f"  Errore lettura PP: {e}"); with open(output_file, 'w', encoding='utf-8') as f_err: f_err.write(f"[ERROR] Read fail: {e}\n"); return

    sanitized_lines = []
    last_valid_end_time = 0.0
    MAX_SEGMENT_DURATION = 30.0
    line_errors, line_adjusted_overlap, line_adjusted_duration, line_adjusted_endstart = 0, 0, 0, 0
    lines_written = 0
    lines_removed_repeat = 0

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("[ERROR]") or line.startswith("[INFO]"):
            if line: sanitized_lines.append(line + '\n'); lines_written+=1 # Mantieni errori/info
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

            # --- NUOVA PULIZIA TESTO ---
            text_part_stripped = text_part_raw.strip()
            # 1. Rimuovi ripetizioni interne alla riga
            text_part_cleaned = remove_internal_repetitions(text_part_stripped, min_len=3, max_lookback=10) # Aggiusta min_len/max_lookback se necessario
            if len(text_part_cleaned) < len(text_part_stripped):
                 lines_removed_repeat +=1 # Conta come linea con ripetizioni rimosse
                 # print(f"    DEBUG REPEAT - Line {line_num+1} cleaned: '{text_part_cleaned}' (was: '{text_part_stripped}')")

            # 2. Se il testo pulito è vuoto, salta la linea
            if not text_part_cleaned.strip():
                 # print(f"  DEBUG - Line {line_num+1} skipped (empty after cleaning repeats).")
                 continue

            # --- Fine Pulizia Testo ---

            processed_line = f'[{start_f:.2f}, {end_f:.2f}] {text_part_cleaned}\n' # Usa testo pulito
            sanitized_lines.append(processed_line)
            last_valid_end_time = end_f
            lines_written += 1

        except ValueError as e: print(f"  Errore valore (L{line_num+1}): '{line}' - {e}"); line_errors += 1
        except Exception as e: print(f"  Errore generico (L{line_num+1}): '{line}' - {e}"); line_errors += 1

    # --- Pulizia Finale Minima (RIPETIZIONI *INTERE* RIGHE - dovrebbe fare poco ora) ---
    final_lines = []; prev_line_text = None; removed_final_repeats = 0
    for line in sanitized_lines:
        if line.startswith("[ERROR]") or line.startswith("[INFO]"): final_lines.append(line); continue
        match = re.match(r'\[.*?\]\s*(.*)', line);
        if not match: continue
        text_part = match.group(1).strip()
        if text_part and text_part == prev_line_text: removed_final_repeats += 1; continue
        final_lines.append(line)
        if text_part: prev_line_text = text_part

    print(f"  Sanitization summary: Overlap={line_adjusted_overlap}, DurTrunc={line_adjusted_duration}, End<Start={line_adjusted_endstart}, Malformed/Skip={line_errors}, Lines with internal repeats reduced={lines_removed_repeat}, Final immediate line repeats removed={removed_final_repeats}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f: f.writelines(final_lines)
        print(f"  Post-processing completato. {len(final_lines)} lines saved to: {os.path.basename(output_file)}")
    except Exception as e: print(f"  Errore scrittura PP: {e}") # Gestione errore scrittura invariata

# --- END OF transcriptionUtils/transcribeAudio.py ---