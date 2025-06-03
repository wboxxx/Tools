import re # For Brint

# --- START: Brint a_debug_flags definition (copied from menuBD2.py for standalone testing) ---
DEBUG_FLAGS = {
    "BRINT": True,
    "TRANSCRIBE": True, # Enable this to see repetition detection logs
    "SCREENSHOT": False,
    "NAV": False,
    "NOTE": False,
    "AUDIO": False
}

def Brint(*args, **kwargs):
    if not args:
        return
    first_arg = str(args[0])
    tags = re.findall(r"\[(.*?)\]", first_arg)

    print_message = DEBUG_FLAGS.get("BRINT", False)

    if not print_message and tags:
        for tag_str in tags:
            keywords = tag_str.upper().split()
            if any(DEBUG_FLAGS.get(kw, False) for kw in keywords):
                print_message = True
                break

    if print_message:
        print(*args, **kwargs)
# --- END: Brint a_debug_flags definition ---

# --- Repetition Detection Logic (Copied from CORRECTED menuBD2.py's transcribe_file) ---
recent_words_history = []
N_HISTORY_LENGTH = 10  # Max length of the word history
M_SEQUENCE_LENGTH = 3  # Length of sequence to check for repetition

emitted_words = [] # This will store words that are not filtered
# --- State for Pruning Tag (simplified for this test, as we don't write to UI) ---
end_time_of_last_good_word = None
pruning_occurred_since_last_good_word = False
# --- End State for Pruning Tag ---


# insert_word now needs start_time and end_time
def insert_word(word_text, start_time, end_time, conf):
    global recent_words_history, emitted_words, end_time_of_last_good_word, pruning_occurred_since_last_good_word

    # --- Corrected Repetition Check START ---
    word_should_be_skipped = False
    if M_SEQUENCE_LENGTH > 0:
        current_potential_sequence = None
        if M_SEQUENCE_LENGTH == 1:
            current_potential_sequence = [word_text]
        elif len(recent_words_history) >= M_SEQUENCE_LENGTH - 1:
            current_potential_sequence = recent_words_history[-(M_SEQUENCE_LENGTH - 1):] + [word_text]

        if current_potential_sequence:
            comparison_target_sequence_in_history = None
            if M_SEQUENCE_LENGTH == 1:
                if len(recent_words_history) > 0:
                    comparison_target_sequence_in_history = [recent_words_history[-1]]
            else:
                required_history_len = (2 * M_SEQUENCE_LENGTH) - 1
                if len(recent_words_history) >= required_history_len:
                    start_idx_comparison = len(recent_words_history) - (M_SEQUENCE_LENGTH - 1) - M_SEQUENCE_LENGTH
                    end_idx_comparison = start_idx_comparison + M_SEQUENCE_LENGTH
                    comparison_target_sequence_in_history = recent_words_history[start_idx_comparison : end_idx_comparison]

            if comparison_target_sequence_in_history and current_potential_sequence == comparison_target_sequence_in_history:
                Brint(f"[TRANSCRIBE] [REPETITION DETECTED] Sequence '{current_potential_sequence}' matches earlier sequence '{comparison_target_sequence_in_history}'. Skipping word '{word_text}'.")
                word_should_be_skipped = True
    # --- Corrected Repetition Check END ---

    if word_should_be_skipped:
        pruning_occurred_since_last_good_word = True
        return # Skip insertion

    # Word is NOT skipped
    if pruning_occurred_since_last_good_word:
        if end_time_of_last_good_word is not None:
            duration_pruned = start_time - end_time_of_last_good_word
            if duration_pruned > 0.01:
                prune_tag_text = f"[ {duration_pruned:.2f}s pruned ]"
                emitted_words.append(prune_tag_text)
        pruning_occurred_since_last_good_word = False

    emitted_words.append(word_text)

    end_time_of_last_good_word = end_time

    recent_words_history.append(word_text)
    if len(recent_words_history) > N_HISTORY_LENGTH:
        recent_words_history.pop(0)
# --- End of Repetition Detection Logic ---

def run_test():
    global recent_words_history, emitted_words, end_time_of_last_good_word, pruning_occurred_since_last_good_word
    recent_words_history = []
    emitted_words = []
    end_time_of_last_good_word = None
    pruning_occurred_since_last_good_word = False

    simulated_whisper_output = [
        ("alpha", 0.1, 0.4, 0.9), ("bravo", 0.5, 0.8, 0.9), ("charlie", 1.0, 1.3, 0.9),
        ("alpha", 1.5, 1.8, 0.9), ("bravo", 2.0, 2.3, 0.9), ("charlie", 2.5, 2.8, 0.9), # 2nd C (index 5) will be skipped
        ("alpha", 3.0, 3.3, 0.9), ("bravo", 3.5, 3.8, 0.9), ("charlie", 4.0, 4.3, 0.9),
        ("delta", 4.5, 4.8, 0.9),
        ("echo", 5.0, 5.3, 0.9), ("foxtrot", 5.5, 5.8, 0.9), ("golf", 6.0, 6.3, 0.9),
        ("echo", 6.5, 6.8, 0.9), ("foxtrot", 7.0, 7.3, 0.9), ("golf", 7.5, 7.8, 0.9),   # 2nd G (index 14) will be skipped
        ("hotel", 8.0, 8.3, 0.9),
        ("alpha", 8.5, 8.8, 0.9),
        ("alpha", 9.0, 9.3, 0.9),
        ("alpha", 9.5, 9.8, 0.9),
        ("november", 10.0, 10.3, 0.9), ("oscar", 10.5, 10.8, 0.9), ("papa", 11.0, 11.3, 0.9),
        ("november", 11.5, 11.8, 0.9), ("oscar", 12.0, 12.3, 0.9), ("papa", 12.5, 12.8, 0.9), # 2nd P (index 23) will be skipped
        ("quebec", 13.0, 13.3, 0.9)
    ]

    Brint("[TEST RUNNER] Processing simulated Whisper output with CORRECTED logic...")
    for word_data in simulated_whisper_output:
        insert_word(word_data[0], word_data[1], word_data[2], word_data[3])

    if pruning_occurred_since_last_good_word and end_time_of_last_good_word is not None:
        # This case is unlikely to be hit with current test data if last word is not skipped
        # but if it were, we'd need a T2. For now, this test focuses on explicit skips.
        pass

    Brint("\n[TEST RUNNER] --- Test Execution Finished ---")
    Brint(f"[TEST RUNNER] Emitted words (raw): {emitted_words}")

    emitted_words_content_only_from_test = [w for w in emitted_words if not w.startswith("[")]
    Brint(f"[TEST RUNNER] Emitted words (content only): {emitted_words_content_only_from_test}")
    Brint(f"[TEST RUNNER] Word history: {recent_words_history}")

    # Expected output based on the actual behavior of the CORRECTED logic:
    # - First A B C is emitted.
    # - Second A B are emitted. Second C is skipped (forms A B C, matches previous A B C). Prune tag.
    # - Third A B C are all emitted (history was [A,B,C,A,B], then C skipped, then A,B,C added, so no immediate prior A B C for 3rd A B C to match against its specific comparison target)
    # Similar logic for E F G and N O P.
    expected_emitted_words_content_only = [
        "alpha", "bravo", "charlie",  #1st ABC
        "alpha", "bravo",            #2nd AB (C skipped)
        "alpha", "bravo", "charlie",  #3rd ABC
        "delta",
        "echo", "foxtrot", "golf",    #1st EFG
        "echo", "foxtrot",            #2nd EF (G skipped)
        "hotel",
        "alpha", "alpha", "alpha",    #A A A sequence
        "november", "oscar", "papa",  #1st NOP
        "november", "oscar",          #2nd NO (P skipped)
        "quebec"
    ]

    # Check prune tag count
    expected_prune_tags_count = 3
    actual_prune_tags_count = sum(1 for w in emitted_words if "pruned" in w)

    if emitted_words_content_only_from_test == expected_emitted_words_content_only:
        Brint("\n[TEST RUNNER] ✅✅✅ Word Content Test Passed with CORRECTED logic! ✅✅✅")
        if actual_prune_tags_count == expected_prune_tags_count:
            Brint(f"[TEST RUNNER] ✅ Correct number of prune tags found: {actual_prune_tags_count}")
            print("\nTest Result: PASSED")
        else:
            Brint(f"[TEST RUNNER] ❌ Incorrect number of prune tags. Expected: {expected_prune_tags_count}, Got: {actual_prune_tags_count}")
            print("\nTest Result: FAILED (Prune Tag Count Mismatch)")
    else:
        Brint("\n[TEST RUNNER] ❌❌❌ Word Content Test Failed with CORRECTED logic! ❌❌❌")
        Brint(f"[TEST RUNNER] Expected Content: {expected_emitted_words_content_only}")
        print("\nTest Result: FAILED (Word Content Mismatch)")

if __name__ == "__main__":
    run_test()
