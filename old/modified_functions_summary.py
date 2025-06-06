# ========== MODIFIED FUNCTIONS (diff size descending) ==========


def __diff____init__():
    # [MODIFIED] __init__ — diff size: 272
    # --- original
    # +++ modified
    # @@ -234,7 +234,9 @@
    #          self.freeze_interpolation = False

    #          self.last_seek_time = 0

    #  

    # -

    # +        # Variables for new checkboxes

    # +        self.show_waveform_rms = tk.BooleanVar(value=False)

    # +        self.show_time_ticks = tk.BooleanVar(value=False)

    #  

    #          

    #          # === DEMARRAGE ===

    # @@ -269,6 +271,26 @@
    #          self.result_box.config(state='disabled')

    #  

    #          # === CONTROLS PRINCIPAUX ===

    # +

    # +        # New Checkbuttons for Waveform RMS and Time Ticks

    # +        self.waveform_rms_checkbox = tk.Checkbutton(

    # +            self.controls_top,

    # +            text="Waveform RMS",

    # +            variable=self.show_waveform_rms,

    # +            command=self._toggle_waveform_rms

    # +        )

    # +        self.waveform_rms_checkbox.pack(side=LEFT, padx=5)

    # +        ToolTip(self.waveform_rms_checkbox, "Show/Hide Waveform RMS")

    # +

    # +        self.time_ticks_checkbox = tk.Checkbutton(

    # +            self.controls_top,

    # +            text="Time Ticks",

    # +            variable=self.show_time_ticks,

    # +            command=self._toggle_time_ticks

    # +        )

    # +        self.time_ticks_checkbox.pack(side=LEFT, padx=5)

    # +        ToolTip(self.time_ticks_checkbox, "Show/Hide Time Ticks")

    # +

    #  

    #          # === MENUS ET CONTRÔLES BOTTOM ===

    #          self.loop_length_var = tk.IntVar(value=2)  # 2 mesures par défaut

    return  # end of diff block


def __diff__Brint():
    # [MODIFIED] Brint — diff size: 106
    # --- original
    # +++ modified
    # @@ -6,12 +6,8 @@
    #      tags = re.findall(r"\[(.*?)\]", first_arg)

    #  

    #      # 🔒 Mode silencieux global : BRINT = False désactive TOUT

    # -    if DEBUG_FLAGS.get("BRINT", None) is None:

    # +    if DEBUG_FLAGS.get("BRINT", None) is False:

    #          return

    # -

    # -    # 🔒 Mode silencieux global : BRINT = False désactive TOUT

    # -    if DEBUG_FLAGS.get("BRINT", None) is False:

    # -        pass

    #  

    #      # 💥 Mode super-debug : BRINT = True affiche tout

    #      if DEBUG_FLAGS.get("BRINT", None) is True:

    return  # end of diff block


def __diff__jump_playhead():
    # [MODIFIED] jump_playhead — diff size: 50
    # --- original
    # +++ modified
    # @@ -1,52 +1,30 @@
    #      def jump_playhead(self, direction, level):

    #          assert direction in (-1, 1), "Direction must be +1 or -1"

    # -

    # -        original_level = level

    # -        override_reason = ""

    # -

    # -        if not self.is_loop_effectively_defined():

    # -            # 🔁 Aucun loop actif → override durées en SECONDES

    # -            override_seconds = {

    # -                "beat": 10,

    # -                "8th": 60,

    # -                "16th": 300,

    # -                "64th": 600

    # -            }

    # -            seconds = override_seconds.get(level, None)

    # -            if seconds is not None:

    # -                delta_ms = int(seconds * 1000)

    # -                override_reason = f"[NO LOOP] override {original_level} → {seconds:.3f}s"

    # -                Brint(f"[JUMP] Aucun loop actif → {original_level} remplacé par {seconds:.3f}s ({delta_ms} ms)")

    # -            else:

    # -                delta_ms = self.get_jump_duration_ms(level)

    # -        else:

    # -            delta_ms = self.get_jump_duration_ms(level)

    # -

    # +        

    # +        delta_ms = self.get_jump_duration_ms(level)

    # +        

    #          mode = self.edit_mode.get() if hasattr(self, "edit_mode") else None

    #  

    #          if mode == "loop_start" and self.loop_start is not None:

    #              current_ms = self.loop_start

    # -            Brint("[JUMP] Mode édition : loop_start")

    #          elif mode == "loop_end" and self.loop_end is not None:

    #              current_ms = self.loop_end

    # -            Brint("[JUMP] Mode édition : loop_end")

    #          else:

    #              current_ms = int(self.playhead_time * 1000)

    # -            Brint(f"[JUMP] Mode normal depuis {current_ms} ms")

    #  

    # -        target_ms = current_ms + direction * delta_ms

    # +        target_ms = current_ms + direction * int(delta_ms)

    #          target_ms = self.snap_time_to_grid(target_ms, level)

    #          target_ms = max(0, target_ms)

    #  

    # +        # 🎯 Mise à jour en respectant le mode

    #          if mode == "loop_start":

    #              self.record_loop_marker("loop_start", milliseconds=target_ms, auto_exit=False)

    #          elif mode == "loop_end":

    #              self.record_loop_marker("loop_end", milliseconds=target_ms, auto_exit=False)

    #          else:

    # -            self.safe_jump_to_time(target_ms, source="jump_playhead")

    # -            self.safe_update_playhead(target_ms, source="jump_playhead")

    # +            # self.jump_to_time(target_ms)

    # +            self.safe_jump_to_time(int(target_ms), source="jump_playhead")

    # +            self.safe_update_playhead(int(target_ms), source="jump_playhead")

    #  

    # -        Brint(f"➡️ Jump {original_level} {direction:+} → {target_ms} ms (delta : {delta_ms} ms) {override_reason}")

    # -

    # -

    # -

    # +        # ✅ Ajout du delta réel affiché (utile pour vérifier la réversibilité immédiate)

    # +        Brint(f"➡️ Jump {level} {direction:+} → {target_ms} ms (delta demandé : {int(delta_ms)} ms)")

    return  # end of diff block


def __diff__snap_time_to_grid():
    # [MODIFIED] snap_time_to_grid — diff size: 2
    # --- original
    # +++ modified
    # @@ -5,4 +5,5 @@
    #          delta = self.get_jump_duration_ms(level)

    #          snapped = round(time_ms / delta) * delta

    #          return int(snapped)

    # -        

    # +

    # +

    return  # end of diff block


def __diff__replay_from_A():
    # [MODIFIED] replay_from_A — diff size: 2
    # --- original
    # +++ modified
    # @@ -7,5 +7,3 @@
    #              self.last_loop_jump_time = time.perf_counter()

    #              Brint("[PH LOOPJUMP] 🔁 last_loop_jump_time resynchronisé après retour à A via R")

    #  

    # -

    # -    

    return  # end of diff block


def __diff__get_rhythm_levels():
    # [MODIFIED] get_rhythm_levels — diff size: 1
    # --- original
    # +++ modified
    # @@ -17,7 +17,7 @@
    #              "64th": base_beat / (24 if rhythm_type == "ternary" else 16)

    #  

    #          }

    # -        Brint(f"[SCORE jumps] RHYTHMe jump → BPM={bpm:.2f} | bar={levels['bar']} | beat={levels['beat']}")

    # +        Brint(f"[DEBUG] RHYTHMe jump → BPM={bpm:.2f} | bar={levels['bar']} | beat={levels['beat']}")

    #  

    #  

    #          return levels

    return  # end of diff block


def __diff__clear_loop():
    # [MODIFIED] clear_loop — diff size: 1
    # --- original
    # +++ modified
    # @@ -1,5 +1,5 @@
    #      def clear_loop(self, _=None):

    # -        Brint("[CLEAR]Clear Loop")

    # +        print("Clear Loop")

    #          self.cached_canvas_width = self.grid_canvas.winfo_width()

    #          if hasattr(self.current_loop, "chords"):

    #              self.current_loop.chords = []

    return  # end of diff block

# ========== ADDED FUNCTIONS ==========

# [ADDED] request_timeline_redraw
    def request_timeline_redraw(self):
        """Placeholder function to be called when timeline needs redrawing."""
        Brint("[UI REQ] Request timeline redraw")
        # This will be implemented later to trigger actual redraw
        self.draw_rhythm_grid_canvas() # Example call, might need more specific redraw logic



# [ADDED] _toggle_time_ticks
    def _toggle_time_ticks(self):
        self.show_time_ticks.set(not self.show_time_ticks.get()) # This line is redundant
        Brint(f"[UI TOGGLE] Time Ticks set to: {self.show_time_ticks.get()}")
        self.request_timeline_redraw()

    


# [ADDED] _toggle_waveform_rms
    def _toggle_waveform_rms(self):
        self.show_waveform_rms.set(not self.show_waveform_rms.get()) # This line is redundant due to tk.BooleanVar auto-updating
        Brint(f"[UI TOGGLE] Waveform RMS set to: {self.show_waveform_rms.get()}")
        self.request_timeline_redraw()



