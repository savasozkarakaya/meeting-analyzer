import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logging
import subprocess
import sys
import os
import time

# Set theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TextHandler(logging.Handler):
    """This class allows you to log to a Tkinter Text or ScrolledText widget"""
    def __init__(self, text):
        logging.Handler.__init__(self)
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            self.text.yview(tk.END)
        self.text.after(0, append)

class MeetingAnalyzerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Meeting Analyzer & Diarization")
        self.geometry("800x850")
        
        # Variables
        self.audio_path = tk.StringVar()
        self.out_dir = tk.StringVar(value="output")
        self.device = tk.StringVar(value="auto")
        self.accept_thresh = tk.DoubleVar(value=0.65)
        self.reject_thresh = tk.DoubleVar(value=0.45)
        self.min_seg = tk.DoubleVar(value=2.0)
        self.vad_presegment = tk.BooleanVar(value=False)
        self.min_speakers = tk.StringVar(value="")
        self.max_speakers = tk.StringVar(value="")
        
        # Speaker Management
        self.speakers = [] 
        
        self.create_widgets()
        self.setup_logging()
        
        # Run startup checks
        self.after(100, self.run_startup_checks)

    def run_startup_checks(self):
        from . import checks
        
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, "Running startup checks...\n")
        self.log_area.configure(state='disabled')
        
        results = []
        results.append(checks.check_ffmpeg())
        results.append(checks.check_python_version())
        results.append(checks.check_torch())
        results.append(checks.check_imports())
        results.append(checks.check_hf_token())
        
        failed = False
        for success, msg in results:
            status = "PASS" if success else "FAIL"
            level = logging.INFO if success else logging.ERROR
            logging.log(level, f"[{status}] {msg}")
            if not success:
                failed = True
                
        if failed:
            messagebox.showwarning("Startup Checks Failed", "Some environment checks failed. Check the log for details.")
        else:
            logging.info("All startup checks passed.")

    def create_widgets(self):
        # Main Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1) # Log area expands

        # 1. Header
        self.lbl_header = ctk.CTkLabel(self, text="Meeting Analyzer", font=("Roboto Medium", 24))
        self.lbl_header.grid(row=0, column=0, pady=(20, 10), padx=20, sticky="ew")

        # 2. Input Frame
        self.frame_inputs = ctk.CTkFrame(self)
        self.frame_inputs.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.frame_inputs.grid_columnconfigure(1, weight=1)

        # Audio
        ctk.CTkLabel(self.frame_inputs, text="Meeting Audio:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(self.frame_inputs, textvariable=self.audio_path).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.frame_inputs, text="Browse", command=self.browse_audio, width=80).grid(row=0, column=2, padx=10, pady=10)

        # Output
        ctk.CTkLabel(self.frame_inputs, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(self.frame_inputs, textvariable=self.out_dir).grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.frame_inputs, text="Browse", command=self.browse_out, width=80).grid(row=1, column=2, padx=10, pady=10)

        # 3. Speaker Management
        self.frame_speakers = ctk.CTkFrame(self)
        self.frame_speakers.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.frame_speakers.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.frame_speakers, text="Speaker Management", font=("Roboto Medium", 16)).grid(row=0, column=0, columnspan=3, pady=(10, 5))
        
        # Add Speaker
        self.entry_spk_name = ctk.CTkEntry(self.frame_speakers, placeholder_text="Speaker Name", width=150)
        self.entry_spk_name.grid(row=1, column=0, padx=10, pady=5)
        
        self.entry_spk_path = ctk.CTkEntry(self.frame_speakers, placeholder_text="Reference Audio Path")
        self.entry_spk_path.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(self.frame_speakers, text="Browse", command=self.browse_spk_audio, width=60).grid(row=1, column=2, padx=(0, 5), pady=5)
        ctk.CTkButton(self.frame_speakers, text="Add", command=self.add_speaker, width=60).grid(row=1, column=3, padx=10, pady=5)
        
        # Listbox (CustomTkinter doesn't have Listbox, using Textbox read-only)
        self.list_speakers = ctk.CTkTextbox(self.frame_speakers, height=80)
        self.list_speakers.grid(row=2, column=0, columnspan=4, padx=10, pady=5, sticky="ew")
        self.list_speakers.configure(state="disabled")
        
        ctk.CTkButton(self.frame_speakers, text="Clear Speakers", command=self.clear_speakers, fg_color="transparent", border_width=1).grid(row=3, column=0, columnspan=4, pady=5)

        # 4. Settings & Actions
        self.frame_settings = ctk.CTkFrame(self)
        self.frame_settings.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # Settings Grid
        ctk.CTkLabel(self.frame_settings, text="Device:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkOptionMenu(self.frame_settings, variable=self.device, values=["auto", "cpu", "cuda"]).grid(row=0, column=1, padx=10, pady=5)
        
        ctk.CTkLabel(self.frame_settings, text="Accept Thresh:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.frame_settings, textvariable=self.accept_thresh, width=60).grid(row=0, column=3, padx=10, pady=5)
        
        ctk.CTkLabel(self.frame_settings, text="Reject Thresh:").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.frame_settings, textvariable=self.reject_thresh, width=60).grid(row=1, column=3, padx=10, pady=5)
        
        ctk.CTkLabel(self.frame_settings, text="Min Seg (s):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.frame_settings, textvariable=self.min_seg, width=60).grid(row=1, column=1, padx=10, pady=5)

        ctk.CTkCheckBox(self.frame_settings, text="VAD Pre-segmentation", variable=self.vad_presegment).grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        ctk.CTkLabel(self.frame_settings, text="Min Speakers:").grid(row=2, column=2, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.frame_settings, textvariable=self.min_speakers, width=60).grid(row=2, column=3, padx=10, pady=5)
        ctk.CTkLabel(self.frame_settings, text="Max Speakers:").grid(row=3, column=2, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.frame_settings, textvariable=self.max_speakers, width=60).grid(row=3, column=3, padx=10, pady=5)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.frame_settings, mode="indeterminate")
        self.progress_bar.grid(row=4, column=0, columnspan=4, padx=10, pady=(15, 5), sticky="ew")
        self.progress_bar.set(0)

        # Run Button
        self.btn_run = ctk.CTkButton(self.frame_settings, text="RUN ANALYSIS", command=self.start_processing, font=("Roboto Medium", 14), height=40, fg_color="#2CC985", hover_color="#229A65")
        self.btn_run.grid(row=5, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # 5. Log Area
        self.log_area = ctk.CTkTextbox(self, state='disabled')
        self.log_area.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="nsew")
        
        # Open Output
        self.btn_open = ctk.CTkButton(self, text="Open Output Folder", command=self.open_output, state="disabled")
        self.btn_open.grid(row=5, column=0, padx=20, pady=(0, 20))

    def browse_audio(self):
        path = filedialog.askopenfilename(title="Select Meeting Audio")
        if path:
            self.audio_path.set(path)

    def browse_spk_audio(self):
        path = filedialog.askopenfilename(title="Select Speaker Audio")
        if path:
            self.entry_spk_path.delete(0, tk.END)
            self.entry_spk_path.insert(0, path)
            
    def browse_out(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.out_dir.set(path)
            
    def add_speaker(self):
        name = self.entry_spk_name.get().strip()
        path = self.entry_spk_path.get().strip()
        
        if not name or not path:
            messagebox.showwarning("Warning", "Please enter both name and audio path.")
            return
            
        if not os.path.exists(path):
            messagebox.showerror("Error", "Audio file does not exist.")
            return
            
        self.speakers.append({"name": name, "path": path})
        self.update_speaker_list()
        
        # Clear inputs
        self.entry_spk_name.delete(0, tk.END)
        self.entry_spk_path.delete(0, tk.END)
        
    def clear_speakers(self):
        self.speakers = []
        self.update_speaker_list()
        
    def update_speaker_list(self):
        self.list_speakers.configure(state="normal")
        self.list_speakers.delete("0.0", tk.END)
        for spk in self.speakers:
            self.list_speakers.insert(tk.END, f"• {spk['name']}: {os.path.basename(spk['path'])}\n")
        self.list_speakers.configure(state="disabled")

    def setup_logging(self):
        text_handler = TextHandler(self.log_area)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicates if re-run
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(text_handler)

    def start_processing(self):
        if not self.audio_path.get():
            messagebox.showerror("Error", "Please select meeting audio.")
            return
            
        if not self.speakers:
            if not messagebox.askyesno("Warning", "No speakers added. All speakers will be marked as UNKNOWN. Continue?"):
                return
            
        self.btn_run.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.progress_bar.start()
        
        # Run in thread
        thread = threading.Thread(target=self.run_pipeline_thread)
        thread.start()

    def _build_review_options(self, segment):
        options = ["UNKNOWN"]
        for spk in self.speakers:
            name = spk.get("name")
            if name and name not in options:
                options.append(name)
        for candidate in segment.get("candidate_speakers", []):
            name = candidate.get("name")
            if name and name not in options:
                options.append(name)
        current = segment.get("speaker")
        if current and current not in options:
            options.append(current)
        return options

    def _show_uncertain_review_modal(self, segments):
        uncertain_indices = [idx for idx, seg in enumerate(segments) if seg.get("decision") == "uncertain"]
        if not uncertain_indices:
            return segments, 0

        updated_segments = [dict(seg) for seg in segments]
        review_count = len(uncertain_indices)
        review_result = {"applied": False}

        dialog = ctk.CTkToplevel(self)
        dialog.title("Manual Review - Uncertain Segments")
        dialog.geometry("980x650")
        dialog.transient(self)
        dialog.grab_set()
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(
            dialog,
            text=f"Found {review_count} uncertain segment(s). Select a speaker or keep UNKNOWN.",
            font=("Roboto Medium", 14),
        )
        header.grid(row=0, column=0, padx=16, pady=(14, 8), sticky="w")

        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        selections = {}
        for row_idx, seg_idx in enumerate(uncertain_indices):
            seg = segments[seg_idx]
            seg_frame = ctk.CTkFrame(scroll)
            seg_frame.grid(row=row_idx, column=0, padx=6, pady=6, sticky="ew")
            seg_frame.grid_columnconfigure(0, weight=1)

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            ctk.CTkLabel(
                seg_frame,
                text=f"[{start:.2f} - {end:.2f}] current: {seg.get('speaker', 'UNKNOWN')} | score={seg.get('score', 0.0):.3f} | conf={seg.get('confidence', 0.0):.3f}",
                anchor="w",
                justify="left",
            ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")

            text_preview = (seg.get("text", "") or "").strip()
            if len(text_preview) > 180:
                text_preview = text_preview[:180] + "..."
            ctk.CTkLabel(
                seg_frame,
                text=text_preview if text_preview else "(no transcript text)",
                anchor="w",
                justify="left",
                wraplength=900,
            ).grid(row=1, column=0, padx=10, pady=4, sticky="ew")

            candidates = seg.get("candidate_speakers", [])
            candidate_str = ", ".join(
                f"{c.get('name', '?')}:{float(c.get('score', 0.0)):.3f}" for c in candidates[:3]
            ) or "N/A"
            ctk.CTkLabel(
                seg_frame,
                text=f"Top candidates: {candidate_str}",
                anchor="w",
                justify="left",
            ).grid(row=2, column=0, padx=10, pady=4, sticky="ew")

            options = self._build_review_options(seg)
            default_choice = seg.get("speaker", "UNKNOWN")
            if default_choice not in options:
                default_choice = "UNKNOWN"
            choice_var = tk.StringVar(value=default_choice)
            ctk.CTkOptionMenu(seg_frame, values=options, variable=choice_var).grid(
                row=3, column=0, padx=10, pady=(4, 10), sticky="w"
            )
            selections[seg_idx] = choice_var

        footer = ctk.CTkFrame(dialog)
        footer.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="ew")
        footer.grid_columnconfigure((0, 1), weight=1)

        def apply_review():
            changed = 0
            reviewed_at = int(time.time())
            for seg_idx, choice_var in selections.items():
                choice = (choice_var.get() or "UNKNOWN").strip() or "UNKNOWN"
                seg_copy = dict(updated_segments[seg_idx])
                previous_speaker = seg_copy.get("speaker", "UNKNOWN")
                seg_copy["speaker"] = choice
                if choice == "UNKNOWN":
                    seg_copy["decision"] = "reject"
                    seg_copy["decision_reason"] = "manual_review_marked_unknown"
                else:
                    seg_copy["decision"] = "accept"
                    seg_copy["decision_reason"] = "manual_review_assigned_speaker"
                seg_copy["manual_review"] = {
                    "was_uncertain": True,
                    "previous_speaker": previous_speaker,
                    "resolved_speaker": choice,
                    "resolved_at_unix": reviewed_at,
                }
                if previous_speaker != choice or seg_copy.get("decision") != "uncertain":
                    changed += 1
                updated_segments[seg_idx] = seg_copy

            review_result["applied"] = True
            review_result["changed"] = changed
            dialog.destroy()

        def skip_review():
            review_result["applied"] = False
            dialog.destroy()

        ctk.CTkButton(
            footer,
            text="Keep Existing Decisions",
            command=skip_review,
            fg_color="transparent",
            border_width=1,
        ).grid(row=0, column=0, padx=(8, 4), pady=8, sticky="ew")
        ctk.CTkButton(
            footer,
            text="Apply Manual Review",
            command=apply_review,
            fg_color="#2CC985",
            hover_color="#229A65",
        ).grid(row=0, column=1, padx=(4, 8), pady=8, sticky="ew")

        dialog.wait_window()
        if review_result.get("applied"):
            return updated_segments, int(review_result.get("changed", 0))
        return segments, 0

    def _review_uncertain_segments(self, processed_segments):
        pending = [s for s in processed_segments if s.get("decision") == "uncertain"]
        if not pending:
            return processed_segments, 0

        done = threading.Event()
        payload = {"segments": processed_segments, "changed": 0}

        def show_modal():
            try:
                segments_after, changed = self._show_uncertain_review_modal(processed_segments)
                payload["segments"] = segments_after
                payload["changed"] = changed
            finally:
                done.set()

        self.after(0, show_modal)
        done.wait()
        return payload["segments"], int(payload["changed"])

    def _persist_outputs(self, segments, out_dir):
        from . import io

        io.write_segments(segments, os.path.join(out_dir, "segments.json"))
        io.write_transcript(segments, os.path.join(out_dir, "speaker_attributed_transcript.txt"))
        io.write_word_speaker_attribution_json(
            segments, os.path.join(out_dir, "word_speaker_attribution.json")
        )
        io.write_word_speaker_attribution_txt(
            segments, os.path.join(out_dir, "word_speaker_attribution.txt")
        )

        summary_path = os.path.join(out_dir, "summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write("Manual review was applied in GUI for uncertain segments.\n")

    def run_pipeline_thread(self):
        try:
            min_speakers = int(self.min_speakers.get()) if self.min_speakers.get().strip() else None
            max_speakers = int(self.max_speakers.get()) if self.max_speakers.get().strip() else None
            from . import pipeline
            processed_segments = pipeline.run_pipeline(
                audio_path=self.audio_path.get(),
                references=self.speakers,
                out_dir=self.out_dir.get(),
                device=self.device.get(),
                lang="tr",
                accept_threshold=self.accept_thresh.get(),
                reject_threshold=self.reject_thresh.get(),
                min_segment_sec=self.min_seg.get(),
                vad_presegment=self.vad_presegment.get(),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            reviewed_segments, changed_count = self._review_uncertain_segments(processed_segments)
            if changed_count > 0:
                self._persist_outputs(reviewed_segments, self.out_dir.get())
                logging.info("Manual review applied for %s uncertain segment(s).", changed_count)

            self.after(0, lambda: messagebox.showinfo("Success", "Processing completed successfully!"))
            self.after(0, lambda: self.btn_open.configure(state="normal"))
        except Exception as e:
            err_msg = str(e)
            logging.error(f"Error: {err_msg}")
            self.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {err_msg}"))
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal"))
            self.after(0, self.progress_bar.stop)

    def open_output(self):
        path = self.out_dir.get()
        if not path or not os.path.isdir(path):
            messagebox.showerror("Error", f"Output folder not found: {path}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.check_call(["open", path])
            else:
                subprocess.check_call(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open output folder:\n{e}")

def main():
    app = MeetingAnalyzerGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
