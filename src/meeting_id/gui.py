import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logging
import sys
import os
from . import pipeline

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

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.frame_settings, mode="indeterminate")
        self.progress_bar.grid(row=2, column=0, columnspan=4, padx=10, pady=(15, 5), sticky="ew")
        self.progress_bar.set(0)

        # Run Button
        self.btn_run = ctk.CTkButton(self.frame_settings, text="RUN ANALYSIS", command=self.start_processing, font=("Roboto Medium", 14), height=40, fg_color="#2CC985", hover_color="#229A65")
        self.btn_run.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

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

    def run_pipeline_thread(self):
        try:
            pipeline.run_pipeline(
                audio_path=self.audio_path.get(),
                references=self.speakers,
                out_dir=self.out_dir.get(),
                device=self.device.get(),
                lang="tr",
                accept_threshold=self.accept_thresh.get(),
                reject_threshold=self.reject_thresh.get(),
                min_segment_sec=self.min_seg.get()
            )
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
        if os.path.exists(path):
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])

def main():
    app = MeetingAnalyzerGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
