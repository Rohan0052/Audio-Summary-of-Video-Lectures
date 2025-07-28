import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
from Main_project import process_video
import shutil
import os

class VideoSummaryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Summary Generator")
        self.root.geometry("600x400")
        self.root.configure(bg="#1e1e2f")
        self.root.resizable(False, False)

        self.video_path = None
        self.output_path = None

        self.build_ui()

    def build_ui(self):
        header = tk.Label(self.root, text="Video to Audio Summary Converter", font=("Segoe UI", 18, "bold"), bg="#1e1e2f", fg="white")
        header.pack(pady=20)

        self.video_label = tk.Label(self.root, text="No video selected", fg="#bbbbbb", bg="#1e1e2f", font=("Segoe UI", 10))
        self.video_label.pack(pady=5)
        tk.Button(self.root, text="Browse Video", font=("Segoe UI", 10, "bold"), bg="#3a80f2", fg="white",
                  relief="flat", padx=10, pady=5, command=self.select_video).pack()

        self.output_label = tk.Label(self.root, text="No output location selected", fg="#bbbbbb", bg="#1e1e2f", font=("Segoe UI", 10))
        self.output_label.pack(pady=10)
        tk.Button(self.root, text="Save As", font=("Segoe UI", 10, "bold"), bg="#34c759", fg="white",
                  relief="flat", padx=10, pady=5, command=self.select_output_path).pack()

        self.start_button = tk.Button(self.root, text="Generate Summary", font=("Segoe UI", 12, "bold"),
                                      bg="#ff9f0a", fg="white", relief="flat", padx=15, pady=8, command=self.start_processing)
        self.start_button.pack(pady=25)

        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, mode='indeterminate', length=300)
        self.progress.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
        self.progress.lower()  

        self.status_label = tk.Label(self.root, text="", fg="white", bg="#1e1e2f", font=("Segoe UI", 10))
        self.status_label.pack(pady=10)

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")])
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=f"{os.path.basename(file_path)}", fg="white")

    def select_output_path(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Output Audio As",
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3")]
        )
        if file_path:
            self.output_path = file_path
            self.output_label.config(text=f"{os.path.basename(file_path)}", fg="white")

    def start_processing(self):
        if not self.video_path:
            messagebox.showwarning("Missing Video", "Please select a video file.")
            return

        self.start_button.config(state="disabled")
        self.status_label.config(text="Processing... Please wait...")
        self.progress.lift()
        self.progress.start()

        threading.Thread(target=self.run_pipeline).start()

    def run_pipeline(self):
        success = process_video(self.video_path)

        generated_mp3 = "summary.mp3"
        if success and self.output_path:
            try:
                shutil.move(generated_mp3, self.output_path)
                self.status_label.config(text="Summary generated and saved successfully!", fg="#00ff00")
            except Exception as e:
                self.status_label.config(text=f"Error saving file: {str(e)}", fg="#ff4f4f")
        elif success:
            self.status_label.config(text="Summary generated. (Saved as: summary.mp3)", fg="#00ff00")
        else:
            self.status_label.config(text="Processing failed. See console for details.", fg="#ff4f4f")

        self.progress.stop()
        self.progress.lower()
        self.start_button.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use('default')
    style.configure("TProgressbar", troughcolor='#2c2c3e', background='#ff9f0a', thickness=6)

    app = VideoSummaryApp(root)
    root.mainloop()
