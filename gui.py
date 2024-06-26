import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog
import detect


class DetectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detect GUI")

        self.file_path = tk.StringVar()
        self.save_correct_csv = tk.BooleanVar()
        self.save_incorrect_csv = tk.BooleanVar()
        self.save_cropped_jpg = tk.BooleanVar()
        self.save_frames_jpg = tk.BooleanVar()

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="File/Stream Path:").grid(row=0, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=10, pady=10)

        tk.Checkbutton(self.root, text="Save Correct Recognitions to CSV", variable=self.save_correct_csv).grid(row=1,
                                                                                                                columnspan=3,
                                                                                                                padx=10,
                                                                                                                pady=10)
        tk.Checkbutton(self.root, text="Save Incorrect Recognitions to CSV", variable=self.save_incorrect_csv).grid(
            row=2, columnspan=3, padx=10, pady=10)
        tk.Checkbutton(self.root, text="Save Cropped Plates to JPG", variable=self.save_cropped_jpg).grid(row=3,
                                                                                                          columnspan=3,
                                                                                                          padx=10,
                                                                                                          pady=10)
        tk.Checkbutton(self.root, text="Save Frames with Detections to JPG", variable=self.save_frames_jpg).grid(row=4,
                                                                                                                 columnspan=3,
                                                                                                                 padx=10,
                                                                                                                 pady=10)

        tk.Button(self.root, text="Run Detect", command=self.run_detect).grid(row=5, columnspan=3, padx=10, pady=20)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi"), ("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)

    def run_detect(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file or enter a stream path.")
            return

        options = {
            'save_correct_csv': self.save_correct_csv.get(),
            'save_incorrect_csv': self.save_incorrect_csv.get(),
            'save_cropped_jpg': self.save_cropped_jpg.get(),
            'save_frames_jpg': self.save_frames_jpg.get(),
        }

        try:
            #detect.run(file_path, options)
            detect.run(file_path)
            messagebox.showinfo("Success", "Detection process completed.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectApp(root)
    root.mainloop()
