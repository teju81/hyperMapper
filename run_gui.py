import customtkinter as ctk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, askdirectory
import pathlib
import threading
import os
import sys
import webbrowser
import webview
import subprocess

# Your imports
from sfm_project.pycolmap_sfm.sfm_colmap import run_sfm
from sfm_project.pycolmap_sfm.utils import extract_frames, process_pointcloud
sys.path.append('gsplat_light')
from internal.entrypoints.gspl import cli as gspl_pipeline
sys.path.append('gsplat_light/utils')
from utils.sd_feature_extraction import main as extract_sd_features
from utils.sd_feature_extraction import parse_args as sd_parse_args

project_name = "alpha"
# Arguments
custom_args = [
    "fit",
    "--config", "gsplat_light/configs/spot_less_splats/gsplat-cluster.yaml",
    "--data.parser.split_mode", "reconstruction",
    "--data.path", "output/mvs",
    "-n", str(project_name),
    "--trainer.max_epochs", "50",
    "--viewer",
]

# Dark Mode Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ConsoleRedirect:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, message):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", message)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def flush(self):
        pass  # Needed for Python's internal use

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SFM + GSplat Material UI")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")
        self.minsize(1200, 700)

        self.sidebar_visible = True
        self.theme_mode = "dark"

        self.create_sidebar()
        self.create_main_area()
        self.create_embedded_viewer()
        # Console Frame
        


    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        # Collapsible toggle
        toggle_btn = ctk.CTkButton(self.sidebar, text="⬅ Collapse", command=self.toggle_sidebar)
        toggle_btn.pack(pady=(5, 10))

        # Input fields
        self.video_entry = self.create_entry("Video Path", browse=True, video=True)
        self.image_entry = self.create_entry("Image Dir", browse=True)
        self.output_entry = self.create_entry("Output Dir", browse=True)

        ctk.CTkLabel(self.sidebar, text="FPS").pack(pady=(10, 0))
        self.fps_entry = ctk.CTkEntry(self.sidebar, placeholder_text="1")
        self.fps_entry.pack()
        self.fps_entry.insert(0, "1")

        # Buttons
        self.run_button = ctk.CTkButton(self.sidebar, text="▶ Run Pipeline", command=self.run_pipeline)
        self.run_button.pack(pady=(15, 5))

        self.viewer_button = ctk.CTkButton(self.sidebar, text="Open Viewer",command=self.open_embedded_websocket_viewer)
        self.viewer_button.pack(pady=5)

        self.theme_button = ctk.CTkButton(self.sidebar, text="🌗 Toggle Theme", command=self.toggle_theme)
        self.theme_button.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(self.sidebar, width=200)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar.pack_forget()
        else:
            self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar_visible = not self.sidebar_visible

    def toggle_theme(self):
        if self.theme_mode == "dark":
            ctk.set_appearance_mode("light")
            self.theme_mode = "light"
        else:
            ctk.set_appearance_mode("dark")
            self.theme_mode = "dark"

    def create_entry(self, label_text, browse=False, video=False):
        ctk.CTkLabel(self.sidebar, text=label_text).pack(pady=(10, 0))
        entry = ctk.CTkEntry(self.sidebar, width=240)
        entry.pack()
        if browse:
            btn = ctk.CTkButton(self.sidebar, text="Browse", width=100,
                                command=lambda: self.browse_file(entry, video))
            btn.pack(pady=2)
        return entry

    def browse_file(self, entry, video=False):
        path = askopenfilename() if video else askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def create_main_area(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        title = ctk.CTkLabel(self.main_frame, text="Pipeline Console", font=("Segoe UI", 20, "bold"))
        title.pack(pady=(10, 5))


        self.console_label = ctk.CTkLabel(self.main_frame, text="Console Log")
        self.console_label.pack(anchor="w", padx=5)

        self.console = ctk.CTkTextbox(self.main_frame, height=150, wrap="word")
        self.console.pack(fill="x", padx=5)
        self.console.configure(state="disabled")

        # Redirect stdout and stderr
        sys.stdout = ConsoleRedirect(self.console)
        sys.stderr = ConsoleRedirect(self.console)

        # self.console = ctk.CTkTextbox(self.main_frame, height=200, font=("Consolas", 12))
        # self.console.pack(fill="both", expand=True, padx=10, pady=5)
        # self.console.insert("end", "Logs will appear here...\n")
        # self.console.configure(state="disabled")

    def create_embedded_viewer(self):
        self.viewer_frame = ctk.CTkFrame(self.main_frame, height=150)
        self.viewer_frame.configure(fg_color="#1e1e1e")  # dark background
        self.viewer_frame.pack(fill="both", expand=False, padx=10, pady=5)

        self.viewer_label = ctk.CTkLabel(
            self.viewer_frame,
            text="WebSocket viewer not launched yet.",
            text_color="white"
        )
        self.viewer_label.pack(pady=10)

    def log(self, message):
        self.console.configure(state="normal")
        self.console.insert("end", f"{message}\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def open_browser(self):
        self.log("Opening WebSocket Viewer...")
        self.progress_bar.set(0.95)
        self.viewer_label.configure(text="Launching WebSocket viewer...")

        # Create a launch function to be safely called by tkinter main loop
        def launch_webview():
            webview.create_window(
                "WebSocket Viewer",
                "http://localhost:8080",
                width=1280,
                height=800,
                background_color='#1e1e1e',
            )
            webview.start()

        # Schedule the launch in the Tkinter main thread
        self.after(100, launch_webview)

        self.viewer_label.configure(text="WebSocket Viewer Launched.")

    def open_embedded_websocket_viewer(self):
        self.log("Launching embedded WebSocket viewer...")

        # Force a backend (optional but safer)
        os.environ["PYWEBVIEW_GUI"] = "qt"  # Use 'gtk' on Linux if preferred

        def launch():
            webview.create_window(
                "WebSocket Viewer",
                url="http://localhost:8080",
                width=1280,
                height=800,
                background_color='#1e1e1e',
                resizable=True
            )
            webview.start()

        # pywebview must run on main thread: schedule via .after()
        self.after(100, launch)

        self.viewer_label.configure(text="WebSocket Viewer launched")

    def run_pipeline(self):
        def thread_fn():
            try:
                self.progress_bar.set(0.1)
                video_path = self.video_entry.get()
                image_dir = pathlib.Path(self.image_entry.get())
                output_path = pathlib.Path(self.output_entry.get())
                fps = int(self.fps_entry.get())

                self.log(f"Extracting frames from: {video_path}")
                extract_frames(video_path, image_dir, fps)
                self.progress_bar.set(0.3)

                self.log("Running SfM...")
                run_sfm(image_dir=image_dir, output_path=output_path, is_dense=False)
                self.progress_bar.set(0.5)

                self.log("Processing Pointcloud...")
                process_pointcloud(output_path, view_pcd=False, save_pcd=True, filter_pcd=True, is_dense=False)
                self.progress_bar.set(0.6)

                sd_args = sd_parse_args()
                sd_args.image_dir = str(output_path / "mvs/images")
                sd_args.output = str(output_path / "mvs/SD")

                if not os.path.exists(sd_args.output):
                    self.log("Extracting SD Features...")
                    extract_sd_features(args=sd_args)
                self.progress_bar.set(0.8)

                self.log("Running GSplat Pipeline...")
                if not os.path.exists("/home/doer/hyperMapper/gsplat_light/outputs/" + custom_args[8]):
                    gspl_pipeline(args=custom_args)

                self.progress_bar.set(1.0)
                self.log("✅ Pipeline Completed Successfully!")

            except Exception as e:
                self.log(f"❌ Error: {e}")
                messagebox.showerror("Error", str(e))

        threading.Thread(target=thread_fn).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
