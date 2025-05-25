import customtkinter as ctk
from tkinter import messagebox, font
from tkinter.filedialog import askopenfilename, askdirectory
import pathlib
import threading
import os
import sys
import webbrowser
import subprocess
import yaml
import logging
import argparse
from typing import List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SFM_GUI')

# Your imports
from sfm_project.pycolmap_sfm.sfm_colmap import run_sfm
from sfm_project.pycolmap_sfm.utils import extract_frames, process_pointcloud
sys.path.append('gsplat_light')
from internal.entrypoints.gspl import cli as gspl_pipeline
sys.path.append('gsplat_light/utils')
from utils.sd_feature_extraction import main as extract_sd_features
from utils.sd_feature_extraction import parse_args as sd_parse_args

# ANSI Color codes for console
class ConsoleColors:
    HEADER = '\033[95m'
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Add default GSplat settings if not present
        if 'gspl' not in config:
            config['gspl'] = {
                'args': [
                    "fit",
                    "--config", "gsplat_light/configs/spot_less_splats/gsplat-cluster.yaml",
                    "--data.parser.split_mode", "reconstruction",
                    "--data.path", "output/mvs",
                    "-n", "spotless_watchtower",
                    "--trainer.max_epochs", "1",
                    "--viewer"
                ]
            }
            
        # Load GSplat pipeline args if present
        config['gspl_args'] = config['gspl']['args']
        return config
        
    except FileNotFoundError:
        return {
            'video': {'input_path': '', 'fps': 1},
            'directories': {
                'image_dir': 'output/images',
                'output_dir': 'output',
                'mvs_dir': 'output/mvs',
                'sd_features_dir': 'output/mvs/SD'
            },
            'flags': {
                'is_dense': False,
                'view_pointcloud': False,
                'save_pointcloud': True,
                'filter_pointcloud': True
            },
            'gspl': {
                'args': [
                    "fit",
                    "--config", "gsplat_light/configs/spot_less_splats/gsplat-cluster.yaml",
                    "--data.parser.split_mode", "reconstruction",
                    "--data.path", "output/mvs",
                    "-n", "spotless_watchtower",
                    "--trainer.max_epochs", "1",
                    "--viewer"
                ]
            }
        }

# Load default configuration
default_config = load_config()

# Update project name to match config.yaml
project_name = "spotless_watchtower"

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

def create_gsplat_args(config: dict) -> List[str]:
    """Create GSplat arguments in the correct format"""
    return [
        "fit",
        "--config", "gsplat_light/configs/spot_less_splats/gsplat-cluster.yaml",
        "--data.parser.split_mode", "reconstruction",
        "--data.path", "output/mvs",
        "-n", str(project_name),
        "--trainer.max_epochs", str(config['max_epochs']),
        "--viewer"
    ]

def create_gsplat_config(config: dict) -> dict:
    """Create GSplat configuration dictionary"""
    return {
        "model": {
            "gaussian": {
                "class_path": "internal.models.vanilla_gaussian.VanillaGaussian",
                "init_args": {
                    "num_points": config['num_pts'],
                    "sh_degree": config['sh_degree'],
                    "lambda_dssim": config['lambda_dssim']
                }
            }
        },
        "trainer": {
            "max_epochs": config['max_epochs'],
            "batch_size": config['batch_size'],
        },
        "optimizer": {
            "lr": config['learning_rate']
        }
    }

class ConsoleRedirect:
    def __init__(self, textbox, tag=None):
        self.textbox = textbox
        self.tag = tag
        self.terminal = sys.stdout if tag == "stdout" else sys.stderr
        
        # Get the underlying Tkinter Text widget
        self._text_widget = self.textbox._textbox
        
        # Configure text tags for different message types
        self._text_widget.tag_configure("INFO", foreground="#3498db", font=("Consolas", 12))
        self._text_widget.tag_configure("ERROR", foreground="#e74c3c", font=("Consolas", 12, "bold"))
        self._text_widget.tag_configure("SUCCESS", foreground="#2ecc71", font=("Consolas", 12))
        self._text_widget.tag_configure("WARNING", foreground="#f1c40f", font=("Consolas", 12))
        self._text_widget.tag_configure("HEADER", foreground="#9b59b6", font=("Consolas", 14, "bold"))
        self._text_widget.tag_configure("TIMESTAMP", foreground="#95a5a6", font=("Consolas", 10))

    def write(self, message):
        if message.strip():  # Only process non-empty messages
            if "gspl" in message.lower() or "gaussian" in message.lower():
                # Write GSplat messages to terminal
                self.terminal.write(message)
            else:
                # Format the message with timestamp and appropriate styling
                self.textbox.configure(state="normal")
                
                # Add timestamp
                timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "
                self._text_widget.insert("end", timestamp, "TIMESTAMP")
                
                # Determine message type and apply appropriate tag
                if "error" in message.lower():
                    self._text_widget.insert("end", message + "\n", "ERROR")
                elif "success" in message.lower() or "completed" in message.lower():
                    self._text_widget.insert("end", message + "\n", "SUCCESS")
                elif "warning" in message.lower():
                    self._text_widget.insert("end", message + "\n", "WARNING")
                elif "running" in message.lower() or "starting" in message.lower():
                    self._text_widget.insert("end", message + "\n", "HEADER")
                else:
                    self._text_widget.insert("end", message + "\n", "INFO")
                
                self._text_widget.see("end")
                self.textbox.configure(state="disabled")

    def flush(self):
        pass

class ViewerManager:
    def __init__(self):
        self._is_running = False
        self.viewer_check_attempts = 0
        self.max_check_attempts = 30  # 30 seconds timeout

    def check_viewer_ready(self, parent_window):
        """Check if the viser viewer is ready by attempting to connect"""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Try to connect to the viser server
            result = sock.connect_ex(('localhost', 8080))
            if result == 0:
                logger.info("Viser viewer is ready")
                sock.close()
                self.viewer_check_attempts = 0
                self.launch_viewer()
                return
        except Exception:
            pass
        finally:
            sock.close()

        self.viewer_check_attempts += 1
        if self.viewer_check_attempts < self.max_check_attempts:
            # Check again in 1 second
            parent_window.after(1000, lambda: self.check_viewer_ready(parent_window))
        else:
            # Timeout reached
            self.viewer_check_attempts = 0
            logger.error("Timeout waiting for viser viewer")
            messagebox.showerror("Error", "Timeout waiting for viser viewer to start. Please try again.")

    def launch_viewer(self):
        """Launch the viser viewer in the default web browser"""
        try:
            logger.info("Opening viser viewer at http://localhost:8080")
            webbrowser.open("http://localhost:8080")
            self._is_running = True
        except Exception as e:
            logger.error(f"Failed to open viewer in browser: {str(e)}")
            self._is_running = False

    def start_viewer_process(self, model_path):
        """Start the viser viewer process"""
        try:
            # Kill any existing processes on port 8080
            try:
                subprocess.run(['fuser', '-k', '8080/tcp'], stderr=subprocess.DEVNULL)
                import time
                time.sleep(1)  # Give time for the port to be released
            except Exception:
                pass

            # Start the viewer process
            viewer_script = os.path.join("gsplat_light", "viewer.py")
            if not os.path.exists(viewer_script):
                logger.error(f"Viewer script not found at {viewer_script}")
                return False

            # Run the viewer with the model path
            subprocess.Popen([
                sys.executable,
                viewer_script,
                model_path
            ])
            
            return True
        except Exception as e:
            logger.error(f"Failed to start viewer process: {str(e)}")
            return False

    def open_viewer(self, parent_window, model_path):
        """Start checking for viser viewer availability"""
        if self.start_viewer_process(model_path):
            self.viewer_check_attempts = 0
            self.check_viewer_ready(parent_window)
        else:
            messagebox.showerror("Error", "Failed to start viewer process")

    def is_running(self):
        """Check if viewer is running"""
        return self._is_running

    def cleanup(self):
        """Clean up resources"""
        self._is_running = False
        # Kill the viewer process
        try:
            subprocess.run(['fuser', '-k', '8080/tcp'], stderr=subprocess.DEVNULL)
        except Exception:
            pass

# Create a global viewer instance
viewer = ViewerManager()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SFM + GSplat Material UI")
        
        # Register cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize viewer
        self.viewer = viewer
        
        # Rest of the initialization
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.minsize(1200, 800)

        self.sidebar_visible = True
        self.theme_mode = "dark"
        self.config = default_config

        self.create_sidebar()
        self.create_main_area()
        self.create_embedded_viewer()

    def on_closing(self):
        """Handle window closing"""
        self.viewer.cleanup()
        self.quit()

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=15, pady=15)

        # Create a notebook (tabbed interface)
        self.tab_view = ctk.CTkTabview(self.sidebar)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add tabs
        self.tab_view.add("Main")
        self.tab_view.add("GSplat")
        
        # Main Tab
        self.create_main_tab(self.tab_view.tab("Main"))
        
        # GSplat Tab
        self.create_gsplat_tab(self.tab_view.tab("GSplat"))
        
        # Progress bar at the bottom
        self.progress_bar = ctk.CTkProgressBar(self.sidebar, width=250)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

    def create_main_tab(self, parent):
        # Title
        title = ctk.CTkLabel(parent, text="Configuration", font=("Segoe UI", 20, "bold"))
        title.pack(pady=(10, 20))

        # Input fields with default values
        self.video_entry = self.create_entry(
            parent,
            "Video Path", 
            default_value=self.config['video']['input_path'],
            browse=True, 
            video=True
        )
        self.image_entry = self.create_entry(
            parent,
            "Image Dir", 
            default_value=self.config['directories']['image_dir'],
            browse=True
        )
        self.output_entry = self.create_entry(
            parent,
            "Output Dir", 
            default_value=self.config['directories']['output_dir'],
            browse=True
        )

        # FPS with default value
        fps_frame = ctk.CTkFrame(parent)
        fps_frame.pack(pady=(15, 5), fill="x", padx=20)
        
        ctk.CTkLabel(fps_frame, text="FPS").pack(side="left", padx=(0, 10))
        self.fps_entry = ctk.CTkEntry(fps_frame, width=70)
        self.fps_entry.pack(side="left")
        self.fps_entry.insert(0, str(self.config['video']['fps']))

        # Checkboxes for flags
        flags_frame = ctk.CTkFrame(parent)
        flags_frame.pack(pady=(15, 5), fill="x", padx=20)
        
        self.is_dense_var = ctk.BooleanVar(value=self.config['flags']['is_dense'])
        self.view_pcd_var = ctk.BooleanVar(value=self.config['flags']['view_pointcloud'])
        self.save_pcd_var = ctk.BooleanVar(value=self.config['flags']['save_pointcloud'])
        self.filter_pcd_var = ctk.BooleanVar(value=self.config['flags']['filter_pointcloud'])
        
        ctk.CTkCheckBox(flags_frame, text="Dense Reconstruction", variable=self.is_dense_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(flags_frame, text="View Pointcloud", variable=self.view_pcd_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(flags_frame, text="Save Pointcloud", variable=self.save_pcd_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(flags_frame, text="Filter Pointcloud", variable=self.filter_pcd_var).pack(anchor="w", pady=2)

        # Buttons
        buttons_frame = ctk.CTkFrame(parent)
        buttons_frame.pack(pady=(20, 10), fill="x", padx=20)
        
        self.run_button = ctk.CTkButton(
            buttons_frame, 
            text="▶ Run Pipeline", 
            command=self.run_pipeline,
            height=40,
            font=("Segoe UI", 14)
        )
        self.run_button.pack(pady=(0, 10), fill="x")

        self.viewer_button = ctk.CTkButton(
            buttons_frame, 
            text="🔍 Open Viewer",
            command=self.open_viewer,
            height=35
        )
        self.viewer_button.pack(pady=(0, 5), fill="x")

        self.theme_button = ctk.CTkButton(
            buttons_frame, 
            text="🌗 Toggle Theme", 
            command=self.toggle_theme,
            height=35
        )
        self.theme_button.pack(pady=(0, 10), fill="x")

    def create_gsplat_tab(self, parent):
        title = ctk.CTkLabel(parent, text="GSplat Settings", font=("Segoe UI", 20, "bold"))
        title.pack(pady=(10, 20))

        # Create entries for GSplat parameters
        self.gsplat_entries = {}
        
        # Training parameters
        training_frame = ctk.CTkFrame(parent)
        training_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(training_frame, text="Training Parameters", font=("Segoe UI", 16)).pack(pady=(5, 10))
        
        # Load config to get default values
        config = load_config()
        default_epochs = 1  # Default from config.yaml
        
        # Create Max Epochs entry with integer validation
        epochs_frame = ctk.CTkFrame(training_frame)
        epochs_frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(epochs_frame, text="Max Epochs").pack(side="left", padx=(5, 10))
        
        def validate_int(value):
            if value == "":
                return True
            try:
                val = int(value)
                return val >= 1
            except ValueError:
                return False
            
        vcmd = (self.register(validate_int), '%P')
        
        epochs_entry = ctk.CTkEntry(epochs_frame, width=100, validate="key", validatecommand=vcmd)
        epochs_entry.pack(side="left", padx=5)
        epochs_entry.insert(0, str(default_epochs))
        
        def update_epochs(delta):
            try:
                current = int(epochs_entry.get() or "0")
                new_value = max(1, current + delta)
                epochs_entry.delete(0, "end")
                epochs_entry.insert(0, str(new_value))
            except ValueError:
                epochs_entry.delete(0, "end")
                epochs_entry.insert(0, str(default_epochs))
        
        ctk.CTkButton(epochs_frame, text="-", width=30, command=lambda: update_epochs(-1)).pack(side="left", padx=2)
        ctk.CTkButton(epochs_frame, text="+", width=30, command=lambda: update_epochs(1)).pack(side="left", padx=2)
        
        self.gsplat_entries['max_epochs'] = epochs_entry

    def create_numeric_entry(self, parent, label, default_value, min_val, max_val, step=1):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(frame, text=label).pack(side="left", padx=(5, 10))
        
        var = ctk.DoubleVar(value=default_value)
        entry = ctk.CTkEntry(frame, width=100, textvariable=var)
        entry.pack(side="left", padx=5)
        
        def validate_and_update(delta):
            try:
                current = float(var.get())
                new_value = current + delta
                if min_val <= new_value <= max_val:
                    var.set(new_value)
            except ValueError:
                var.set(default_value)
        
        ctk.CTkButton(frame, text="-", width=30, command=lambda: validate_and_update(-step)).pack(side="left", padx=2)
        ctk.CTkButton(frame, text="+", width=30, command=lambda: validate_and_update(step)).pack(side="left", padx=2)
        
        return var

    def create_entry(self, parent, label_text, default_value="", browse=False, video=False):
        frame = ctk.CTkFrame(parent)
        frame.pack(pady=(5, 10), fill="x", padx=20)
        
        ctk.CTkLabel(frame, text=label_text, anchor="w").pack(fill="x")
        entry = ctk.CTkEntry(frame, width=250)
        entry.pack(side="left", pady=(5, 0))
        entry.insert(0, default_value)
        
        if browse:
            btn = ctk.CTkButton(
                frame, 
                text="📂", 
                width=40,
                command=lambda: self.browse_file(entry, video)
            )
            btn.pack(side="left", padx=(5, 0), pady=(5, 0))
        
        return entry

    def browse_file(self, entry, video=False):
        path = askopenfilename() if video else askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def create_main_area(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)

        # Console Section
        console_frame = ctk.CTkFrame(self.main_frame)
        console_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title = ctk.CTkLabel(
            console_frame, 
            text="Pipeline Console", 
            font=("Segoe UI", 24, "bold")
        )
        title.pack(pady=(10, 5))

        # Create a larger console with custom font
        self.console = ctk.CTkTextbox(
            console_frame, 
            height=400,
            font=("Consolas", 12),
            wrap="word"
        )
        self.console.pack(fill="both", expand=True, padx=10, pady=5)
        self.console.configure(state="disabled")

        # Configure console colors based on the current theme
        if self.theme_mode == "dark":
            self.console._textbox.configure(bg="#2b2b2b", fg="#ffffff")
        else:
            self.console._textbox.configure(bg="#ffffff", fg="#000000")

        # Redirect stdout and stderr
        sys.stdout = ConsoleRedirect(self.console, "stdout")
        sys.stderr = ConsoleRedirect(self.console, "stderr")

    def create_embedded_viewer(self):
        self.viewer_frame = ctk.CTkFrame(self.main_frame)
        self.viewer_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        viewer_header = ctk.CTkFrame(self.viewer_frame)
        viewer_header.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            viewer_header,
            text="3D Viewer",
            font=("Segoe UI", 16, "bold")
        ).pack(side="left", padx=5)

        self.viewer_status = ctk.CTkLabel(
            viewer_header,
            text="WebSocket viewer not launched yet.",
            text_color="gray"
        )
        self.viewer_status.pack(side="right", padx=5)

        self.viewer_content = ctk.CTkFrame(self.viewer_frame)
        self.viewer_content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.viewer_content.configure(fg_color="#1e1e1e")  # dark background

    def log(self, message):
        self.console.configure(state="normal")
        self.console.insert("end", f"{message}\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def open_viewer(self, model_path=None):
        """Open the viser viewer"""
        self.log("Launching viser viewer...")
        self.viewer_status.configure(text="Launching viewer...", text_color="orange")

        if model_path is None:
            # Try to find the latest model
            gsplat_output_dir = os.path.join("gsplat_light/outputs", project_name)
            if not os.path.exists(gsplat_output_dir):
                error_msg = "No GSplat output directory found. Please run the pipeline first."
                self.log(f"❌ {error_msg}")
                self.viewer_status.configure(text="Viewer not ready", text_color="red")
                messagebox.showerror("Error", error_msg)
                return

            checkpoint_path = os.path.join(gsplat_output_dir, "checkpoints")
            ply_files = [f for f in os.listdir(gsplat_output_dir) if f.endswith('.ply')]
            
            if os.path.exists(checkpoint_path):
                model_path = checkpoint_path
            elif ply_files:
                model_path = os.path.join(gsplat_output_dir, ply_files[0])
            else:
                error_msg = "No model checkpoint or PLY file found. Please run the pipeline first."
                self.log(f"❌ {error_msg}")
                self.viewer_status.configure(text="Viewer not ready", text_color="red")
                messagebox.showerror("Error", error_msg)
                return

        # Try to open the viewer
        self.viewer.open_viewer(self, model_path)
        self.viewer_status.configure(text="Waiting for viewer...", text_color="orange")
        self.log("⏳ Waiting for viser viewer to start...")

    def run_pipeline(self):
        def thread_fn():
            try:
                self.progress_bar.set(0.1)
                video_path = self.video_entry.get()
                image_dir = pathlib.Path(self.image_entry.get())
                output_path = pathlib.Path(self.output_entry.get())
                fps = int(self.fps_entry.get())

                # Get flag values
                is_dense = self.is_dense_var.get()
                view_pcd = self.view_pcd_var.get()
                save_pcd = self.save_pcd_var.get()
                filter_pcd = self.filter_pcd_var.get()

                logger.info(f"Starting pipeline with video: {video_path}")
                logger.info(f"Extracting frames at {fps} FPS")
                extract_frames(video_path, image_dir, fps)
                self.progress_bar.set(0.3)

                logger.info("Running SfM...")
                run_sfm(image_dir=image_dir, output_path=output_path, is_dense=is_dense)
                self.progress_bar.set(0.5)

                logger.info("Processing Pointcloud...")
                process_pointcloud(
                    output_path, 
                    view_pcd=view_pcd,
                    save_pcd=save_pcd,
                    filter_pcd=filter_pcd,
                    is_dense=is_dense
                )
                self.progress_bar.set(0.6)

                sd_args = sd_parse_args()
                sd_args.image_dir = str(output_path / "mvs/images")
                sd_args.output = str(output_path / "mvs/SD")

                if not os.path.exists(sd_args.output):
                    logger.info("Extracting SD Features...")
                    extract_sd_features(args=sd_args)
                self.progress_bar.set(0.8)

                logger.info("Running GSplat Pipeline...")
                
                # Load GSplat arguments from config
                config = load_config()
                gsplat_args = config['gspl_args']

                # Create output directory if it doesn't exist
                gsplat_output_dir = os.path.join("gsplat_light/outputs", project_name)
                os.makedirs(gsplat_output_dir, exist_ok=True)

                # Run GSplat pipeline if needed
                if not os.path.exists(gsplat_output_dir) or not any(f.endswith('.ply') for f in os.listdir(gsplat_output_dir)):
                    gspl_pipeline(args=gsplat_args)

                # Check for checkpoint or point cloud
                checkpoint_path = os.path.join(gsplat_output_dir, "checkpoints")
                ply_files = [f for f in os.listdir(gsplat_output_dir) if f.endswith('.ply')]
                
                if not os.path.exists(checkpoint_path) and not ply_files:
                    raise Exception("GSplat pipeline did not generate necessary output files")

                # Use the latest checkpoint or PLY file
                if os.path.exists(checkpoint_path):
                    model_path = checkpoint_path
                else:
                    model_path = os.path.join(gsplat_output_dir, ply_files[0])

                self.progress_bar.set(1.0)
                logger.info("✅ Pipeline Completed Successfully!")

                # Automatically open the viewer after pipeline completion
                self.after(1000, lambda: self.open_viewer(model_path))

            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                messagebox.showerror("Error", str(e))

        threading.Thread(target=thread_fn).start()

    def toggle_theme(self):
        if self.theme_mode == "dark":
            ctk.set_appearance_mode("light")
            self.theme_mode = "light"
        else:
            ctk.set_appearance_mode("dark")
            self.theme_mode = "dark"


if __name__ == "__main__":
    app = App()
    app.mainloop()
