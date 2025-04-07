import customtkinter as ctk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, askdirectory
import pathlib
import threading
import os
import sys
import webview
import subprocess
from PIL import Image, ImageTk

# Import project modules
from sfm_project.pycolmap_sfm.sfm_colmap import run_sfm
from sfm_project.pycolmap_sfm.utils import extract_frames, process_pointcloud
sys.path.append('gsplat_light')
from internal.entrypoints.gspl import cli as gspl_pipeline
sys.path.append('gsplat_light/utils')
from utils.sd_feature_extraction import main as extract_sd_features
from utils.sd_feature_extraction import parse_args as sd_parse_args

# Material Design Colors
COLORS = {
    "primary": "#3F51B5",       # Indigo
    "primary_dark": "#303F9F",  # Dark Indigo
    "primary_light": "#C5CAE9", # Light Indigo
    "accent": "#FF4081",        # Pink
    "text_primary": "#FFFFFF",  # White
    "text_secondary": "#B0BEC5",# Bluish Grey
    "bg_dark": "#121212",       # Material Dark Background
    "card_dark": "#1E1E1E",     # Material Dark Card
    "card_light": "#FFFFFF",    # Material Light Card
    "bg_light": "#F5F5F5",      # Material Light Background
    "error": "#F44336",         # Red
    "success": "#4CAF50",       # Green
    "warning": "#FFC107",       # Amber
}

# Set appearance defaults
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

class MaterialCard(ctk.CTkFrame):
    """A styled card-like frame with a title following Material Design principles"""
    
    def __init__(self, master, title, **kwargs):
        super().__init__(master, corner_radius=10, fg_color=COLORS["card_dark"], **kwargs)
        
        # Title bar
        self.title_bar = ctk.CTkFrame(self, corner_radius=8, fg_color=COLORS["primary"])
        self.title_bar.pack(fill="x", padx=2, pady=(2, 0))
        
        # Title label
        self.title_label = ctk.CTkLabel(
            self.title_bar, 
            text=title, 
            font=("Roboto", 14, "bold"),
            text_color=COLORS["text_primary"]
        )
        self.title_label.pack(pady=8)
        
        # Content frame
        self.content = ctk.CTkFrame(self, corner_radius=8, fg_color=COLORS["card_dark"])
        self.content.pack(fill="both", expand=True, padx=2, pady=(0, 2))

class ParameterSlider(ctk.CTkFrame):
    """A labeled slider with current value display"""
    
    def __init__(self, master, text, from_, to, **kwargs):
        super().__init__(master, fg_color="transparent")
        
        self.value_var = ctk.DoubleVar(value=from_)
        
        # Layout
        self.grid_columnconfigure(0, weight=6)
        self.grid_columnconfigure(1, weight=1)
        
        # Label
        self.label = ctk.CTkLabel(self, text=text)
        self.label.grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        
        # Value label
        self.value_label = ctk.CTkLabel(self, text=str(from_), width=50)
        self.value_label.grid(row=0, column=1, sticky="e", padx=(0, 10), pady=(5, 0))
        
        # Slider
        self.slider = ctk.CTkSlider(
            self, 
            from_=from_, 
            to=to, 
            variable=self.value_var,
            command=self.update_value
        )
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))
    
    def update_value(self, value):
        # Format as integer if it's a whole number
        if value == int(value):
            self.value_label.configure(text=f"{int(value)}")
        else:
            self.value_label.configure(text=f"{value:.2f}")
    
    def get(self):
        return self.value_var.get()
    
    def set(self, value):
        self.value_var.set(value)
        self.update_value(value)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("3D Reconstruction Pipeline")
        self.geometry(f"{1400}x{900}")
        self.minsize(1200, 700)

        self.sidebar_visible = True
        self.theme_mode = "dark"
        
        # Project name entry
        self.project_name = "alpha"
        
        # Pipeline parameters with defaults
        self.parameters = {
            "fps": 1,
            "max_epochs": 50,
            "num_splats": 100000,
            "learning_rate": 0.01,
            "is_dense": False,
        }

        # Configure the main grid
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Main content
        self.grid_rowconfigure(0, weight=1)     # Make the main row expandable

        self.create_sidebar()
        self.create_main_area()
        
        # Console redirection
        sys.stdout = ConsoleRedirect(self.console)
        sys.stderr = ConsoleRedirect(self.console)
        
        # Log initial message
        self.log("System ready. Please configure your parameters and select input files.")

    def create_sidebar(self):
        # Main sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=COLORS["card_dark"])
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.sidebar.grid_propagate(False)
        
        # App title
        title_frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS["primary"], corner_radius=10)
        title_frame.pack(fill="x", padx=10, pady=(10, 15))
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text="3D Reconstruction",
            font=("Roboto", 20, "bold"),
            text_color=COLORS["text_primary"]
        )
        title_label.pack(pady=12)
        
        # Input Files Card
        input_card = MaterialCard(self.sidebar, "Input Files")
        input_card.pack(fill="x", padx=10, pady=10)
        
        # Video input
        video_frame = ctk.CTkFrame(input_card.content, fg_color="transparent")
        video_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        video_label = ctk.CTkLabel(video_frame, text="Video Input")
        video_label.pack(anchor="w")
        
        video_browse_frame = ctk.CTkFrame(video_frame, fg_color="transparent")
        video_browse_frame.pack(fill="x", pady=(0, 10))
        video_browse_frame.columnconfigure(0, weight=1)
        video_browse_frame.columnconfigure(1, weight=0)
        
        self.video_entry = ctk.CTkEntry(video_browse_frame, placeholder_text="Select video file...")
        self.video_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        video_browse_btn = ctk.CTkButton(
            video_browse_frame, 
            text="Browse",
            width=80,
            command=lambda: self.browse_file(self.video_entry, True)
        )
        video_browse_btn.grid(row=0, column=1)
        
        # Image directory
        img_frame = ctk.CTkFrame(input_card.content, fg_color="transparent")
        img_frame.pack(fill="x", padx=10, pady=5)
        
        img_label = ctk.CTkLabel(img_frame, text="Image Directory")
        img_label.pack(anchor="w")
        
        img_browse_frame = ctk.CTkFrame(img_frame, fg_color="transparent")
        img_browse_frame.pack(fill="x", pady=(0, 10))
        img_browse_frame.columnconfigure(0, weight=1)
        img_browse_frame.columnconfigure(1, weight=0)
        
        self.image_entry = ctk.CTkEntry(img_browse_frame, placeholder_text="Select image directory...")
        self.image_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        img_browse_btn = ctk.CTkButton(
            img_browse_frame, 
            text="Browse",
            width=80,
            command=lambda: self.browse_file(self.image_entry, False)
        )
        img_browse_btn.grid(row=0, column=1)
        
        # Output directory
        output_frame = ctk.CTkFrame(input_card.content, fg_color="transparent")
        output_frame.pack(fill="x", padx=10, pady=5)
        
        output_label = ctk.CTkLabel(output_frame, text="Output Directory")
        output_label.pack(anchor="w")
        
        output_browse_frame = ctk.CTkFrame(output_frame, fg_color="transparent")
        output_browse_frame.pack(fill="x", pady=(0, 10))
        output_browse_frame.columnconfigure(0, weight=1)
        output_browse_frame.columnconfigure(1, weight=0)
        
        self.output_entry = ctk.CTkEntry(output_browse_frame, placeholder_text="Select output directory...")
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        output_browse_btn = ctk.CTkButton(
            output_browse_frame, 
            text="Browse",
            width=80,
            command=lambda: self.browse_file(self.output_entry, False)
        )
        output_browse_btn.grid(row=0, column=1)
        
        # Project name
        proj_name_frame = ctk.CTkFrame(input_card.content, fg_color="transparent")
        proj_name_frame.pack(fill="x", padx=10, pady=5)
        
        proj_name_label = ctk.CTkLabel(proj_name_frame, text="Project Name")
        proj_name_label.pack(anchor="w")
        
        self.proj_name_entry = ctk.CTkEntry(proj_name_frame)
        self.proj_name_entry.pack(fill="x", pady=(0, 10))
        self.proj_name_entry.insert(0, self.project_name)
        
        # Parameters Card
        params_card = MaterialCard(self.sidebar, "Parameters")
        params_card.pack(fill="x", padx=10, pady=10)
        
        # FPS slider
        self.fps_slider = ParameterSlider(
            params_card.content, 
            "Frame Extraction FPS", 
            from_=1, 
            to=30
        )
        self.fps_slider.pack(fill="x", padx=10, pady=5)
        
        # Max epochs slider
        self.epochs_slider = ParameterSlider(
            params_card.content, 
            "Max Epochs", 
            from_=10, 
            to=100
        )
        self.epochs_slider.pack(fill="x", padx=10, pady=5)
        self.epochs_slider.set(50)  # Default value
        
        # Num splats slider
        self.splats_slider = ParameterSlider(
            params_card.content, 
            "Number of Splats (thousands)", 
            from_=10, 
            to=500
        )
        self.splats_slider.pack(fill="x", padx=10, pady=5)
        self.splats_slider.set(100)  # Default value
        
        # Learning rate slider
        self.lr_slider = ParameterSlider(
            params_card.content, 
            "Learning Rate", 
            from_=0.001, 
            to=0.1
        )
        self.lr_slider.pack(fill="x", padx=10, pady=5)
        self.lr_slider.set(0.01)  # Default value
        
        # Dense reconstruction checkbox
        self.dense_var = ctk.BooleanVar(value=False)
        self.dense_checkbox = ctk.CTkCheckBox(
            params_card.content, 
            text="Use Dense Reconstruction",
            variable=self.dense_var,
            onvalue=True,
            offvalue=False
        )
        self.dense_checkbox.pack(anchor="w", padx=10, pady=10)
        
        # Buttons Card
        buttons_card = MaterialCard(self.sidebar, "Actions")
        buttons_card.pack(fill="x", padx=10, pady=10)
        
        # Run Pipeline Button
        self.run_button = ctk.CTkButton(
            buttons_card.content,
            text="▶ Run Pipeline",
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_dark"],
            command=self.run_pipeline
        )
        self.run_button.pack(fill="x", padx=10, pady=(10, 5))
        
        # Open Viewer Button
        self.viewer_button = ctk.CTkButton(
            buttons_card.content,
            text="🔍 Open Viewer",
            fg_color=COLORS["accent"],
            hover_color="#E91E63",  # Darker pink
            command=self.open_embedded_websocket_viewer
        )
        self.viewer_button.pack(fill="x", padx=10, pady=5)
        
        # Toggle Theme Button
        self.theme_button = ctk.CTkButton(
            buttons_card.content,
            text="🌗 Toggle Theme",
            fg_color="#607D8B",  # Blue Grey
            hover_color="#455A64",  # Darker Blue Grey
            command=self.toggle_theme
        )
        self.theme_button.pack(fill="x", padx=10, pady=(5, 10))
        
        # Progress Bar
        progress_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        self.progress_label = ctk.CTkLabel(progress_frame, text="Progress: 0%")
        self.progress_label.pack(anchor="w", pady=(0, 5))
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

    def create_main_area(self):
        # Main content area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS["bg_dark"])
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)  # Console takes most space
        self.main_frame.grid_rowconfigure(1, weight=2)  # Viewer gets more space
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Console card
        console_card = MaterialCard(self.main_frame, "Console Output")
        console_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        
        self.console = ctk.CTkTextbox(
            console_card.content, 
            height=200, 
            font=("Consolas", 12),
            fg_color=COLORS["bg_dark"],
            text_color=COLORS["text_secondary"],
        )
        self.console.pack(fill="both", expand=True, padx=10, pady=10)
        self.console.configure(state="disabled")
        
        # Viewer card
        viewer_card = MaterialCard(self.main_frame, "3D Viewer")
        viewer_card.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        self.viewer_frame = ctk.CTkFrame(viewer_card.content, fg_color=COLORS["bg_dark"])
        self.viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder for viewer
        self.viewer_label = ctk.CTkLabel(
            self.viewer_frame,
            text="WebSocket viewer not launched yet.\nRun the pipeline or click 'Open Viewer' to start.",
            font=("Roboto", 16),
            text_color=COLORS["text_secondary"]
        )
        self.viewer_label.pack(expand=True, fill="both")

    def log(self, message):
        """Write message to the console and update UI"""
        self.console.configure(state="normal")
        self.console.insert("end", f"{message}\n")
        self.console.see("end")
        self.console.configure(state="disabled")
        self.update()  # Update UI to show progress

    def browse_file(self, entry, video=False):
        """Show file/directory dialog and update entry"""
        path = askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")]) if video else askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def toggle_theme(self):
        """Switch between light and dark themes"""
        if self.theme_mode == "dark":
            ctk.set_appearance_mode("light")
            self.theme_mode = "light"
            # Update colors for light theme
            self.main_frame.configure(fg_color=COLORS["bg_light"])
            self.console.configure(fg_color="#EEEEEE", text_color="#333333")
            self.viewer_frame.configure(fg_color="#EEEEEE")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_mode = "dark"
            # Update colors for dark theme
            self.main_frame.configure(fg_color=COLORS["bg_dark"])
            self.console.configure(fg_color=COLORS["bg_dark"], text_color=COLORS["text_secondary"])
            self.viewer_frame.configure(fg_color=COLORS["bg_dark"])
            
    def open_embedded_websocket_viewer(self):
        """Launch the websocket viewer in a new window"""
        self.log("Launching embedded WebSocket viewer...")

        # Force a backend (optional but safer)
        os.environ["PYWEBVIEW_GUI"] = "qt"  # Use 'gtk' on Linux if preferred

        def launch():
            webview.create_window(
                "3D Reconstruction Viewer",
                url="http://localhost:8080",
                width=1280,
                height=800,
                background_color='#1e1e1e',
                resizable=True
            )
            webview.start()

        # pywebview must run on main thread, schedule via .after()
        self.after(100, launch)

        self.viewer_label.configure(text="WebSocket Viewer launched in a new window")
    
    def update_progress(self, value, message=""):
        """Update progress bar and label"""
        self.progress_bar.set(value)
        percentage = int(value * 100)
        self.progress_label.configure(text=f"Progress: {percentage}%")
        if message:
            self.log(message)
        self.update()

    def run_pipeline(self):
        """Run the SFM+GSplat pipeline with the current parameters"""
        # Get values from UI
        video_path = self.video_entry.get()
        image_dir = self.image_entry.get()
        output_path = self.output_entry.get()
        self.project_name = self.proj_name_entry.get()
        
        # Get parameters
        fps = int(self.fps_slider.get())
        max_epochs = int(self.epochs_slider.get())
        num_splats = int(self.splats_slider.get() * 1000)  # Convert from thousands
        learning_rate = self.lr_slider.get()
        is_dense = self.dense_var.get()
        
        # Validate inputs
        if not video_path and not image_dir:
            messagebox.showerror("❌Error", "Please select either a video file or image directory❌")
            return
        
        if not output_path:
            messagebox.showerror("❌Error", "Please select an output directory❌")
            return
        
        if not self.project_name:
            messagebox.showerror("❌Error", "Please enter a project name❌")
            return
        
        # Convert paths to proper objects
        if image_dir:
            image_dir = pathlib.Path(image_dir)
        output_path = pathlib.Path(output_path)
        
        # Configure GSplat arguments
        custom_args = [
            "fit",
            "--config", "gsplat_light/configs/spot_less_splats/gsplat-cluster.yaml",
            "--data.parser.split_mode", "reconstruction",
            "--data.path", str(output_path / "mvs"),
            "-n", self.project_name,
            "--trainer.max_epochs", str(max_epochs),
        ]
        
        # Disable UI controls during processing
        self.run_button.configure(state="disabled", text="Running...")
        
        def thread_fn():
            try:
                # 1. Extract frames if video path is provided
                if video_path:
                    self.update_progress(0.1, f"Extracting frames from: {video_path}")
                    extract_frames(video_path, image_dir, fps)
                
                # 2. Run SFM
                self.update_progress(0.3, "✅\nRunning Structure from Motion...")
                run_sfm(image_dir=image_dir, output_path=output_path, is_dense=is_dense)
                
                # 3. Process pointcloud
                self.update_progress(0.5, "✅\nProcessing pointcloud...")
                process_pointcloud(output_path, view_pcd=False, save_pcd=True, filter_pcd=True, is_dense=is_dense)
                
                # 4. Extract SD features
                sd_args = sd_parse_args()
                sd_args.image_dir = str(output_path / "mvs/images")
                sd_args.output = str(output_path / "mvs/SD")

                if not os.path.exists(sd_args.output):
                    self.update_progress(0.7, "✅\nExtracting SD Features...")
                    extract_sd_features(args=sd_args)
                
                # 5. Run GSplat Pipeline
                self.update_progress(0.8, "✅\nRunning GSplat Pipeline...")
                output_dir = os.path.join("/home/doer/hyperMapper/gsplat_light/outputs/", self.project_name)
                if not os.path.exists(output_dir):
                    gspl_pipeline(args=custom_args)

                self.update_progress(1.0, "✅ Pipeline Completed Successfully!")
                
                # Re-enable UI controls
                self.after(0, lambda: self.run_button.configure(state="normal", text="▶ Run Pipeline"))
                
                # Prompt to open viewer
                if messagebox.askyesno("Success", "Pipeline completed successfully! Open the 3D viewer?"):
                    self.open_embedded_websocket_viewer()
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", str(e))
                self.after(0, lambda: self.run_button.configure(state="normal", text="▶ Run Pipeline"))

        # Run in background thread
        threading.Thread(target=thread_fn, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()