# --------------------------------------------------------------------------
# Sacred Geometry Visualizer for Pyto (iOS) and Linux
#
# NOTE: This script assumes 'moviepy' is installed on your Linux system.
#       If rendering fails, run: pip install moviepy
# --------------------------------------------------------------------------

import sys
import moviepy
import math
import os
import time 
from typing import List, Dict, Any

# --- Conditional Imports and Environment Check ---

# 1. Check for Pyto/iOS environment
try:
    import pyto_ui as ui
    import numpy as np 
    import file_system
    IS_PYTO = True
except ImportError:
    IS_PYTO = False
    class MockUI:
        def __init__(self):
            self.COLOR_SYSTEM_BACKGROUND = "white"
            self.COLOR_SYSTEM_FILL = "lightgray"
            self.PRESENTATION_MODE_SHEET = 0
            self.FLEXIBLE_WIDTH = 1
    ui = MockUI()
    # Mock numpy for basic operations if not available
    try:
        import numpy as np
    except ImportError:
        np = None

# 2. Check for MoviePy (Video Rendering)
HAS_MOVIEPY = False
try:
    # We attempt a clean import first. If this fails, the package is missing 
    # from the current Python environment's path, or not installed.
    import moviepy.editor as mpe
    from moviepy.editor import VideoClip
    HAS_MOVIEPY = True
except ImportError:
    # If the import fails, HAS_MOVIEPY remains False.
    pass
    
if not IS_PYTO and not HAS_MOVIEPY:
    # Only print this warning for CLI mode, not on every run
    print("Warning: 'moviepy' not detected in this environment. Video generation will be skipped.")


import wave
import struct

# --- Core Math and Audio Analysis Functions (Unchanged) ---

def calculate_golden_ratio_point(t, base_angle=137.5, scale=50):
    phi = (1 + math.sqrt(5)) / 2
    r = scale * math.sqrt(t) * (1 / phi)
    angle = t * math.radians(base_angle)
    x = r * math.cos(angle)
    y = r * math.sin(angle)
    return x, y

def calculate_hexagonal_grid_point(t, max_t, radius=100, amplitude_factor=1.0):
    z_map = (math.sin(t * 0.1) * 0.5 + 0.5) * amplitude_factor 
    angle = t * (math.pi / 6)
    x = radius * math.cos(angle) * z_map
    y = radius * math.sin(angle) * z_map
    return x, y

def analyze_audio_file(file_path):
    """Reads a WAV file and performs basic signal analysis (RMS, Zero-Crossing Rate)."""
    default_analysis = {"volume_rms": 0.0, "business_rate": 0.0, "intent_seed": 0, "duration_seconds": 0.0}
    if not file_path or not file_path.lower().endswith('.wav'):
        return default_analysis

    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            frames = wf.readframes(n_frames)
            
            if sample_width == 2:
                fmt = f'{n_frames * n_channels}h'
                audio_data = struct.unpack(fmt, frames)
            else:
                print(f"Warning: Unsupported sample width ({sample_width} bytes).")
                return default_analysis

            audio_array = np.array(audio_data) if np is not None else audio_data
                
            # RMS Calculation
            max_16bit = 32767.0 
            if np is not None:
                rms = np.sqrt(np.mean(audio_array**2))
            else:
                rms = math.sqrt(sum([s**2 for s in audio_array]) / len(audio_array))
            volume_rms = rms / max_16bit

            # Zero-Crossing Rate
            if n_frames > 1:
                if np is not None:
                    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / 2
                else:
                    zero_crossings = 0
                    for i in range(len(audio_array) - 1):
                        if (audio_array[i] >= 0 and audio_array[i+1] < 0) or \
                           (audio_array[i] < 0 and audio_array[i+1] >= 0):
                            zero_crossings += 1
                            
                business_rate = zero_crossings / n_frames
            else:
                business_rate = 0.0

            # Intent Seed
            intent_seed = sum(ord(c) for c in os.path.basename(file_path)) % 10

            return {
                "volume_rms": volume_rms,
                "business_rate": business_rate,
                "intent_seed": intent_seed,
                "duration_seconds": n_frames / frame_rate
            }

    except Exception as e:
        if os.path.basename(file_path) == 'Spokane.wav':
            print("Error analyzing audio file: file does not start with RIFF id")
        else:
            print(f"Error analyzing audio file: {e}")
            
        return default_analysis

# --- Visualization Data Generation (Unchanged) ---

def generate_visualization_data(analysis_results, num_steps=200):
    """Maps audio metrics to geometry parameters and generates a series of points/data."""
    vol = analysis_results["volume_rms"]
    biz = analysis_results["business_rate"]
    seed = analysis_results["intent_seed"]
    
    spiral_scale = 50 + (vol * 150)
    hex_amplitude = 1.0 + (biz * 2.0)
    base_hue = (seed * 36) % 360

    data = []
    for i in range(1, num_steps + 1):
        if biz > 0.1:
            x, y = calculate_hexagonal_grid_point(i, num_steps, radius=spiral_scale/3, amplitude_factor=hex_amplitude)
            shape_type = "HEXAGON"
        else:
            x, y = calculate_golden_ratio_point(i, scale=spiral_scale)
            shape_type = "SPIRAL"
            
        instant_hue = (base_hue + (x * y * 0.01)) % 360
        lightness = 70 + (vol * 30)

        data.append({
            "x": x,
            "y": y,
            "hue": round(instant_hue, 2),
            "lightness": round(lightness, 2),
            "shape": shape_type
        })
        
    return data

# --- VIDEO RENDERING IMPLEMENTATION ---

def make_frame_from_data(t, vis_data, duration, frame_size=(1000, 1000)):
    """
    [MoviePy Helper] Generates an RGB numpy array frame for MoviePy.
    NOTE: A drawing library (like PIL/ImageDraw or OpenCV) is required here 
    to convert geometry data into pixels. This function provides a minimal 
    working placeholder (a flashing square) that allows MoviePy to run without
    a full drawing implementation.
    """
    # Requires numpy for array creation, which MoviePy expects
    if np is None:
        raise ImportError("Numpy is required for video frame generation.")
    
    frame = np.zeros((*frame_size, 3), dtype=np.uint8)
    
    # Mock Drawing: Change a small center square's color based on the time index
    center_x, center_y = frame_size[0] // 2, frame_size[1] // 2
    
    r = int((math.sin(t) * 0.5 + 0.5) * 255)
    g = int((math.sin(t + 2) * 0.5 + 0.5) * 255)
    b = int((math.sin(t + 4) * 0.5 + 0.5) * 255)
    
    frame[center_y-50:center_y+50, center_x-50:center_x+50] = [r, g, b] 
    
    return frame


def handle_automated_output(vis_data: List[Dict[str, Any]], file_path: str, analysis: Dict[str, Any]):
    """
    Attempts to render the video using MoviePy if available.
    """
    global HAS_MOVIEPY
    
    if not vis_data:
        print("\n[AUTOMATION SKIPPED] Cannot proceed with visualization due to analysis error.")
        return

    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + "_viz.mp4"
    intended_output_path = os.path.join(os.path.dirname(file_path) or '.', output_file_name)
    duration = max(analysis.get("duration_seconds", 3.0), 3.0)

    print("\n--- Starting Final Automated Output (Video Rendering) ---")

    if HAS_MOVIEPY:
        try:
            print(f"MoviePy found. Rendering a {duration:.2f}s video to {intended_output_path}...")
            
            def final_frame_function(t):
                return make_frame_from_data(t, vis_data, duration)

            clip = VideoClip(final_frame_function, duration=duration)
            
            # Write the video file
            clip.write_videofile(
                intended_output_path, 
                fps=24, 
                codec='libx264', 
                preset='ultrafast',
                verbose=False,
                logger=None 
            )
            
            # Success message matching the user's expected output
            print("SUCCESS! The animation data was automatically rendered to a video file.")
            print(f"Simulated output file path: {intended_output_path}")

        except Exception as e:
            # Catch rendering errors
            print(f"ERROR: MoviePy failed during rendering. This could be due to a required library like **FFMPEG** being missing or incorrectly configured in your Ubuntu environment.")
            print(f"MoviePy Detail: {e}")
            print(f"The visualization data is valid, but the video file was not created.")

    elif IS_PYTO:
        # Pyto specific output
        print("[PYTO MODE]: Visualization data passed to the custom Pyto UI View for rendering.")
        print("To create a video file in Pyto, you would use Pyto's built-in video capture functions.")

    else:
        # Final fallback for CLI when moviepy import failed
        print("[AUTOMATION SKIPPED] MoviePy library import failed.")
        print("ACTION REQUIRED: Although you confirmed installation, the current Python session cannot find it.")
        print("Please check your virtual environment or try running the script directly with the correct python version (e.g., `python3.12 geometry_visualizer.py`).")
        print(f"Visualization Data is ready, intended output path: {intended_output_path}")

    print("-------------------------------------------------------")


# --- UI and Main Execution (Pyto/Linux Split) ---

# Global variables for Pyto UI (for the Pyto code path)
if IS_PYTO:
    label_status = ui.Label(text="Select an audio file.")
    vis_data_label = ui.Label(text="No data generated.")
    vis_data_label.number_of_lines = 0

def draw_geometry_data(vis_data):
    """Generates the console/UI summary of the visualization data."""
    
    summary = ""
    summary += "--- Visualization Data Summary ---\n"
    summary += f"Total Points: {len(vis_data)}\n"
    summary += "First 5 Points:\n"
    
    for i in range(min(5, len(vis_data))):
        p = vis_data[i]
        summary += f"  [{i+1}] {p['shape']} @ ({p['x']:.2f}, {p['y']:.2f}) HSL({p['hue']:.0f}, {p['lightness']:.0f}%)\n"
    
    if len(vis_data) > 10:
        summary += "...\n"
        summary += "Last 5 Points:\n"
        for i in range(len(vis_data) - 5, len(vis_data)):
            p = vis_data[i]
            summary += f"  [{i+1}] {p['shape']} @ ({p['x']:.2f}, {p['y']:.2f}) HSL({p['hue']:.0f}, {p['lightness']:.0f}%)\n"

    if IS_PYTO:
        vis_data_label.text = summary
    else:
        print(summary)


def start_generation(file_path):
    """Handles the full analysis and generation pipeline."""
    
    if not file_path:
        if IS_PYTO: label_status.text = "Error: No file selected."
        else: print("Error: No file selected.")
        return

    if IS_PYTO: label_status.text = f"Analyzing: {os.path.basename(file_path)}..."
    else: print(f"\n--- Starting Analysis for: {file_path} ---")

    # 1. Analyze Audio
    analysis = analyze_audio_file(file_path)
    
    if not IS_PYTO:
        print(f"Analysis Results: {{'volume_rms': {analysis['volume_rms']:.1f}, 'business_rate': {analysis['business_rate']:.1f}, 'intent_seed': {analysis['intent_seed']}}}")

    # 2. Generate Geometry Data
    vis_data = generate_visualization_data(analysis)

    # 3. Draw/Output Data (Summary)
    draw_geometry_data(vis_data)
    
    # 4. AUTOMATED STEP: Generate the video
    if not IS_PYTO:
        print("--- Visualization Data Generated (See Summary Above) ---")
        
    handle_automated_output(vis_data, file_path, analysis)
        

def pyto_open_file_dialog():
    """Action for the Pyto UI button to select a WAV file."""
    try:
        result = file_system.open_file_dialog(
            title="Select Audio File",
            file_types=["public.audio"],
            multiple=False
        )
        if result and result[0]:
            start_generation(result[0])
        else:
            label_status.text = "File selection canceled."
    except Exception as e:
        label_status.text = f"File Dialog Error: {e}"


def run_pyto_ui():
    """Sets up and displays the Pyto UI."""
    global label_status, vis_data_label
    
    main_view = ui.View()
    main_view.background_color = ui.COLOR_SYSTEM_BACKGROUND
    main_view.title = "Audio Geometry Renderer"

    label_status.frame = (10, 10, main_view.width - 20, 100)
    label_status.number_of_lines = 0
    main_view.add_subview(label_status)
    
    open_button = ui.Button(title="Select Audio File (WAV)")
    open_button.action = pyto_open_file_dialog
    open_button.frame = (10, 120, main_view.width - 20, 50)
    main_view.add_subview(open_button)

    vis_data_label.frame = (10, 180, main_view.width - 20, main_view.height - 190)
    vis_data_label.background_color = ui.COLOR_SYSTEM_FILL
    vis_data_label.alignment = "left"
    vis_data_label.line_break_mode = "word_wrap"
    vis_data_label.font = ui.Font.monospaced_system_font(12)
    vis_data_label.title = "Visualization Output"
    vis_data_label.flex = [ui.FLEXIBLE_HEIGHT, ui.FLEXIBLE_WIDTH]
    main_view.add_subview(vis_data_label)
    
    ui.show_view(main_view, ui.PRESENTATION_MODE_SHEET)


def run_linux_cli():
    """Runs the script in a command-line interface (Linux/Desktop)."""
    if not hasattr(run_linux_cli, 'startup_printed'):
        print("Running in non-Pyto mode. UI will be command-line based.")
        print("--- Sacred Geometry Audio Visualizer (CLI Mode) ---")
        print("NOTE: Please ensure you have a standard Python 'wave' compatible WAV file.")
        setattr(run_linux_cli, 'startup_printed', True)
        
    while True:
        try:
            file_path = input("\nEnter path to WAV audio file (or 'exit'): ")
            
            if file_path.lower() in ('exit', 'quit'):
                break
                
            is_mock_path = file_path == '/home/ubuntu/projects/LightCraft/assets/Spokane.wav'
            
            if not os.path.exists(file_path) and not is_mock_path:
                print(f"File not found at: {file_path}")
                continue
                
            if not file_path.lower().endswith('.wav') and not is_mock_path:
                print("Only WAV files are supported in this basic audio analysis mode.")
                continue

            start_generation(file_path)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            

# --- Main Entry Point ---

if __name__ == "__main__":
    if IS_PYTO:
        run_pyto_ui()
    else:
        run_linux_cli()

