import math
import random
import time
import os
import numpy as np
from typing import List, Dict, Any, Tuple

# --- Performance & Core Libraries ---
from numba import jit 
import librosa
from moviepy.editor import VideoClip, AudioFileClip, AudioFileClip

# --- Configuration Constants ---
SACRED_NUMBER = 7 # Drives the cycle of shape complexity (e.g., layers)
ASTRO_CYCLES = 12 # Drives the slow background thematic shift (Zodiac)
RENDER_FPS = 30
MOCK_DURATION_S = 30.0 # Standard duration for simulation if file not found

# Astrological Archetypes (12 Cycles) - Color and Theme Mapping
ZODIAC_ARCHETYPES = [
    {"sign": "Aries", "element": "Fire", "color": (200, 50, 50), "theme": "Primordial Surge"},
    {"sign": "Taurus", "element": "Earth", "color": (50, 150, 50), "theme": "Rooted Structure"},
    {"sign": "Gemini", "element": "Air", "color": (150, 150, 200), "theme": "Dual Light"},
    {"sign": "Cancer", "element": "Water", "color": (100, 150, 200), "theme": "Lunar Flow"},
    {"sign": "Leo", "element": "Fire", "color": (255, 160, 0), "theme": "Solar Radiance"},
    {"sign": "Virgo", "element": "Earth", "color": (100, 100, 100), "theme": "Ordered Grid"},
    {"sign": "Libra", "element": "Air", "color": (200, 100, 200), "theme": "Balanced Harmony"},
    {"sign": "Scorpio", "element": "Water", "color": (80, 0, 120), "theme": "Deep Transformation"},
    {"sign": "Sagittarius", "element": "Fire", "color": (180, 80, 0), "theme": "Expanding Horizon"},
    {"sign": "Capricorn", "element": "Earth", "color": (50, 50, 80), "theme": "Cosmic Mountain"},
    {"sign": "Aquarius", "element": "Air", "color": (0, 150, 200), "theme": "Future Wave"},
    {"sign": "Pisces", "element": "Water", "color": (100, 100, 255), "theme": "Mystic Veil"},
]

def _interpolate_color(c1, c2, blend):
    """Linearly interpolates between two RGB colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * blend),
        int(c1[1] + (c2[1] - c1[1]) * blend),
        int(c1[2] + (c2[2] - c2[2]) * blend),
    )

@jit(nopython=True, fastmath=True)
def draw_sacred_geometry_numba(geo_array: np.ndarray, width: int, height: int, radius: float, purity: float, layers: int, time_s: float, main_color: np.ndarray) -> np.ndarray:
    """
    The Numba-accelerated core rendering loop. Cycles through Circles (1-3), Squares (4-5), 
    and Hexagons (6-7) based on the 'layers' parameter (Numerology Cycle).
    """
    cx, cy = width / 2, height / 2
    radius_sq = radius**2
    
    # Purity now controls distortion, where 1.0 is a perfect shape and 0.0 is chaos
    distortion_factor = 1.0 - purity
    
    # Base movement speed is low, but influenced by time and instability
    base_spin = time_s * 0.1 * (1.0 + distortion_factor * 2.0)

    # --- Mode 1: CIRCLE GEOMETRY (Layers 1-3: Vesica Piscis, Seed of Life) ---
    if layers <= 3:
        max_circles = 7
        centers = np.zeros((max_circles, 2), dtype=np.float64) 
        
        num_circles = 0
        centers[num_circles] = [cx, cy] # Center Circle
        num_circles += 1
        
        if layers >= 2:
            for i in range(min(6, max_circles - 1)):
                angle = i * (2 * math.pi / 6) + base_spin
                
                # Distortion based on dissonance
                distort = distortion_factor * math.sin(angle * 5 + time_s) * 0.1

                centers[num_circles] = [
                    cx + radius * (1.0 + distort) * math.cos(angle),
                    cy + radius * (1.0 + distort) * math.sin(angle)
                ]
                num_circles += 1

        for y in range(height):
            for x in range(width):
                active_circles = 0
                for i in range(num_circles):
                    center_x, center_y = centers[i]
                    dist_sq = (x - center_x)**2 + (y - center_y)**2
                    if dist_sq < radius_sq:
                        active_circles += 1

                if active_circles > 0:
                    color_scale = min(1.0, active_circles / layers)
                    geo_array[y, x, 0] = np.uint8(main_color[0] * color_scale)
                    geo_array[y, x, 1] = np.uint8(main_color[1] * color_scale)
                    geo_array[y, x, 2] = np.uint8(main_color[2] * color_scale)

    # --- Mode 2: SQUARE/GRID GEOMETRY (Layers 4-5: Cube Archetype) ---
    elif layers <= 5:
        half_side = radius * (1.0 - distortion_factor * 0.2)
        grid_scale = int(5 * purity) + 2 # Grid density increases with purity
        
        # Grid rotation based on time and distortion
        grid_angle = base_spin * 5.0
        cos_a = math.cos(grid_angle)
        sin_a = math.sin(grid_angle)

        for y in range(height):
            for x in range(width):
                # Translate to center
                x_prime = x - cx
                y_prime = y - cy
                
                # Apply rotation
                rot_x = x_prime * cos_a - y_prime * sin_a
                rot_y = x_prime * sin_a + y_prime * cos_a

                # Check bounds of the primary square
                if abs(rot_x) < half_side and abs(rot_y) < half_side:
                    
                    # Create a subtle pulsing inner grid pattern
                    grid_pulse = abs(math.sin(time_s * 5))
                    
                    # Check if pixel falls on a grid line (for visual complexity)
                    if (int(rot_x) % grid_scale == 0) or (int(rot_y) % grid_scale == 0):
                        color_scale = 0.5 + grid_pulse * 0.5  # Flash grid lines
                        geo_array[y, x, 0] = np.uint8(main_color[0] * color_scale)
                        geo_array[y, x, 1] = np.uint8(main_color[1] * color_scale)
                        geo_array[y, x, 2] = np.uint8(main_color[2] * color_scale)

    # --- Mode 3: HEXAGON/TRIANGLE GEOMETRY (Layers 6-7: Vortex/Merkaba Archetype) ---
    else:
        # Define 6 vertices for the main hexagon
        hex_points = np.zeros((6, 2), dtype=np.float64)
        for i in range(6):
            # Hexagon rotates based on time and distortion
            angle = i * (2 * math.pi / 6) + base_spin * 2.0
            hex_points[i] = [
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle)
            ]
        
        # We draw radiating lines from the center, which become blurred/distorted
        line_thickness = 0.005 + distortion_factor * 0.005 # Line thickness increases with dissonance

        for y in range(height):
            for x in range(width):
                dx = x - cx
                dy = y - cy
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 1.0 and dist < radius: 
                    # Calculate the angle of the current pixel relative to center
                    angle_p = math.atan2(dy, dx)
                    
                    # Check proximity to 6 radiating lines (0, 60, 120... degrees)
                    line_found = False
                    for i in range(6):
                        # Calculate the angle of the current vertex
                        angle_v = i * (2 * math.pi / 6) + base_spin * 2.0
                        
                        # Calculate the smallest angular difference
                        diff = abs(angle_p - angle_v)
                        # Handle wrap-around (e.g., 350 deg vs 10 deg)
                        diff = min(diff, 2 * math.pi - diff)
                        
                        if diff < line_thickness: 
                            line_found = True
                            break
                    
                    if line_found:
                        color_scale = 0.4 + purity * 0.6
                        geo_array[y, x, 0] = np.uint8(main_color[0] * color_scale)
                        geo_array[y, x, 1] = np.uint8(main_color[1] * color_scale)
                        geo_array[y, x, 2] = np.uint8(main_color[2] * color_scale)

    return geo_array


class SacredAnalyzer:
    """Analyzes raw audio data to extract 'sacred' parameters using Tonnetz and Phrasing."""
    def __init__(self, audio_data: np.ndarray, sample_rate: int, duration_s: float):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        
        # Analysis Data Storage
        self.beat_times: np.ndarray = np.array([])
        self.tonnetz: np.ndarray = np.array([])
        self.frame_times: np.ndarray = np.array([])
        self.phrase_boundaries: np.ndarray = np.array([])
        self._current_cycle = 0 # Tracks the phrase-aligned numerology cycle
        self._pre_analyze_music()
        
    def _pre_analyze_music(self):
        """Extracts all necessary audio features using Librosa."""
        if self.audio_data.size == 0:
            print("[Analyzer] Cannot pre-analyze: audio data is empty.")
            return

        # 1. Rhythmic Analysis (Beat Tracking)
        _, self.beat_times = librosa.beat.beat_track(y=self.audio_data, sr=self.sample_rate, units='time')
        print(f"  [Analyzer] Found {len(self.beat_times)} beats.")
        
        # 2. Harmonic Analysis (Tonal Centroids / Tonnetz for Intervals/Key)
        # Chroma features are prerequisite for Tonnetz
        chroma = librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate)
        # Tonnetz represents the Tonal Space (Major/Minor, Perfect Fifth, etc.)
        self.tonnetz = librosa.feature.tonnetz(chroma=chroma)

        # Calculate corresponding times for the Tonnetz frames
        self.frame_times = librosa.frames_to_time(np.arange(self.tonnetz.shape[1]), sr=self.sample_rate)
        
        # 3. Phrasing Analysis (Geometry Switch Triggers)
        # Use simple energy (RMS) to find structural changes in the music
        rms = librosa.feature.rms(y=self.audio_data)[0]
        
        # Detect large drops in energy (phrase boundaries)
        # Find points where the RMS drops significantly relative to the median
        median_rms = np.median(rms)
        rms_boundary_frames = librosa.util.peak_pick(
            rms, pre_max=20, post_max=20, pre_avg=10, post_avg=10, delta=median_rms * 0.5, wait=10
        )
        self.phrase_boundaries = librosa.frames_to_time(rms_boundary_frames, sr=self.sample_rate)
        print(f"  [Analyzer] Found {len(self.phrase_boundaries)} musical phrases.")


    def get_sacred_parameters(self, frame_time: float) -> Dict[str, Any]:
        """Generates a dictionary of visual parameters for a specific video time."""
        
        # --- Harmonically Aligned Numerology Cycle ---
        # Advance the cycle only when a phrase boundary is crossed
        if self.phrase_boundaries.size > 0:
            for boundary_time in self.phrase_boundaries:
                if boundary_time > (frame_time - 1/RENDER_FPS) and boundary_time <= frame_time:
                    # Increment the cycle when we cross a boundary
                    self._current_cycle = (self._current_cycle % SACRED_NUMBER) + 1
                    break
        
        geometry_layers = self._current_cycle if self._current_cycle > 0 else 1
        
        # Interpolation Index for Tonal Centroid
        tonnetz_idx = np.searchsorted(self.frame_times, frame_time, side='left')
        tonnetz_idx = min(tonnetz_idx, self.tonnetz.shape[1] - 1)
        
        # 1. Geometry (Loudness/Complexity) - Uses actual audio data for simple RMS/peak
        start_sample = int(frame_time * self.sample_rate)
        end_sample = min(start_sample + self.sample_rate // RENDER_FPS, len(self.audio_data))
        
        if end_sample > start_sample:
            audio_chunk = self.audio_data[start_sample:end_sample]
            amplitude_peak = np.max(np.abs(audio_chunk)) if audio_chunk.size > 0 else 0.0
        else:
            amplitude_peak = 0.0
            
        geometry_intensity = min(1.0, amplitude_peak * 5.0) # Amplified for visual effect

        # 2. Numerology (Base Cycle) - Phrasing Aligned
        numerology_cycle = geometry_layers - 1 # 0 to 6
        color_shift = numerology_cycle / SACRED_NUMBER

        # 3. Music Theory: Consonance/Purity (Tonal Centroid Stability)
        # Dissonance is measured by the magnitude of change in the Tonnetz vector
        current_tonnetz = self.tonnetz[:, tonnetz_idx]
        
        # Calculate the change from the previous frame (proxy for harmonic stability/interval tension)
        if tonnetz_idx > 0:
            prev_tonnetz = self.tonnetz[:, tonnetz_idx - 1]
            harmonic_change_magnitude = np.linalg.norm(current_tonnetz - prev_tonnetz)
        else:
            harmonic_change_magnitude = 0.0
            
        # Normalize change magnitude to 0-1, then invert (low change = high purity)
        # Using a fixed max change of 0.8 for predictable scaling
        max_possible_change = 0.8 
        dissonance_score = min(1.0, harmonic_change_magnitude / max_possible_change)
        consonance_score = 1.0 - dissonance_score # High consonance = high purity

        # 4. Astrology (Time Cycles) - Long-term shift (0.0 to 1.0)
        time_ratio = frame_time / self.duration_s if self.duration_s > 0 else 0
        astro_phase = (time_ratio * ASTRO_CYCLES) % ASTRO_CYCLES 
        
        # 5. Music Theory: Rhythmic Pulse - REAL BEAT TRACKING
        pulse = 0.0
        if self.beat_times.size > 0:
            nearest_beat_idx = np.argmin(np.abs(self.beat_times - frame_time))
            time_diff = np.abs(self.beat_times[nearest_beat_idx] - frame_time)
            
            beat_tolerance = 0.1
            if time_diff < beat_tolerance:
                pulse = 1.0 - (time_diff / beat_tolerance) 

        rhythmic_pulse = pulse 

        # Derived parameters
        derived_instability = (geometry_intensity + rhythmic_pulse + dissonance_score) / 3.0
        
        return {
            "time_s": frame_time,
            "geometry_intensity": geometry_intensity,
            "numerology_shift": color_shift,
            "astro_phase": astro_phase,
            "geometry_layers": geometry_layers, # 1-7, driven by phrase boundaries
            "consonance_score": consonance_score, # Driven by harmonic stability
            "rhythmic_pulse": rhythmic_pulse,     
            "velocity_modifier": derived_instability
        }

class VisualGenerator:
    """Generates the final visual frame (NumPy array) using Sacred Geometry logic."""
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

    def _generate_sacred_color_rgb(self, numerology_shift: float, intensity: float) -> Tuple[int, int, int]:
        """Generates an RGB color based on the numerological shift and amplitude intensity."""
        # HSV conversion logic (simplified phase shift)
        r = int(255 * (0.5 + 0.5 * math.cos(numerology_shift * 2 * math.pi)))
        g = int(255 * (0.5 + 0.5 * math.cos(numerology_shift * 2 * math.pi + 2)))
        b = int(255 * (0.5 + 0.5 * math.cos(numerology_shift * 2 * math.pi + 4)))
        
        # Apply intensity (bloom)
        bloom = 1.0 + intensity * 0.5
        return (
            min(255, int(r * bloom)),
            min(255, int(g * bloom)),
            min(255, int(b * bloom)),
        )

    def _get_background_theme(self, astro_phase: float) -> Tuple[Tuple[int, int, int], str]:
        """
        Calculates background color by interpolating between two Zodiac archetypes.
        """
        idx1 = int(astro_phase) % ASTRO_CYCLES
        idx2 = (idx1 + 1) % ASTRO_CYCLES
        blend = astro_phase - idx1

        theme1 = ZODIAC_ARCHETYPES[idx1]
        theme2 = ZODIAC_ARCHETYPES[idx2]

        color = _interpolate_color(theme1['color'], theme2['color'], blend)
        theme_name = f"{theme1['sign']} blending to {theme2['sign']}"
        return color, theme_name

    def make_frame(self, time_s: float, analyzer: 'SacredAnalyzer') -> np.ndarray:
        """Generates the final frame image array at a specific time (used by MoviePy)."""
        parameters = analyzer.get_sacred_parameters(time_s)
        
        # 1. Background Theme (Astrology)
        bg_color, theme_name = self._get_background_theme(parameters['astro_phase'])
        
        # Create background and geometry arrays
        bg_array = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        geo_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Geometry Parameters
        base_radius = self.height * 0.1
        
        # Geometry size is dictated by loudness/rhythm
        radius = base_radius + (parameters['geometry_intensity'] * 0.5 + parameters['rhythmic_pulse'] * 0.5) * self.height * 0.2
        
        # Purity is dictated by harmonic stability
        purity = parameters['consonance_score'] 
        layers = parameters['geometry_layers']
        
        # Convert color to a Numba-compatible NumPy array before passing to the JIT function
        # Color intensity dictated by loudness/rhythm
        main_color_tuple = self._generate_sacred_color_rgb(parameters['numerology_shift'], (parameters['geometry_intensity'] + parameters['rhythmic_pulse']) / 2)
        main_color = np.array(main_color_tuple, dtype=np.uint8)

        # 2. Draw Sacred Geometry (Foreground - Accelerated via global Numba function)
        geo_array = draw_sacred_geometry_numba(
            geo_array, self.width, self.height, float(radius), float(purity), layers, float(time_s), main_color
        )
        
        # 3. Combine Layers (Simple blending: geometry + background)
        # Alpha dictates overall clarity of geometry
        alpha = parameters['geometry_intensity'] * 0.8 + 0.2
        final_array = (bg_array * (1 - alpha) + geo_array * alpha).astype(np.uint8)
        
        return final_array

class LightCraftRenderer:
    """The main engine for rendering LightCraft video from SoundCraft audio."""
    def __init__(self, width: int, height: int, fps: int = RENDER_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.visual_generator = VisualGenerator(width, height)
        self.audio_data: np.ndarray = np.array([])
        self.sample_rate: int = 0
        self.duration_s: float = 0.0

    def _load_audio_file(self, audio_filepath: str) -> bool:
        """Loads and prepares the audio file using Librosa."""
        if not os.path.exists(audio_filepath):
             print(f"[ERROR] Audio file not found at '{audio_filepath}'. Using {MOCK_DURATION_S}s fallback simulation data.")
             self.duration_s = MOCK_DURATION_S
             self.sample_rate = 44100
             num_samples = int(self.duration_s * self.sample_rate)
             self.audio_data = np.random.uniform(-0.8, 0.8, num_samples).astype(np.float32)
             return False
        
        # --- Real Librosa Integration ---
        print(f"[LOADER] Loading and analyzing audio file: {audio_filepath}")
        try:
            y, sr = librosa.load(audio_filepath, sr=44100)
            self.audio_data = y
            self.sample_rate = sr
            self.duration_s = librosa.get_duration(y=y, sr=sr)
            
            print(f"  [LOADER] Analysis complete. Duration: {self.duration_s:.2f}s")
            return True
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load audio file using librosa. Is it installed? {e}")
            # Trigger fallback simulation data
            self.duration_s = MOCK_DURATION_S
            self.sample_rate = 44100
            num_samples = int(self.duration_s * self.sample_rate)
            self.audio_data = np.random.uniform(-0.8, 0.8, num_samples).astype(np.float32)
            return False

    def render_video(self, audio_filepath: str, output_path: str):
        """Orchestrates the analysis and visual generation process."""
        
        if not self._load_audio_file(audio_filepath):
            print("[WARNING] Proceeding with simulation data due to file error or missing library.")

        print(f"\n--- LightCraft Renderer Engine Initiated ({self.width}x{self.height} @ {self.fps} FPS) ---")
        
        # Analyzer is instantiated here so it can do its pre-analysis on the loaded audio data
        analyzer = SacredAnalyzer(self.audio_data, self.sample_rate, self.duration_s)
        
        # --- MOVIEPY INTEGRATION POINT ---
        
        def moviepy_make_frame(time_s: float) -> np.ndarray:
            """Closure function for MoviePy to generate a frame at time_s."""
            if int(time_s * self.fps) % (self.fps * 5) == 0: 
                params = analyzer.get_sacred_parameters(time_s)
                print(f"  [PROGRESS] Time {time_s:.2f}s generated. Purity: {params['consonance_score']:.2f}, Layer: {params['geometry_layers']}, Zodiac: {ZODIAC_ARCHETYPES[int(params['astro_phase']) % 12]['sign']}")
            return self.visual_generator.make_frame(time_s, analyzer)

        start_time = time.time()
        
        print("\n  [OPTIMIZATION] Numba JIT compilation will occur on the first few frames...")
        
        # define visual_clip first so it exists for both try paths
        visual_clip = VideoClip(moviepy_make_frame, duration=self.duration_s).set_fps(self.fps)
        
        try:
            # Attempt full video + audio render first
            audio_clip = AudioFileClip(audio_filepath)
            final_clip = visual_clip.set_audio(audio_clip)
            print("\n  [ENCODER] Attempting full LightCraft render (with audio)...")
            final_clip.write_videofile(
                output_path,
                fps=self.fps,
                codec='libx264',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            print(f"--- Generation Complete in {time.time() - start_time:.2f}s ---")
            print(f"  [SAVER] Final video written to '{output_path}'")

        except Exception as audio_error:
            print(f"[AUDIO ERROR] Failed to combine audio and video ({audio_error}). Falling back to visual-only mode.")
            print("\n  [ENCODER] Compiling visual-only video (no audio)...")
            visual_clip.write_videofile(
                output_path.replace('.mp4', '_visual_only.mp4'),
                fps=self.fps,
                codec='libx264',
                verbose=False,
                logger=None
            )
            
            print(f"--- Visual-only generation complete in {time.time() - start_time:.2f}s ---")
            print(f"  [SAVER] Visual-only video written to '{output_path.replace('.mp4', '_visual_only.mp4')}'")
        
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed during video compilation. Ensure 'librosa', 'moviepy', 'numba', and FFMPEG are installed: {e}")
        
        
if __name__ == "__main__":
    
    RESOLUTIONS = {
        1: (1920, 1080, "16:9 (HD/YouTube)"),
        2: (3840, 2160, "16:9 (4K UHD)"),
        3: (1080, 1920, "9:16 (Vertical/Shorts)"),
        4: (1080, 1080, "1:1 (Square/Instagram)"),
    }

    print("\n--- LightCraft Renderer Interactive Setup ---")
    
    # 1. Audio Path (Required)
    audio_path = input("Enter the path to your SoundCraft audio file (e.g., /home/user/music.wav): ").strip()

    # 2. Resolution Selection
    print("\n--- Select Output Resolution ---")
    for key, (w, h, desc) in RESOLUTIONS.items():
        print(f"  [{key}] {w}x{h} ({desc})")
    
    selected_res = None
    while selected_res is None:
        try:
            choice = input("Enter selection number (default 1): ").strip()
            choice_key = int(choice) if choice else 1
            if choice_key in RESOLUTIONS:
                width, height, _ = RESOLUTIONS[choice_key]
                selected_res = (width, height)
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    width, height = selected_res
    
    # 3. Output Path
    output_path = input(f"\nEnter output video filename (default lightcraft_final_render.mp4): ").strip()
    if not output_path:
        output_path = 'lightcraft_final_render.mp4'

    renderer = LightCraftRenderer(width=width, height=height, fps=RENDER_FPS)
    renderer.render_video(audio_filepath=audio_path, output_path=output_path)

