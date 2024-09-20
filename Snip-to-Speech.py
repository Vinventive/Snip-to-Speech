import sys
import warnings
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import keyboard
import threading
import pystray
import pyttsx3
from pystray import MenuItem as item
import pyperclip
import time
from screeninfo import get_monitors
from rapidfuzz import fuzz
import re  # For regular expression operations
import io  # For BytesIO
import multiprocessing  # For multiprocessing
import mss  # For screen capture across multiple monitors

# Florence-2 imports
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import json
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

from collections import deque  # Import deque for efficient timestamp tracking

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Python 3.12 specific warning
if sys.version_info >= (3, 12):
    logging.warning("This code may not be compatible with Python 3.12 or later.")

# Tooltip Class Implementation
class Tooltip:
    """
    It creates a tooltip for a given widget as the mouse hovers over it.
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     # milliseconds
        self.wraplength = 180   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)
        self.id = None
        self.tw = None

    def on_enter(self, event=None):
        self.schedule()

    def on_leave(self, event=None):
        self.unschedule()
        self.hide_tip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show_tip)

    def unschedule(self):
        _id = self.id
        self.id = None
        if _id:
            self.widget.after_cancel(_id)

    def show_tip(self, event=None):
        # Get the current mouse position
        x = self.widget.winfo_pointerx() + 10
        y = self.widget.winfo_pointery() + 10
        # Creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)  # Removes the window decorations
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

# OCRProcess runs in a separate process for model loading and OCR tasks
class OCRProcess(multiprocessing.Process):
    def __init__(self, request_queue, result_queue):
        super().__init__()
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.loading_model = True  # Indicates if the model is still loading
        self.florence2_model = None
        self.florence2_processor = None
        self.florence2_device = None

    def fixed_get_imports(self, filename):
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

    def run(self):
        # Load the model in this process
        self.load_florence2_model()
        # Inform the main process that the model is loaded
        self.result_queue.put({'type': 'MODEL_LOADED'})
        # Now, enter a loop to process OCR requests
        while True:
            request = self.request_queue.get()
            if request == 'STOP':
                break
            elif request['type'] == 'OCR':
                image_data = request['image_data']
                # Convert image data back to PIL image
                image = Image.open(io.BytesIO(image_data))
                text, ocr_results = self.florence2_ocr(image)
                # Send back the result
                result = {
                    'text': text,
                    'ocr_results': ocr_results,
                }
                self.result_queue.put(result)

    def load_florence2_model(self):
        print('Loading Florence-2 OCR model...')
        model_name = 'microsoft/Florence-2-base'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.florence2_device = device
        with patch("transformers.dynamic_module_utils.get_imports", self.fixed_get_imports):
            self.florence2_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
            self.florence2_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print('Florence-2 OCR model loaded.')
        self.loading_model = False

    def florence2_ocr(self, image):
        device = self.florence2_device
        model = self.florence2_model
        processor = self.florence2_processor
        task_prompt = '<OCR_WITH_REGION>'
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        # Extract the OCR results
        ocr_results_json = parsed_answer.get('<OCR_WITH_REGION>', {})
        ocr_results = []

        # Parse the OCR results
        if isinstance(ocr_results_json, str):
            try:
                ocr_results_data = json.loads(ocr_results_json)
            except json.JSONDecodeError as e:
                print("Error parsing OCR results:", e)
                ocr_results_data = {}
        elif isinstance(ocr_results_json, dict):
            ocr_results_data = ocr_results_json
        else:
            print("Unexpected type for OCR results:", type(ocr_results_json))
            ocr_results_data = {}

        quad_boxes = ocr_results_data.get('quad_boxes', [])
        labels = ocr_results_data.get('labels', [])

        if len(quad_boxes) != len(labels):
            print("Mismatch between number of quad_boxes and labels.")
        else:
            for box, label in zip(quad_boxes, labels):
                # Clean the label to ensure it contains only the recognized text
                clean_label = label.replace('</s>', '').strip()
                # Remove any bounding box coordinates from the label
                match = re.match(r'^(.*?)(?:, Bounding Box:.*)?$', clean_label)
                if match:
                    clean_label = match.group(1).strip()
                ocr_results.append({
                    'quad_boxes': box,
                    'label': clean_label
                })

        # Combine labels into text, ensuring only the recognized text is included
        text = ' '.join([result['label'] for result in ocr_results])
        return text, ocr_results

class TTSProcess(multiprocessing.Process):
    def __init__(self, request_queue):
        super().__init__()
        self.request_queue = request_queue
        self.tts_engine = None

    def run(self):
        import pyttsx3  # Import here to ensure it's imported in the child process
        self.tts_engine = pyttsx3.init()
        while True:
            request = self.request_queue.get()
            if request == 'STOP':
                break
            elif request == 'STOP_PLAYBACK':
                self.stop_playback()
            elif isinstance(request, dict) and 'text' in request:
                text = request['text']
                voice_name = request.get('voice_name', None)
                rate = request.get('rate', None)
                self.speak_text(text, voice_name, rate)
            else:
                # Invalid request
                pass

    def speak_text(self, text, voice_name, rate):
        try:
            engine = self.tts_engine
            voices = engine.getProperty('voices')
            voice_found = False
            if voice_name:
                for voice in voices:
                    if voice.name == voice_name:
                        engine.setProperty('voice', voice.id)
                        voice_found = True
                        logging.debug(f"TTS Engine voice set to: {voice.name}")
                        break
            if not voice_found:
                logging.warning(f"Voice '{voice_name}' not found. Using default voice.")
            if rate:
                engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during text-to-speech: {e}")

    def stop_playback(self):
        try:
            if self.tts_engine:
                self.tts_engine.stop()
                logging.debug("TTS playback stopped.")
        except Exception as e:
            logging.error(f"Error stopping TTS playback: {e}")

class CaptureWindow(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.callback = callback
        self.attributes('-alpha', 0.3, '-topmost', True)
        self.overrideredirect(True)

        # Determine the virtual screen size across all monitors
        self.virtual_width, self.virtual_height, self.min_x, self.min_y = self.get_virtual_screen_size()
        self.geometry(f"{self.virtual_width}x{self.virtual_height}+{self.min_x}+{self.min_y}")

        self.canvas = tk.Canvas(self, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Escape>", self.on_cancel)  # Bind Escape to cancel selection

    def get_virtual_screen_size(self):
        monitors = get_monitors()
        min_x = min(monitor.x for monitor in monitors)
        min_y = min(monitor.y for monitor in monitors)
        max_x = max(monitor.x + monitor.width for monitor in monitors)
        max_y = max(monitor.y + monitor.height for monitor in monitors)
        virtual_width = max_x - min_x
        virtual_height = max_y - min_y
        return virtual_width, virtual_height, min_x, min_y

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x) + self.min_x
        self.start_y = self.canvas.canvasy(event.y) + self.min_y

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x) + self.min_x
        cur_y = self.canvas.canvasy(event.y) + self.min_y

        if self.rect:
            self.canvas.delete(self.rect)
        # Draw rectangle relative to the canvas
        self.rect = self.canvas.create_rectangle(self.start_x - self.min_x, self.start_y - self.min_y,
                                                 cur_x - self.min_x, cur_y - self.min_y, outline='red')

    def on_release(self, event):
        if self.start_x is not None and self.start_y is not None:
            cur_x = self.canvas.canvasx(event.x) + self.min_x
            cur_y = self.canvas.canvasy(event.y) + self.min_y
            x1 = min(self.start_x, cur_x)
            y1 = min(self.start_y, cur_y)
            x2 = max(self.start_x, cur_x)
            y2 = max(self.start_y, cur_y)
            self.callback(x1, y1, x2, y2)
        self.destroy()

    def on_cancel(self, event):
        self.destroy()

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snip-to-Speech")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.20)  # 20% of screen width
        window_height = int(screen_height * 0.6)  # 60% of screen height
        self.geometry(f"{window_width}x{window_height}")  # Set window size dynamically
        self.minsize(256, 576)  # Optional: Set minimum window size
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.capture_in_progress = False
        self.tts_engine = None  # We'll initialize pyttsx3 in speak_text method
        self.is_hidden = False
        self.icon = None
        self.monitoring = False  # Flag to indicate if monitoring is active
        self.selected_region = None  # Store the selected region coordinates
        self.previous_text = ""  # Store the previous recognized text
        self.recognition_log = []  # List to store recognized sentences
        self.preview_photo = None  # To keep a reference to the PhotoImage
        self.original_image = None  # Keep the original image for zooming
        self.current_zoom = 1.0  # Track current zoom level
        self.last_tts_word = None  # Store the last word read by TTS
        self.last_spoken_text = ""  # Store the last spoken text for replay

        # Multiprocessing Queues
        self.request_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.ocr_process = OCRProcess(self.request_queue, self.result_queue)
        self.ocr_process.start()

        # TTS Process
        self.tts_request_queue = multiprocessing.Queue()
        self.tts_process = TTSProcess(self.tts_request_queue)
        self.tts_process.start()

        self.init_ui()
        self.setup_global_hotkey()
        self.create_tray_icon()
        self.center_window()

        # Check if the model is loaded
        self.check_model_loaded()

    def init_ui(self):
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Voice Selection
        ttk.Label(self.main_frame, text="Select Voice (Microsoft TTS)").grid(column=0, row=0, sticky=tk.W)
        self.voice_dropdown = ttk.Combobox(self.main_frame, values=self.get_voices())
        self.voice_dropdown.grid(column=0, row=1, sticky=(tk.W, tk.E))
        if self.get_voices():
            self.voice_dropdown.set(self.get_voices()[0])
        else:
            self.voice_dropdown.set("No voices available")
        Tooltip(self.voice_dropdown, "Choose the voice for text-to-speech.")

        # Speech Rate
        ttk.Label(self.main_frame, text="Speech Rate").grid(column=0, row=2, sticky=tk.W)
        self.rate_scale = tk.Scale(self.main_frame, from_=100, to=200, orient=tk.HORIZONTAL, command=self.update_rate_value)
        self.rate_scale.set(150)
        self.rate_scale.grid(column=0, row=3, sticky=(tk.W, tk.E))
        self.rate_value_label = ttk.Label(self.main_frame, text="50%")
        self.rate_value_label.grid(column=1, row=3, sticky=tk.W)
        Tooltip(self.rate_scale, "Adjust the speed of the speech.")

        # Delay Time
        ttk.Label(self.main_frame, text="Delay time (seconds):").grid(column=0, row=4, sticky=tk.W)
        self.delay_spinbox = ttk.Spinbox(self.main_frame, from_=0, to=30, width=5)
        self.delay_spinbox.set(0)
        self.delay_spinbox.grid(column=1, row=4, sticky=tk.W)
        Tooltip(self.delay_spinbox, "Set a delay before capturing the screen.")

        # Capture Button
        self.capture_button = ttk.Button(self.main_frame, text="Capture and Speak", command=self.prepare_and_capture)
        self.capture_button.grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E))
        self.capture_button.config(state=tk.DISABLED)  # Disable until model is loaded
        Tooltip(self.capture_button, "Capture a region of the screen and speak the recognized text.")

        # **Added "Stop" Button**
        self.stop_button = ttk.Button(self.main_frame, text="Stop", command=self.stop_tts)
        self.stop_button.grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        self.stop_button.config(state=tk.DISABLED)  # Initially disabled
        Tooltip(self.stop_button, "Stop the ongoing text-to-speech playback.")

        # **Added "Replay" Button**
        self.replay_button = ttk.Button(self.main_frame, text="Replay", command=self.replay_tts)
        self.replay_button.grid(column=0, row=7, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 10))
        self.replay_button.config(state=tk.DISABLED)  # Initially disabled
        Tooltip(self.replay_button, "Replay the last spoken text.")

        # Toggle Visibility Checkboxes
        self.show_text_var = tk.BooleanVar(value=True)
        self.show_text_checkbox = ttk.Checkbutton(
            self.main_frame, text="Show Recognized Text", variable=self.show_text_var,
            command=self.toggle_recognized_text
        )
        self.show_text_checkbox.grid(column=0, row=8, columnspan=2, sticky=tk.W)
        Tooltip(self.show_text_checkbox, "Toggle the visibility of the recognized text section.")

        self.show_image_var = tk.BooleanVar(value=True)
        self.show_image_checkbox = ttk.Checkbutton(
            self.main_frame, text="Show Image Preview", variable=self.show_image_var,
            command=self.toggle_image_preview
        )
        self.show_image_checkbox.grid(column=0, row=9, columnspan=2, sticky=tk.W)
        Tooltip(self.show_image_checkbox, "Toggle the visibility of the image preview section.")

        # Text Output with Scrolling
        self.text_output_frame = ttk.Frame(self.main_frame)
        self.text_output_frame.grid(column=0, row=10, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.text_output_scrollbar = ttk.Scrollbar(self.text_output_frame)
        self.text_output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output = tk.Text(self.text_output_frame, height=1, width=100, yscrollcommand=self.text_output_scrollbar.set)
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_output_scrollbar.config(command=self.text_output.yview)
        Tooltip(self.text_output, "Displays the text recognized from the captured image.")

        # Image Preview with Canvas, Scrollbars, and Zoom
        self.image_preview_frame = ttk.Frame(self.main_frame, padding="10")
        self.image_preview_frame.grid(column=0, row=11, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.image_preview_frame.columnconfigure(0, weight=1)
        self.image_preview_frame.rowconfigure(0, weight=1)

        # Create Canvas
        self.canvas = tk.Canvas(self.image_preview_frame, bg='grey')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # Add Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.image_preview_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.v_scrollbar = ttk.Scrollbar(self.image_preview_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')

        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        # Bind mouse wheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Zoom Controls
        zoom_frame = ttk.Frame(self.main_frame)
        zoom_frame.grid(column=0, row=12, columnspan=2, pady=(5,0), sticky=tk.E)
        self.zoom_in_button = ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom_image(1.25))
        self.zoom_in_button.grid(column=0, row=0, padx=(0,5))
        Tooltip(self.zoom_in_button, "Zoom into the image preview.")
        self.zoom_out_button = ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom_image(0.8))
        self.zoom_out_button.grid(column=1, row=0)
        Tooltip(self.zoom_out_button, "Zoom out of the image preview.")

        # Image Preview Photo Reference
        self.image_on_canvas = None
        self.preview_photo = None

        # Copy to Clipboard
        self.copy_to_clipboard_var = tk.BooleanVar()
        self.copy_to_clipboard_checkbox = ttk.Checkbutton(
            self.main_frame, text="Copy recognized text to clipboard", variable=self.copy_to_clipboard_var)
        self.copy_to_clipboard_checkbox.grid(column=0, row=13, columnspan=2, sticky=tk.W)
        Tooltip(self.copy_to_clipboard_checkbox, "Automatically copy the recognized text to the clipboard.")

        # Enable TTS
        self.tts_enabled_var = tk.BooleanVar(value=True)
        self.tts_enabled_checkbox = ttk.Checkbutton(
            self.main_frame, text="Enable Text-to-Speech", variable=self.tts_enabled_var)
        self.tts_enabled_checkbox.grid(column=0, row=14, columnspan=2, sticky=tk.W)
        Tooltip(self.tts_enabled_checkbox, "Enable or disable the text-to-speech functionality.")

        # Don't Read Numbers
        self.ignore_numbers_var = tk.BooleanVar(value=True)
        self.ignore_numbers_checkbox = ttk.Checkbutton(
            self.main_frame, text="Don't read numbers", variable=self.ignore_numbers_var)
        self.ignore_numbers_checkbox.grid(column=0, row=15, columnspan=2, sticky=tk.W)
        Tooltip(self.ignore_numbers_checkbox, "Exclude numbers from the text being read aloud.")

        # Don't Read Nicknames
        self.ignore_nicknames_var = tk.BooleanVar(value=True)
        self.ignore_nicknames_checkbox = ttk.Checkbutton(
            self.main_frame, text="Don't read nicknames (Twitch)", variable=self.ignore_nicknames_var)
        self.ignore_nicknames_checkbox.grid(column=0, row=16, columnspan=2, sticky=tk.W)
        Tooltip(self.ignore_nicknames_checkbox, "Exclude nicknames (e.g., Twitch usernames) from the text being read aloud.")

        # Test TTS Button
        self.test_tts_button = ttk.Button(self.main_frame, text="Test TTS", command=self.test_tts)
        self.test_tts_button.grid(column=0, row=17, columnspan=2, sticky=(tk.W, tk.E))
        Tooltip(self.test_tts_button, "Play a test message to verify TTS settings.")

        # Reset Button
        self.reset_button = ttk.Button(self.main_frame, text="Reset App", command=self.reset_app)
        self.reset_button.grid(column=0, row=18, columnspan=2, sticky=(tk.W, tk.E))
        Tooltip(self.reset_button, "Reset the application to its initial state.")

        # Minimize to Tray
        self.minimize_to_tray_var = tk.BooleanVar(value=False)
        self.minimize_to_tray_checkbox = ttk.Checkbutton(
            self.main_frame, text="Minimize to system tray when closed", variable=self.minimize_to_tray_var)
        self.minimize_to_tray_checkbox.grid(column=0, row=19, columnspan=2, sticky=tk.W)
        Tooltip(self.minimize_to_tray_checkbox, "Minimize the application to the system tray instead of closing it.")

        # Monitoring Button
        self.monitor_button = ttk.Button(self.main_frame, text="Start Monitoring", command=self.start_monitoring)
        self.monitor_button.grid(column=0, row=20, columnspan=2, sticky=(tk.W, tk.E))
        self.monitor_button.config(state=tk.DISABLED)  # Initially disabled
        Tooltip(self.monitor_button, "Continuously monitor the selected screen region for text changes.")

        # Monitoring Status
        self.monitor_status_label = ttk.Label(self.main_frame, text="Monitoring Status: Inactive")
        self.monitor_status_label.grid(column=0, row=21, columnspan=2, sticky=tk.W)
        Tooltip(self.monitor_status_label, "Displays whether monitoring is active or inactive.")

        for child in self.main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Configure grid weights to make widgets responsive
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(10, weight=1)
        self.main_frame.rowconfigure(11, weight=1)

    def _on_mousewheel(self, event):
        """Enable scrolling with mouse wheel on the canvas."""
        if event.state & 0x1:  # If Shift is held, scroll horizontally
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def zoom_image(self, factor):
        """Zoom in or out of the image on the canvas."""
        if self.original_image:
            self.current_zoom *= factor
            new_size = (int(self.original_image.width * self.current_zoom),
                        int(self.original_image.height * self.current_zoom))
            resized_image = self.original_image.resize(new_size, Image.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(resized_image)
            self.canvas.delete("all")
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.preview_photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            logging.debug(f"Image zoomed to {self.current_zoom * 100:.0f}%.")

    def toggle_recognized_text(self):
        """Show or hide the recognized text section."""
        if self.show_text_var.get():
            self.text_output_frame.grid()
            logging.debug("Recognized text section shown.")
        else:
            self.text_output_frame.grid_remove()
            logging.debug("Recognized text section hidden.")

    def toggle_image_preview(self):
        """Show or hide the image preview section."""
        if self.show_image_var.get():
            self.image_preview_frame.grid()
            logging.debug("Image preview section shown.")
        else:
            self.image_preview_frame.grid_remove()
            logging.debug("Image preview section hidden.")

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def setup_global_hotkey(self):
        self.hotkey_thread = threading.Thread(target=self.run_hotkey_listener, daemon=True)
        self.hotkey_thread.start()
        logging.debug("Global hotkey thread started with 'Ctrl + c + c'.")

    def run_hotkey_listener(self):
        ctrl_pressed = False
        c_press_times = deque(maxlen=2)  # Store timestamps of the last two 'c' presses

        def on_press(event):
            nonlocal ctrl_pressed
            if event.name == 'ctrl':
                ctrl_pressed = True

            if ctrl_pressed and event.name == 'c':
                current_time = time.time()
                c_press_times.append(current_time)
                if len(c_press_times) == 2:
                    # Check if the two 'c' presses are within 0.5 seconds
                    if c_press_times[1] - c_press_times[0] <= 0.5:
                        # Trigger the capture action
                        self.prepare_and_capture()
                        # Clear the deque to prevent multiple triggers
                        c_press_times.clear()

        def on_release(event):
            nonlocal ctrl_pressed
            if event.name == 'ctrl':
                ctrl_pressed = False

        # Register the event hooks
        keyboard.on_press(on_press)
        keyboard.on_release(on_release)

        keyboard.wait()  # Keep the listener running

    def prepare_and_capture(self):
        if self.capture_in_progress:
            logging.debug("Capture is already in progress. Ignoring capture request.")
            return
        self.capture_in_progress = True
        try:
            delay = int(self.delay_spinbox.get())
            logging.debug(f"Starting capture with a delay of {delay} seconds.")
            if delay > 0:
                threading.Timer(delay, lambda: self.after(0, self.show_capture_window)).start()
            else:
                self.after(0, self.show_capture_window)
        except ValueError:
            logging.error("Invalid delay value")
            self.capture_in_progress = False

    def show_capture_window(self):
        self.capture_window = CaptureWindow(self, self.capture_and_speak)
        self.capture_window.focus_force()

    def capture_and_speak(self, x1, y1, x2, y2):
        logging.debug(f"Coordinates captured: ({x1}, {y1}), ({x2}, {y2})")
        self.selected_region = (x1, y1, x2, y2)  # Store selected region
        voice = self.voice_dropdown.get()
        rate = int(self.rate_scale.get())
        # Capture the image using mss
        try:
            with mss.mss() as sct:
                monitor = {'left': int(x1), 'top': int(y1), 'width': int(x2 - x1), 'height': int(y2 - y1)}
                sct_img = sct.grab(monitor)
                im = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='PNG')
                img_data = img_byte_arr.getvalue()
                # Send OCR request
                request = {'type': 'OCR', 'image_data': img_data}
                self.request_queue.put(request)
                # Wait for result
                self.after(0, self.check_initial_ocr_result, im, voice, rate)
        except Exception as e:
            logging.error(f"Error capturing image: {e}")
            self.capture_in_progress = False

    def check_initial_ocr_result(self, im, voice, rate):
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result.get('text') is not None:
                    text = result['text']
                    ocr_results = result['ocr_results']
                    image_with_boxes = self.draw_ocr_bboxes(im.copy(), ocr_results)
                    # Now process the text as before
                    if text:
                        self.text_output.delete(1.0, tk.END)
                        self.text_output.insert(tk.END, text)
                        logging.debug(f"Extracted text: {text}")
                        self.previous_text = text  # Initialize previous text
                        self.add_to_recognition_log(text)
                        if self.copy_to_clipboard_var.get():
                            try:
                                pyperclip.copy(text)
                                logging.debug("Text copied to clipboard.")
                            except Exception as e:
                                logging.error(f"Failed to copy text to clipboard: {e}")
                        if self.tts_enabled_var.get():
                            # Adjust text to read based on last TTS word
                            adjusted_text = self.adjust_text_based_on_last_word(text)
                            # Speak the adjusted text on the main thread
                            self.speak_text(adjusted_text, voice, rate)
                        # Update image preview with highlighted text
                        self.update_image_preview(image_with_boxes)
                        # Enable the monitoring button
                        self.monitor_button.config(state=tk.NORMAL)
                        # **Enable "Stop" and "Replay" buttons only if not monitoring**
                        if not self.monitoring:
                            self.stop_button.config(state=tk.NORMAL)
                            self.replay_button.config(state=tk.NORMAL)
                        logging.debug("Capture and Speak completed.")
                    else:
                        logging.debug("No text extracted.")
                    self.capture_in_progress = False
                else:
                    # Unexpected result
                    logging.error("Unexpected OCR result received.")
                    self.capture_in_progress = False
            else:
                # Result not ready yet, check again after a short delay
                self.after(100, self.check_initial_ocr_result, im, voice, rate)
        except Exception as e:
            logging.error(f"Error checking OCR result: {e}")
            self.capture_in_progress = False

    def test_tts(self):
        if self.tts_enabled_var.get():
            voice = self.voice_dropdown.get()
            rate = int(self.rate_scale.get())
            test_text = "This is a test of the text to speech functionality."
            self.after(0, self.speak_text, test_text, voice, rate)
            logging.debug("Test TTS triggered.")
        else:
            logging.debug("TTS is disabled. Test TTS not performed.")

    def add_to_recognition_log(self, text):
        sentences = [sentence.strip() for sentence in text.split('\n') if sentence.strip()]
        for sentence in sentences:
            if not self.is_sentence_recognized(sentence):
                self.recognition_log.append(sentence)
                logging.debug(f"Added to recognition log: {sentence}")
            else:
                logging.debug(f"Sentence already recognized: {sentence}")

    def is_sentence_recognized(self, new_sentence, threshold=85):
        for recognized in self.recognition_log:
            similarity = fuzz.partial_ratio(new_sentence.lower(), recognized.lower())
            if similarity >= threshold:
                logging.debug(f"Similarity between '{new_sentence}' and '{recognized}': {similarity}%")
                return True
        return False

    def start_monitoring(self):
        if not self.monitoring:
            if not self.selected_region:
                messagebox.showwarning("No Region Selected", "Please capture a screen region first.")
                return
            # Start a new TTSProcess if not alive
            if self.tts_process is None or not self.tts_process.is_alive():
                self.tts_request_queue = multiprocessing.Queue()
                self.tts_process = TTSProcess(self.tts_request_queue)
                self.tts_process.start()
                logging.debug("Started a new TTS process.")
            self.monitoring = True
            self.monitor_status_label.config(text="Monitoring Status: Active")
            self.monitor_button.config(text="Stop Monitoring", command=self.stop_monitoring)
            # **Disable "Stop" and "Replay" buttons during monitoring**
            self.stop_button.config(state=tk.DISABLED)
            self.replay_button.config(state=tk.DISABLED)
            self.monitor_screen()
            logging.debug("Started monitoring the selected region for text changes.")

    def stop_monitoring(self):
        if self.monitoring:
            self.monitoring = False
            self.monitor_status_label.config(text="Monitoring Status: Inactive")
            self.monitor_button.config(text="Start Monitoring", command=self.start_monitoring)
            # Stop any ongoing TTS playback
            if self.tts_process and self.tts_process.is_alive():
                self.tts_request_queue.put('STOP_PLAYBACK')
                # Wait a short time to see if TTSProcess stops playback
                time.sleep(0.5)
                # Optionally, we can terminate the TTSProcess if needed
                if self.tts_process.is_alive():
                    logging.warning("TTS playback did not stop. Terminating TTS process.")
                    self.tts_process.terminate()
                    self.tts_process.join()
                # Reset TTS process and queue
                self.tts_process = None
                self.tts_request_queue = None
            else:
                logging.warning("TTS process is not alive. Cannot stop playback.")
                self.tts_process = None
                self.tts_request_queue = None
            # **Disable "Stop" and "Replay" buttons after stopping monitoring**
            self.stop_button.config(state=tk.DISABLED)
            self.replay_button.config(state=tk.DISABLED)
            logging.debug("Stopped monitoring the selected region and requested TTS playback to stop.")

    def monitor_screen(self):
        if not self.monitoring:
            return
        try:
            x1, y1, x2, y2 = self.selected_region
            # Capture current screen using mss
            with mss.mss() as sct:
                monitor = {'left': int(x1), 'top': int(y1), 'width': int(x2 - x1), 'height': int(y2 - y1)}
                sct_img = sct.grab(monitor)
                im = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='PNG')
                img_data = img_byte_arr.getvalue()
                # Send OCR request
                request = {'type': 'OCR', 'image_data': img_data}
                # Clear queues if they get too large to prevent overflow
                if self.request_queue.qsize() > 5:
                    logging.warning("Request queue size exceeded limit. Clearing queue.")
                    while not self.request_queue.empty():
                        self.request_queue.get_nowait()
                self.request_queue.put(request)
                # Schedule a method to check for the result
                self.after(0, self.check_ocr_result, im)
        except Exception as e:
            logging.error(f"Error during monitoring: {e}")
            self.stop_monitoring()

    def check_ocr_result(self, im):
        if not self.monitoring:
            return
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result.get('text') is not None:
                    text = result['text']
                    ocr_results = result['ocr_results']
                    # Process the text
                    # Get the incremental new text
                    incremental_text = self.get_incremental_text(self.previous_text, text)
                    # Update previous text
                    self.previous_text = text
                    if incremental_text:
                        # Highlight similar word sequences
                        self.update_text_output_with_highlight(text)
                        # Check if we have a complete sentence
                        complete_sentences = self.get_complete_sentences(incremental_text)
                        if complete_sentences and not self.is_sentence_recognized(complete_sentences):
                            self.add_to_recognition_log(complete_sentences)
                            if self.copy_to_clipboard_var.get():
                                try:
                                    pyperclip.copy(complete_sentences)
                                    logging.debug("Complete sentences copied to clipboard.")
                                except Exception as e:
                                    logging.error(f"Failed to copy text to clipboard: {e}")
                            if self.tts_enabled_var.get():
                                voice = self.voice_dropdown.get()
                                rate = int(self.rate_scale.get())
                                # Adjust text to read based on last TTS word
                                adjusted_text = self.adjust_text_based_on_last_word(complete_sentences)
                                # Speak the adjusted text on the main thread
                                self.speak_text(adjusted_text, voice, rate)
                    else:
                        # No new text; update the text output to show current recognized text
                        self.update_text_output_with_highlight(text)
                    # Overlay bounding boxes on the original image
                    image_with_boxes = self.draw_ocr_bboxes(im.copy(), ocr_results)
                    # Update image preview in the main thread
                    self.update_image_preview(image_with_boxes)
                else:
                    # Unexpected result
                    logging.error("Unexpected OCR result received.")
                # Schedule the next monitoring cycle
                self.after(300, self.monitor_screen)
            else:
                # Result not ready yet, check again after a short delay
                self.after(100, self.check_ocr_result, im)
        except Exception as e:
            logging.error(f"Error checking OCR result: {e}")
            # Instead of stopping monitoring, continue to next cycle
            self.after(300, self.monitor_screen)

    def adjust_text_based_on_last_word(self, text):
        words = text.strip().split()
        if not words:
            return ""
        if self.last_tts_word:
            # Find the index of the last TTS word in the current text
            try:
                index = words.index(self.last_tts_word) + 1
                adjusted_words = words[index:]
            except ValueError:
                # If last TTS word not found, read the entire text
                adjusted_words = words
        else:
            adjusted_words = words
        adjusted_text = ' '.join(adjusted_words)
        logging.debug(f"Adjusted text to read: '{adjusted_text}'")
        return adjusted_text

    def get_incremental_text(self, previous_text, current_text):
        # Compare previous and current text to find new additions
        similarity = fuzz.partial_ratio(previous_text.lower(), current_text.lower())
        logging.debug(f"Text similarity: {similarity}%")
        if similarity >= 99:
            # Texts are almost identical
            return None
        else:
            # Find the divergence point
            prev_words = previous_text.strip().split()
            curr_words = current_text.strip().split()
            index = 0
            while index < len(prev_words) and index < len(curr_words) and prev_words[index].lower() == curr_words[index].lower():
                index += 1
            new_words = curr_words[index:]
            if new_words:
                incremental_text = ' '.join(new_words)
                logging.debug(f"Incremental text: '{incremental_text}'")
                return incremental_text
            else:
                return None

    def update_text_output_with_highlight(self, full_text):
        self.text_output.delete(1.0, tk.END)
        # Highlight the matching part
        self.text_output.tag_configure("matched", foreground="blue")
        self.text_output.tag_configure("new", foreground="green")

        # Split texts into words
        prev_words = self.previous_text.strip().split()
        curr_words = full_text.strip().split()

        match_length = 0
        while match_length < len(prev_words) and match_length < len(curr_words) and prev_words[match_length].lower() == curr_words[match_length].lower():
            match_length += 1

        for i, word in enumerate(curr_words):
            if i < match_length:
                self.text_output.insert(tk.END, word + ' ', "matched")
            else:
                self.text_output.insert(tk.END, word + ' ', "new")

    def get_complete_sentences(self, text):
        # Check for sentence-ending punctuation
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                sentences.append(line)
        if sentences:
            return '\n'.join(sentences)
        return None

    def on_closing(self):
        if self.minimize_to_tray_var.get():
            self.hide_window()
        else:
            self.exit_application()

    def speak_text(self, text, voice_name, rate):
        if not text.strip():
            logging.debug("No text to speak.")
            return
        # Remove unwanted symbols before speaking
        text = self.remove_unwanted_symbols(text)
        if not text.strip():
            logging.debug("No text to speak after removing unwanted symbols.")
            return

        # Apply additional filters based on checkboxes
        if self.ignore_numbers_var.get():
            text = self.remove_numbers(text)
            logging.debug(f"Text after removing numbers: '{text}'")
        if self.ignore_nicknames_var.get():
            text = self.remove_nicknames(text)
            logging.debug(f"Text after removing nicknames: '{text}'")

        if not text.strip():
            logging.debug("No text to speak after applying filters.")
            return

        # Update last_tts_word
        words = text.strip().split()
        if words:
            self.last_tts_word = words[-1]
            logging.debug(f"Last TTS word updated to: '{self.last_tts_word}'")
            # **Reset last_tts_word to None if not monitoring**
            if not self.monitoring:
                self.last_tts_word = None
                logging.debug("last_tts_word reset to None after single capture.")

        # Store the last spoken text for replay
        self.last_spoken_text = text

        # **Enable "Replay" button since there is text to replay**
        if not self.monitoring:
            self.replay_button.config(state=tk.NORMAL)

        # Check if TTS process is alive
        if self.tts_process is None or not self.tts_process.is_alive():
            # Start a new TTSProcess
            self.tts_request_queue = multiprocessing.Queue()
            self.tts_process = TTSProcess(self.tts_request_queue)
            self.tts_process.start()
            logging.debug("Started a new TTS process.")

        # Send the text to the TTS process
        request = {'text': text, 'voice_name': voice_name, 'rate': rate}
        self.tts_request_queue.put(request)
        logging.debug(f"Sent text to TTS process: '{text}'")

    def remove_numbers(self, text):
        # Remove all digits from the text
        return re.sub(r'\d+', '', text)

    def remove_nicknames(self, text):
        # Remove any 'nickname:' patterns, regardless of their position
        # This removes substrings like 'User1: ', 'Nickname: ', etc.
        return re.sub(r'\b\S+?:\s*', '', text)

    def remove_unwanted_symbols(self, text):
        # Define unwanted symbols
        unwanted_symbols = '£$€¥¢₤₵₩₭₦₨₱₲₳₴₰₯₡₢₣₤₥₦₧₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₺₻₼₽₾₿@#%^&*()[]{}<>/\\|~`_=+'
        # Use regular expression to remove unwanted symbols
        pattern = f"[{re.escape(unwanted_symbols)}]"
        cleaned_text = re.sub(pattern, '', text)
        logging.debug(f"Text after removing unwanted symbols: '{cleaned_text}'")
        return cleaned_text

    def update_rate_value(self, value):
        try:
            percentage = int(((float(value) - 100) / (200 - 100)) * 100)
            self.rate_value_label.config(text=f"{percentage}%")
            logging.debug(f"Rate slider updated to {percentage}%.")
        except Exception as e:
            logging.error(f"Error updating rate value: {e}")

    def draw_ocr_bboxes(self, image, ocr_results):
        draw = ImageDraw.Draw(image)
        for result in ocr_results:
            if isinstance(result, dict) and 'quad_boxes' in result and 'label' in result:
                box = result['quad_boxes']
                label = result['label']
                try:
                    if isinstance(box, list) and len(box) == 8:
                        # Convert flat list to list of (x, y) tuples
                        box = list(zip(box[::2], box[1::2]))
                        draw.polygon(box, outline='lime', width=2)
                        # Calculate the position for the label
                        min_x = min(point[0] for point in box)
                        min_y = min(point[1] for point in box)
                        draw.text((min_x, min_y - 10), label, fill='red')
                    else:
                        print("Invalid quad_boxes format:", box)
                except Exception as e:
                    print(f"Error drawing box for result {result}: {e}")
            else:
                print("Invalid OCR result format:", result)
        return image

    def get_voices(self):
        try:
            temp_engine = pyttsx3.init()
            voices = [voice.name for voice in temp_engine.getProperty('voices')]
            temp_engine.stop()
            return voices
        except Exception as e:
            logging.error(f"Error getting voices: {e}")
            return []

    def update_image_preview(self, pil_image):
        try:
            self.original_image = pil_image.copy()  # Keep the original image for zooming
            self.current_zoom = 1.0  # Reset zoom level
            # Get the size of the image preview label
            self.canvas.delete("all")  # Clear previous image
            self.preview_photo = ImageTk.PhotoImage(pil_image)
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.preview_photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            logging.debug("Image preview updated.")
        except Exception as e:
            logging.error(f"Error updating image preview: {e}")

    def create_tray_icon(self):
        image = Image.new('RGB', (64, 64), color=(73, 109, 137))
        d = ImageDraw.Draw(image)
        d.text((10, 25), "S2S", fill=(255, 255, 0))

        menu = pystray.Menu(
            item("Show", self.show_window),
            item("Reset", self.reset_app),
            item("Exit", self.exit_application)
        )
        self.icon = pystray.Icon("Snip-to-Speech", image, "Snip-to-Speech\n[Ctrl + C + C]", menu)
        threading.Thread(target=self.icon.run, daemon=True).start()

    def show_window(self, icon=None, item=None):
        self.is_hidden = False
        self.deiconify()
        self.lift()
        self.focus_force()
        logging.debug("Window shown from system tray.")

    def hide_window(self):
        self.is_hidden = True
        self.withdraw()
        logging.debug("Window minimized to system tray.")

    def exit_application(self, icon=None, item=None):
        logging.debug("Exiting application...")
        try:
            keyboard.unhook_all()  # Unhook all keyboard events
            if self.icon:
                self.icon.stop()
            self.monitoring = False  # Stop monitoring if active
            # Stop the OCR process
            self.request_queue.put('STOP')
            self.ocr_process.join()
            # Stop the TTS process
            if self.tts_process and self.tts_process.is_alive():
                self.tts_request_queue.put('STOP')
                self.tts_process.join()
            self.quit()
            self.destroy()  # Ensure all Tkinter windows are destroyed
            logging.debug("Application exit successful.")
        except Exception as e:
            logging.error(f"Error during application exit: {e}")
        finally:
            sys.exit(0)  # Force exit the Python process

    def reset_app(self):
        logging.debug("Resetting application to initial state.")
        # Stop monitoring if active
        if self.monitoring:
            self.stop_monitoring()

        # Reset all variables and GUI components
        self.text_output.delete(1.0, tk.END)
        self.canvas.delete("all")
        self.preview_photo = None
        self.original_image = None
        self.previous_text = ""
        self.recognition_log = []
        self.last_tts_word = None
        self.last_spoken_text = ""
        self.capture_in_progress = False

        # Reset monitor button
        self.monitor_button.config(state=tk.DISABLED)
        self.monitor_status_label.config(text="Monitoring Status: Inactive")
        self.monitor_button.config(text="Start Monitoring", command=self.start_monitoring)

        # **Disable "Stop" and "Replay" buttons**
        self.stop_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)

        # Reset visibility checkboxes
        self.show_text_var.set(True)
        self.show_image_var.set(True)
        self.toggle_recognized_text()
        self.toggle_image_preview()

        logging.debug("Application reset completed.")

    def check_model_loaded(self):
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result.get('type') == 'MODEL_LOADED':
                    self.capture_button.config(state=tk.NORMAL)
                    logging.debug("Florence-2 model loaded. Capture button enabled.")
                    return
            self.after(100, self.check_model_loaded)
        except Exception as e:
            logging.error(f"Error checking model status: {e}")
            self.after(100, self.check_model_loaded)

    def run(self):
        try:
            self.mainloop()
        except Exception as e:
            logging.critical(f"Critical error in main loop: {e}")
            self.exit_application()

    # **New Method: stop_tts**
    def stop_tts(self):
        try:
            if self.tts_process and self.tts_process.is_alive():
                self.tts_request_queue.put('STOP')
                self.tts_process.terminate()
                self.tts_process.join()
                self.tts_process = None
                self.tts_request_queue = multiprocessing.Queue()
                logging.debug("TTS process stopped via Stop button.")
                # Disable "Stop" and "Replay" buttons after stopping
                self.stop_button.config(state=tk.DISABLED)
                self.replay_button.config(state=tk.DISABLED)
            else:
                logging.debug("No active TTS process to stop.")
        except Exception as e:
            logging.error(f"Error stopping TTS process: {e}")

    # **New Method: replay_tts**
    def replay_tts(self):
        if not self.last_spoken_text.strip():
            logging.debug("No text available to replay.")
            return
        if self.monitoring:
            logging.debug("Cannot replay during monitoring mode.")
            return
        voice = self.voice_dropdown.get()
        rate = int(self.rate_scale.get())
        text = self.last_spoken_text
        self.speak_text(text, voice, rate)
        logging.debug("Replay TTS triggered.")

if __name__ == '__main__':
    logging.debug("Application started.")
    multiprocessing.freeze_support()  # For Windows support
    app = MainApp()
    app.run()
