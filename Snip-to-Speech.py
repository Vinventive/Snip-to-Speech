# pyttsx3: used to convert the extracted text into speech

# You'll need the following library installed:
# pip install pytesseract 
# (IMPORTANT: "pytesseract" library requires also a Windows application in order to recognize text from images - use the latest installer from https://github.com/UB-Mannheim/tesseract/wiki#tesseract-installer-for-windows)


import sys
import io
import warnings
import logging
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import pytesseract
import keyboard
import pyttsx3
import threading
import pystray
import pyperclip

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Python 3.12 specific warning
if sys.version_info >= (3, 12):
    logging.warning("This code may not be compatible with Python 3.12 or later.")

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class CaptureWindow(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.callback = callback
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.attributes('-alpha', 0.3, '-topmost', True)
        self.overrideredirect(True)

        self.canvas = tk.Canvas(self, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Escape>", self.on_cancel)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline='red')

    def on_release(self, event):
        if self.start_x is not None and self.start_y is not None:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            self.callback(x1, y1, x2, y2)
        self.destroy()

    def on_cancel(self, event):
        self.destroy()

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snip-to-Speech")
        self.geometry("400x400")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.last_key_press_time = 0
        self.capture_in_progress = False
        self.tts_engine = None
        self.is_hidden = False
        self.icon = None
        self.init_ui()
        self.setup_global_hotkey()
        self.create_tray_icon()
        self.center_window()

    def init_ui(self):
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(self.main_frame, text="Select Voice (Microsoft TTS)").grid(column=0, row=0, sticky=tk.W)
        self.voice_dropdown = ttk.Combobox(self.main_frame, values=self.get_voices())
        self.voice_dropdown.grid(column=0, row=1, sticky=(tk.W, tk.E))
        self.voice_dropdown.set(self.get_voices()[0] if self.get_voices() else "No voices available")

        ttk.Label(self.main_frame, text="Speech Rate").grid(column=0, row=2, sticky=tk.W)
        self.rate_scale = ttk.Scale(self.main_frame, from_=100, to=200, orient=tk.HORIZONTAL, command=self.update_rate_value)
        self.rate_scale.set(150)
        self.rate_scale.grid(column=0, row=3, sticky=(tk.W, tk.E))
        self.rate_value_label = ttk.Label(self.main_frame, text="50%")
        self.rate_value_label.grid(column=1, row=3, sticky=tk.W)

        ttk.Label(self.main_frame, text="Delay time (seconds):").grid(column=0, row=4, sticky=tk.W)
        self.delay_spinbox = ttk.Spinbox(self.main_frame, from_=0, to=30, width=5)
        self.delay_spinbox.set(0)
        self.delay_spinbox.grid(column=1, row=4, sticky=tk.W)

        self.capture_button = ttk.Button(self.main_frame, text="Capture and Speak", command=self.prepare_and_capture)
        self.capture_button.grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E))

        self.text_output = tk.Text(self.main_frame, height=10, width=50)
        self.text_output.grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.copy_to_clipboard_var = tk.BooleanVar()
        self.copy_to_clipboard_checkbox = ttk.Checkbutton(self.main_frame, text="Copy recognized text to clipboard", variable=self.copy_to_clipboard_var)
        self.copy_to_clipboard_checkbox.grid(column=0, row=7, columnspan=2, sticky=tk.W)

        self.tts_enabled_var = tk.BooleanVar(value=True)
        self.tts_enabled_checkbox = ttk.Checkbutton(self.main_frame, text="Enable Text-to-Speech", variable=self.tts_enabled_var)
        self.tts_enabled_checkbox.grid(column=0, row=8, columnspan=2, sticky=tk.W)

        self.minimize_to_tray_var = tk.BooleanVar(value=False)
        self.minimize_to_tray_checkbox = ttk.Checkbutton(self.main_frame, text="Minimize to system tray when closed", variable=self.minimize_to_tray_var)
        self.minimize_to_tray_checkbox.grid(column=0, row=9, columnspan=2, sticky=tk.W)

        for child in self.main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(6, weight=1)

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
        logging.debug("Global hotkey thread started.")

    def run_hotkey_listener(self):
        keyboard.on_press_key('c', self.on_key_press, suppress=False)
        keyboard.wait()  # This keeps the thread alive

    def on_key_press(self, e):
        if keyboard.is_pressed('ctrl'):
            current_time = self.get_current_time()
            if current_time - self.last_key_press_time < 500:  # 500ms threshold for double press
                logging.debug("Double Ctrl+C detected.")
                self.after(0, self.prepare_and_capture)  # Schedule capture in the main thread
            self.last_key_press_time = current_time

    def get_current_time(self):
        return int(self.tk.call('clock', 'milliseconds'))

    def prepare_and_capture(self):
        if self.capture_in_progress:
            logging.debug("Capture is already in progress. Ignoring capture request.")
            return
        self.capture_in_progress = True
        try:
            delay = int(self.delay_spinbox.get())
            logging.debug(f"Starting capture with a delay of {delay} seconds.")
            self.after(delay * 1000, self.show_capture_window)
        except ValueError:
            logging.error("Invalid delay value")
            self.capture_in_progress = False

    def show_capture_window(self):
        self.capture_window = CaptureWindow(self, self.capture_and_speak)
        self.capture_window.focus_force()

    def capture_and_speak(self, x1, y1, x2, y2):
        logging.debug(f"Coordinates captured: ({x1}, {y1}), ({x2}, {y2})")
        voice = self.voice_dropdown.get()
        rate = int(self.rate_scale.get())
        text = self.snip_and_extract_text(x1, y1, x2, y2)
        if text:
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, text)
            logging.debug(f"Extracted text: {text}")
            if self.copy_to_clipboard_var.get():
                try:
                    pyperclip.copy(text)
                    logging.debug("Text copied to clipboard.")
                except Exception as e:
                    logging.error(f"Failed to copy text to clipboard: {e}")
            if self.tts_enabled_var.get():
                threading.Thread(target=self.speak_text, args=(text, voice, rate), daemon=True).start()
        self.capture_in_progress = False

    def on_closing(self):
        if self.minimize_to_tray_var.get():
            self.hide_window()
        else:
            self.exit_application()

    def speak_text(self, text, voice_name, rate):
        logging.debug(f"Starting TTS with voice '{voice_name}' and rate '{rate}'.")
        try:
            if self.tts_engine is None:
                self.tts_engine = pyttsx3.init()
            engine = self.tts_engine
            voices = engine.getProperty('voices')
            voice_found = False
            for voice in voices:
                if voice.name == voice_name:
                    engine.setProperty('voice', voice.id)
                    voice_found = True
                    break
            if not voice_found:
                logging.warning(f"Voice '{voice_name}' not found. Using default voice.")
            engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during text-to-speech: {e}")

    def update_rate_value(self, value):
        try:
            percentage = int(((float(value) - 100) / (200 - 100)) * 100)
            self.rate_value_label.config(text=f"{percentage}%")
            logging.debug(f"Rate slider updated to {percentage}%.")
        except Exception as e:
            logging.error(f"Error updating rate value: {e}")

    def snip_and_extract_text(self, x1, y1, x2, y2):
        try:
            im = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            text = pytesseract.image_to_string(im)
            logging.debug(f"Extracted text from image: {text}")
            return text
        except Exception as e:
            logging.error(f"Error during text extraction: {e}")
            return ""

    def get_voices(self):
        try:
            if self.tts_engine is None:
                self.tts_engine = pyttsx3.init()
            return [voice.name for voice in self.tts_engine.getProperty('voices')]
        except Exception as e:
            logging.error(f"Error getting voices: {e}")
            return []

    def create_tray_icon(self):
        image = Image.new('RGB', (64, 64), color = (73, 109, 137))
        d = ImageDraw.Draw(image)
        d.text((10,10), "S2S", fill=(255,255,0))

        menu = pystray.Menu(
            pystray.MenuItem("Show", self.show_window),
            pystray.MenuItem("Capture", self.prepare_and_capture),
            pystray.MenuItem("Exit", self.exit_application)
        )
        self.icon = pystray.Icon("Snip-to-Speech", image, "Snip-to-Speech", menu)
        threading.Thread(target=self.icon.run, daemon=True).start()

    def show_window(self):
        self.is_hidden = False
        self.deiconify()
        self.lift()
        self.focus_force()

    def hide_window(self):
        self.is_hidden = True
        self.withdraw()
        logging.debug("Window minimized to system tray.")
        # Start a new thread to keep the Tkinter event loop running
        threading.Thread(target=self.keep_alive, daemon=True).start()

    def keep_alive(self):
        while self.is_hidden:
            self.update()
            self.after(100)  # Update every 100ms

    def exit_application(self):
        logging.debug("Exiting application...")
        try:
            if self.tts_engine:
                self.tts_engine.stop()
            keyboard.unhook_all()  # Unhook all keyboard events
            if self.icon:
                self.icon.stop()
            self.quit()
            self.destroy()  # Ensure all Tkinter windows are destroyed
            logging.debug("Application exit successful.")
        except Exception as e:
            logging.error(f"Error during application exit: {e}")
        finally:
            sys.exit(0)  # Force exit the Python process

    def run(self):
        try:
            self.mainloop()
        except Exception as e:
            logging.critical(f"Critical error in main loop: {e}")
            self.exit_application()

if __name__ == '__main__':
    logging.debug("Application started.")
    app = MainApp()
    app.run()
