# sys: used for integrating with the PyQt5 app framework loop
# io: used for caching a snipped image
# time: delay functionality
# PyQt5: used extensively in this code to create the graphical user interface for the screen capture tool
# pytesseract: (OCR) used to extract text from the screenshot captured by the PyQt5 selection tool 
# PIL: used to open and process the image captured from the screen
# pyttsx3: used to convert the extracted text into speech
# gradio: used to create a web-based interface for the app


# You'll need the following libraries installed:
# pip install PyQt5
# pip install gradio
# pip install pyttsx3
# pip install Pillow
# pip install pytesseract 
# (IMPORTANT: "pytesseract" library requires also a Windows application in order to recognize text from images - use the latest installer from https://github.com/UB-Mannheim/tesseract/wiki#tesseract-installer-for-windows)


# Things that might need fixing/adding:
# - Fix "QApplication not created in the main() thread." issue; 

import sys
import io
import time
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QRect, Qt, QBuffer
from PyQt5.QtGui import QPainter, QGuiApplication
import pytesseract
from PIL import Image
import pyttsx3
import gradio as gr

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class CaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.begin = None
        self.end = None
        self.initUI()

    def initUI(self):
        self.setWindowOpacity(0.3)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(QGuiApplication.primaryScreen().virtualGeometry())
        self.show()

    def paintEvent(self, event):
        if self.begin and self.end:
            qp = QPainter(self)
            br = QRect(self.begin, self.end)
            qp.drawRect(br)

    def mousePressEvent(self, event):
        self.begin = event.globalPos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.globalPos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.close()

def get_coordinates(window):
    x1 = min(window.begin.x(), window.end.x())
    y1 = min(window.begin.y(), window.end.y())
    x2 = max(window.begin.x(), window.end.x())
    y2 = max(window.begin.y(), window.end.y())
    return x1, y1, x2, y2

def snip_with_qt(x1, y1, x2, y2):
    screen = QGuiApplication.primaryScreen()
    snip = screen.grabWindow(0, x1, y1, x2 - x1, y2 - y1)
    return snip

def snip_and_extract_text():
    app = QApplication(sys.argv)
    window = CaptureWindow()
    app.exec_()
    x1, y1, x2, y2 = get_coordinates(window)
    snip = snip_with_qt(x1, y1, x2, y2)
    # snip.save("snip_test.png", "PNG")

    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    snip.save(buffer, "PNG")
    pil_im = Image.open(io.BytesIO(buffer.data()))
    text = pytesseract.image_to_string(pil_im)
    return text

def speak_text(text, voice, rate):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for v in voices:
        if v.name == voice:
            engine.setProperty('voice', v.id)
            break
    engine.setProperty('rate', rate)  # Setting the rate (speed)
    engine.say(text)
    engine.runAndWait()

def get_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    return [voice.name for voice in voices]

def snip_and_speak(voice, rate, delay):
    time.sleep(delay)  # Introduce a delay
    text = snip_and_extract_text()
    if text:
        speak_text(text, voice, rate)
    return text

with gr.Blocks(title="Snip-to-Speech") as gui_app:
    gr.Markdown("# <center> Snip-to-Speech </center>")
    voice_dropdown = gr.Dropdown(choices=get_voices(), label="Select Voice (Microsoft TTS)")
    rate_slider = gr.Slider(minimum=100, maximum=200, label="Speed", value=150)

    with gr.Group():
        capture_btn = gr.Button("Snip to Speech")
        delay_slider = gr.Slider(minimum=0, maximum=30, label="Delay (Seconds)", value=3)
        output_text = gr.Text(label="Captured Text (Output)")
        capture_btn.click(snip_and_speak, inputs=[voice_dropdown, rate_slider, delay_slider], outputs=output_text)
        text_input = gr.Textbox(placeholder="Enter text here", label="Custom Text (Input)", lines=2)
        speak_btn = gr.Button("Text to Speech")
        speak_btn.click(speak_text, inputs=[text_input, voice_dropdown, rate_slider], outputs=None)

gui_app.launch()
