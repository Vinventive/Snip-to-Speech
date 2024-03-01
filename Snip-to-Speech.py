# sys: used for integrating with the PyQt5 app framework loop
# io: used for caching a snipped image
# time: delay functionality
# PyQt5: used extensively in this code to create the graphical user interface for the screen capture tool
# pytesseract: (OCR) used to extract text from the screenshot captured by the PyQt5 selection tool 
# PIL: used to open and process the image captured from the screen
# pyttsx3: used to convert the extracted text into speech


# You'll need the following libraries installed:
# pip install PyQt6
# pip install gradio
# pip install pyttsx3
# pip install Pillow
# pip install pytesseract 
# (IMPORTANT: "pytesseract" library requires also a Windows application in order to recognize text from images - use the latest installer from https://github.com/UB-Mannheim/tesseract/wiki#tesseract-installer-for-windows)


import sys
import io
import warnings
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox, QSlider, QSpinBox, QTextEdit, 
                             QHBoxLayout, QSystemTrayIcon, QMenu, QCheckBox, QStyle, QSizePolicy)
from PyQt6.QtCore import QRect, Qt, QBuffer, pyqtSignal, QObject, QTimer, QDateTime, QCoreApplication
from PyQt6.QtGui import QPainter, QGuiApplication, QAction
import pytesseract
from PIL import Image
import pyttsx3


warnings.filterwarnings("ignore", category=DeprecationWarning) # Python 3.12 specific warning
if sys.version_info >= (3, 12):
    print("Warning: This code may not be compatible with Python 3.12 or later.")

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class CaptureFinishedSignal(QObject):
    finished = pyqtSignal(int, int, int, int)

class CaptureWindow(QMainWindow):
    def __init__(self, signal):
        super().__init__()
        self.begin = None
        self.end = None
        self.signal = signal
        self.initUI()

    def initUI(self):
        self.setWindowOpacity(0.3)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(QGuiApplication.primaryScreen().virtualGeometry())
        self.show()

    def paintEvent(self, event):
        if self.begin and self.end:
            qp = QPainter(self)
            br = QRect(self.begin, self.end)
            qp.drawRect(br)

    def mousePressEvent(self, event):
        self.begin = event.globalPosition().toPoint()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.globalPosition().toPoint() 
        self.update()

    def mouseReleaseEvent(self, event):
        x1, y1, x2, y2 = self.get_coordinates()
        self.signal.finished.emit(x1, y1, x2, y2)
        self.close()

    def get_coordinates(self):
        x1 = min(self.begin.x(), self.end.x())
        y1 = min(self.begin.y(), self.end.y())
        x2 = max(self.begin.x(), self.end.x())
        y2 = max(self.begin.y(), self.end.y())
        return x1, y1, x2, y2


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.captureSignal = CaptureFinishedSignal()
        self.setWindowTitle("Snip-to-Speech")
        self.initUI()
        self.initTrayIcon()
        self.ctrlPressed = False
        self.lastCPressTime = None
        self.cPressCount = 0

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Control:
            self.ctrlPressed = True
        elif event.key() == Qt.Key.Key_C and self.ctrlPressed:
            now = QDateTime.currentMSecsSinceEpoch()
            if self.lastCPressTime is None or now - self.lastCPressTime > 500:  # 500 ms threshold for double press
                self.cPressCount = 1
            else:
                self.cPressCount += 1
            self.lastCPressTime = now
            
            if self.cPressCount == 2:
                self.prepareAndCapture()
                self.cPressCount = 0  # Reset count after triggering

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Control:
            self.ctrlPressed = False
            self.cPressCount = 0  # Reset count if Ctrl is released

    def initUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout()

        self.voiceSelectionLabel = QLabel("Select Voice (Microsoft TTS)")
        self.voiceDropdown = QComboBox()
        self.voiceDropdown.addItems(get_voices())

        self.rateSlider = QSlider(Qt.Orientation.Horizontal)
        self.rateSlider.setMinimum(100)
        self.rateSlider.setMaximum(200)
        self.rateSlider.setValue(150)
        self.rateValueLabel = QLabel("50%")
        self.rateSlider.valueChanged.connect(self.updateRateValue)

        voiceRateLayout = QHBoxLayout()
        voiceRateLayout.addWidget(self.rateSlider)
        voiceRateLayout.addWidget(self.rateValueLabel)

        self.delaySpinBox = QSpinBox()
        self.delaySpinBox.setMinimum(0)
        self.delaySpinBox.setMaximum(30)
        self.delaySpinBox.setValue(0)
        self.delaySpinBox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        delayLayout = QHBoxLayout()
        delayLabel = QLabel("Delay time (seconds):")
        delayLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    
        delayLayout.addWidget(delayLabel)
        delayLayout.addWidget(self.delaySpinBox)
        delayLayout.addStretch()

        self.captureButton = QPushButton("Capture and Speak")
        self.captureButton.clicked.connect(self.prepareAndCapture)

        self.textOutput = QTextEdit()
        self.textOutput.setReadOnly(True)

        self.minimizeToTrayCheckBox = QCheckBox("Minimize to tray")

        layout.addWidget(self.minimizeToTrayCheckBox)
        layout.addWidget(self.voiceSelectionLabel)
        layout.addWidget(self.voiceDropdown)

        layout.addLayout(voiceRateLayout)
        layout.addLayout(delayLayout)
        layout.addWidget(self.captureButton)
        layout.addWidget(self.textOutput)

        self.centralWidget.setLayout(layout)
        self.captureSignal.finished.connect(self.captureAndSpeak)

    def initTrayIcon(self):
        self.trayIcon = QSystemTrayIcon(self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.trayIcon.setIcon(icon)
        
        trayMenu = QMenu()
        exitAction = QAction("Exit", self)
        exitAction.triggered.connect(self.exitApplication)
        trayMenu.addAction(exitAction)
        
        self.trayIcon.setContextMenu(trayMenu)
        self.trayIcon.setVisible(True)

        self.trayIcon.activated.connect(self.trayIconActivated)

    def trayIconActivated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.showNormal()
            self.activateWindow()
        
    def closeEvent(self, event):
        if self.minimizeToTrayCheckBox.isChecked():
            event.ignore()
            self.hide()
        else:
            self.exitApplication()

    def exitApplication(self):
        QCoreApplication.quit()

    def updateRateValue(self, value):
        percentage = ((value - 100) / (200 - 100)) * 100
        self.rateValueLabel.setText(f"{percentage:.0f}%")

    def prepareAndCapture(self):
        delay = self.delaySpinBox.value()
        QTimer.singleShot(delay * 1000, self.showCaptureWindow)

    def showCaptureWindow(self):
        self.captureWindow = CaptureWindow(self.captureSignal)
        self.captureWindow.show()

    def captureAndSpeak(self, x1, y1, x2, y2):
        voice = self.voiceDropdown.currentText()
        rate = self.rateSlider.value()
        text = snip_and_extract_text(x1, y1, x2, y2)
        if text:
            speak_text(text, voice, rate)
            self.textOutput.setText(text)


def snip_and_extract_text(x1, y1, x2, y2):
    snip = snip_with_qt(x1, y1, x2, y2)

    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.ReadWrite)
    snip.save(buffer, "PNG")
    pil_im = Image.open(io.BytesIO(buffer.data()))
    text = pytesseract.image_to_string(pil_im)
    return text

def snip_with_qt(x1, y1, x2, y2):
    screen = QGuiApplication.primaryScreen()
    snip = screen.grabWindow(0, x1, y1, x2 - x1, y2 - y1)
    return snip

def speak_text(text, voice_name, rate):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for v in voices:
        if v.name == voice_name:
            engine.setProperty('voice', v.id)
            break
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

def get_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    return [voice.name for voice in voices]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec())
