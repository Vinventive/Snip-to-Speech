# Snip-to-Speech | Florence-2 Integration (WIP)
Snip-to-Speech is an app made in Python that lets you take a snapshot of any part of your computer screen. It pulls out any text from that snapshot and then reads it out loud using Microsoft's default text-to-speech engine. You can use this app to easily extract text from an image or video as well.

![Snip-to-Speech](https://github.com/user-attachments/assets/b3fb0b57-c720-4f41-b712-d722648ffef9)

## UPDATE DETAILS
### Advanced Transformer-Based OCR Model Upgrade
- Replaced the traditional open-source pytesseract OCR with the state-of-the-art Florence-2 model from Microsoft, offering significantly improved text recognition accuracy.

### Enhanced User Interface

- Tooltips: Provides helpful tooltips for all interactive UI elements, improving usability and user experience.
- Image Preview with Zoom and Scroll: View captured images with zoom in/out capabilities and scrollbars for easy navigation (WIP).
- Dynamic Window Sizing: Automatically adjusts the application window size based on screen resolution for optimal display.
- Multiprocessing Support: Utilizes separate processes for OCR and TTS tasks to enhance performance and responsiveness.

### Advanced TTS Controls

- Stop and Replay Buttons: Easily control TTS playback with dedicated buttons to stop ongoing speech or replay the last spoken text.
- Filter Options: Exclude numbers and nicknames (e.g., Twitch usernames) from being read aloud, allowing for a more tailored listening experience.
- Test TTS Functionality: Play a test message to verify and adjust TTS settings before actual use.

### Continuous Mode

- Continuous Text Monitoring: Keep an eye on a selected screen region for real-time text changes (video captions/subtitles, Twitch chat etc.), automatically reading new text as it appears.
- Recognition Log: Maintains a log of recognized sentences, ensuring that each unique sentence is only processed once using fuzzy matching techniques(WIP).

### Improved Global Hotkey Handling

- Robust Shortcut Activation: Enhanced detection of the global hotkey (Ctrl + C + C) to initiate screen capture reliably across different system states.

### System Tray Integration

- Minimize to Tray Option: Choose to minimize the application to the system tray for unobtrusive operation, with easy access to essential controls via the tray icon menu.

### Additional Functionalities

- Reset Application: Quickly reset the application to its initial state, clearing all logs and settings.
- Delay Before Capture: Set a customizable delay before the screen capture begins, allowing you to prepare the desired screen content.
- Clipboard Integration: Optionally copy the recognized text directly to the clipboard for instant copy-and-paste action.
