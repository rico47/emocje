import os

# Force Qt to use xcb (X11) backend to avoid Wayland issues
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import yt_dlp
from deepface import DeepFace
import threading
import time # Import time for sleep


class YouTubeEmotionAnalyzer:
    def __init__(self, video_url):
        self.video_url = video_url
        self.current_frame = None
        self.last_emotion = ""
        self.face_region = [0, 0, 0, 0]
        self.running = True

        # Słownik tłumaczeń emocji
        self.emotion_translations = {
            "angry": "złość",
            "disgust": "zniesmaczenie",
            "fear": "strach",
            "happy": "szczescie",
            "sad": "smutek",
            "surprise": "zaskoczenie",
            "neutral": "neutralny"
        }

        # Emocje, które mogą sugerować stres/kłamstwo
        self.stress_emotions = ["fear", "disgust", "angry"]

        # Uzyskanie bezpośredniego linku do strumienia wideo
        self.stream_url = self.get_stream_url(video_url)
        self.cap = cv2.VideoCapture(self.stream_url)

        # Wątek analizy AI
        self.analysis_thread = threading.Thread(target=self.analyze_logic, daemon=True)
        self.analysis_thread.start()

    def get_stream_url(self, url):
        ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']

    def analyze_logic(self):
        while self.running:
            if self.current_frame is not None:
                try:
                    print("Analysing frame...")
                    results = DeepFace.analyze(
                        self.current_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    if results:
                        self.last_emotion = results[0]['dominant_emotion']
                        r = results[0]['region']
                        self.face_region = [r['x'], r['y'], r['w'], r['h']]
                        print(f"Detected emotion: {self.last_emotion}")
                    else:
                        print("No face detected or no emotion results.")
                except Exception as e:
                    print(f"Error during DeepFace analysis: {e}")
            else:
                print("No current frame to analyze yet.")
            time.sleep(0.5)

    def start(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            frame = cv2.resize(frame, (800, 450))
            self.current_frame = frame.copy()

            x, y, w, h = self.face_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.last_emotion:
                polish_emotion = self.emotion_translations.get(self.last_emotion, self.last_emotion)
                
                # Logika wykrywania "kłamstwa" (stresu)
                if self.last_emotion in self.stress_emotions:
                    # Kolor Czerwony dla stresu
                    color = (0, 0, 255) 
                    text = f"Emocja: {polish_emotion} (! STRES/KLAMSTWO ?)"
                else:
                    # Kolor Zielony dla reszty
                    color = (0, 255, 0)
                    text = f"Emocja: {polish_emotion}"

                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('YouTube AI Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Podaj link do dowolnego filmu na YouTube
#yt_url = "https://www.youtube.com/watch?v=BKre0Ntpna0"
yt_url = "https://www.youtube.com/watch?v=QxsKwGfwMBc"
analyzer = YouTubeEmotionAnalyzer(yt_url)
analyzer.start()
