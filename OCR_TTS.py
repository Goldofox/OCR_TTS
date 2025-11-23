import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output
import numpy as np
import threading
import queue
import pyttsx3
from collections import deque
import time

# =========================
# CONFIGURATION OCR
# =========================
OCR_LANG = "fra"          # Langue pour Tesseract
OCR_OEM = 3               # 3 = LSTM
OCR_CONF_MIN = 60         # Confiance minimale pour considérer un mot valide
OCR_FRAME_SKIP = 6        # OCR toutes les N frames pour garder du FPS

# =========================
# CONFIGURATION TTS
# =========================
TTS_MIN_CHARS = 3         # Minimum de caractères pour prononcer
TTS_COOLDOWN = 0          # Temps minimal entre deux énoncés
TTS_MAX_CHARS = 160       # Nombre max de caractères lus en une fois

# =========================
# TTS NON BLOQUANT
# =========================
class NonBlockingTTS:
    """Thread dédié pour TTS afin que l'OCR continue de tourner pendant que le texte est lu."""
    def __init__(self, lang_hint="fra"):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.engine = pyttsx3.init()
        try:
            self.engine.setProperty('rate', 170)
        except:
            pass

        # Sélection d'une voix française si disponible
        try:
            if "fra" in lang_hint:
                for voice in self.engine.getProperty('voices'):
                    name = (voice.name or "").lower()
                    langs = "".join([str(x).lower() for x in getattr(voice, "languages", [])])
                    if "fr" in name or "french" in name or "fr" in langs:
                        self.engine.setProperty('voice', voice.id)
                        break
        except:
            pass

        self.muted = False
        self.running = True
        self.thread.start()

    def _loop(self):
        while self.running:
            text = self.queue.get()
            if text is None:
                break
            if not self.muted and text:
                try:
                    self.engine.stop()
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass

    def speak(self, text: str):
        if not self.muted:
            self.queue.put(text)

    def set_muted(self, mute: bool):
        self.muted = mute

    def close(self):
        self.running = False
        self.queue.put(None)
        try:
            self.thread.join(timeout=1.0)
        except:
            pass

# =========================
# FONCTIONS OCR
# =========================
def preprocess_image_for_ocr(frame):
    """Prétraitement : gris + blur + seuillage adaptatif"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    return thresh

def group_words_into_lines(ocr_data, conf_min=OCR_CONF_MIN):
    """Regroupe les mots détectés par Tesseract en lignes ordonnées"""
    groups = {}
    n = len(ocr_data["text"])
    for i in range(n):
        word = ocr_data["text"][i]
        try:
            conf = float(ocr_data["conf"][i])
        except:
            conf = -1.0
        if not word or not word.strip() or conf < conf_min:
            continue
        key = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
        groups.setdefault(key, []).append((ocr_data["left"][i], word.strip()))
    ordered_lines = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        words_sorted = sorted(groups[key], key=lambda x: x[0])
        line_text = " ".join([w for _, w in words_sorted])
        if line_text:
            ordered_lines.append(line_text)
    return ordered_lines

def normalize_text(text: str) -> str:
    """Normalise le texte pour comparaison et déduplication"""
    return " ".join(text.strip().split()).lower()

def detect_best_psm(ocr_data):
    """
    Détecte automatiquement le PSM :
    - texte dispersé si plusieurs blocs et lignes
    - ligne unique si une seule ligne
    - sinon bloc classique
    """
    block_count = len(set(ocr_data["block_num"]))
    line_count = len(set(ocr_data["line_num"]))
    if block_count > 1 and line_count > 2:
        return 12  # texte dispersé
    elif line_count == 1:
        return 7   # une seule ligne
    else:
        return 6   # bloc de texte

# =========================
# MAIN OCR + TTS WEBCAM
# =========================
def run_ocr_tts_webcam():
    PSM = 6  # valeur initiale : bloc de texte

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tts_engine = NonBlockingTTS(lang_hint="fra" if "fra" in OCR_LANG else "eng")
    spoken_memory = deque(maxlen=30)
    last_tts_time = 0.0
    muted = False

    last_ocr_results = []
    frame_counter = 0
    prev_time = time.time()

    print("Contrôles : q=quit | b=bloc | s=scene | l=ligne | m=mute | r=reset mémoire")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        processed_frame = preprocess_image_for_ocr(frame)

        frame_counter += 1
        spoke_this_cycle = False

        if frame_counter % OCR_FRAME_SKIP == 0:
            config = f"--oem {OCR_OEM} --psm {PSM}"
            ocr_data = pytesseract.image_to_data(processed_frame,
                                                 lang=OCR_LANG,
                                                 config=config,
                                                 output_type=Output.DICT)
            # PSM auto
            PSM = detect_best_psm(ocr_data)

            # Overlay des mots
            new_ocr_results = []
            for i in range(len(ocr_data["text"])):
                word = ocr_data["text"][i]
                try:
                    conf = float(ocr_data["conf"][i])
                except:
                    conf = -1.0
                if word and word.strip() and conf >= OCR_CONF_MIN:
                    x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
                    new_ocr_results.append((x, y, w, h, word, conf))
            last_ocr_results = new_ocr_results

            # Texte à prononcer
            lines = group_words_into_lines(ocr_data)
            text_to_speak = " . ".join(lines).strip()
            normalized_text = normalize_text(text_to_speak)

            now = time.time()
            if (text_to_speak and
                len(normalized_text) >= TTS_MIN_CHARS and
                normalized_text not in spoken_memory and
                (now - last_tts_time) >= TTS_COOLDOWN):
                
                say_text = text_to_speak if len(text_to_speak) <= TTS_MAX_CHARS else text_to_speak[:TTS_MAX_CHARS] + " ..."
                tts_engine.speak(say_text)
                spoken_memory.append(normalized_text)
                last_tts_time = now
                spoke_this_cycle = True

        # Overlay OCR
        for x, y, w, h, word, conf in last_ocr_results:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(display_frame, word, (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # HUD
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        mode_name = {6: "bloc", 7: "ligne", 12: "texte épars"}.get(PSM, str(PSM))
        status_tts = "PARLE" if spoke_this_cycle and not muted else ("MUTE" if muted else "...")
        hud_text = f"OCR {OCR_LANG} | PSM={PSM} ({mode_name}) | conf>={OCR_CONF_MIN} | {int(fps)} FPS | TTS:{status_tts}"
        cv2.putText(display_frame, hud_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, "b=bloc  s=scene  l=ligne  m=mute  r=reset  q=quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Webcam OCR + TTS", display_frame)

        # Clavier
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            PSM = 6
        elif key == ord('s'):
            PSM = 12
        elif key == ord('l'):
            PSM = 7
        elif key == ord('m'):
            muted = not muted
            tts_engine.set_muted(muted)
        elif key == ord('r'):
            spoken_memory.clear()

    cap.release()
    cv2.destroyAllWindows()
    tts_engine.close()

# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    run_ocr_tts_webcam()
