import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import numpy as np
import time

# --------------------------
# CONFIG GLOBALE
# --------------------------

LANG = "fra+eng"   # Langues utilisées par Tesseract
OEM  = 3           # 3 = LSTM neural net (modèle moderne par défaut)
PSM  = 6           # Mode de segmentation de page (6 = bloc de texte)
CONF_MIN = 50      # Confiance minimale pour afficher un texte reconnu
OCR_EVERY_N_FRAMES = 6  # Faire un OCR toutes les N frames (pour garder du FPS)


# ---------------------------------------------------------------------
# FONCTION DE PRÉTRAITEMENT — améliore fortement la précision de l'OCR
# ---------------------------------------------------------------------
def preprocess_for_ocr(frame_bgr):
    """
    Prépare l'image pour Tesseract :
    - Passage en niveaux de gris
    - Floutage léger (réduit le bruit sans perdre les détails)
    - Seuillage adaptatif (très bon pour texte filmé par caméra)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)  # plus rapide que bilateralFilter
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return th


# --------------------------
# PROGRAMME PRINCIPAL
# --------------------------
def run():
    global PSM

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    # Option : meilleure résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_results = []
    frame_count = 0
    prev_t = time.time()

    print("Contrôles: q=quitter | b=bloc de texte (PSM 6) | d=texte disperse (PSM 12) | l=une seule ligne de texte (PSM 7)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        proc = preprocess_for_ocr(frame)

        frame_count += 1

        # --------------------------
        # LANCER OCR TOUTES LES N FRAMES
        # --------------------------
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            # Config PSM (Page Segmentation Mode)
            # -----------------------------------
            # PSM 6 : bloc de texte (documents, feuilles)
            # PSM 7 : une seule ligne
            # PSM 12 : texte dispersé (scènes, panneaux, objets)
            config = f"--oem {OEM} --psm {PSM}"

            data = pytesseract.image_to_data(
                proc, lang=LANG, config=config,
                output_type=pytesseract.Output.DICT
            )

            new_results = []
            n = len(data["text"])

            for k in range(n):
                txt = data["text"][k].strip()
                conf_raw = data["conf"][k]

                try:
                    conf = float(conf_raw)
                except:
                    conf = -1.0

                if txt and conf >= CONF_MIN:
                    x = data["left"][k]
                    y = data["top"][k]
                    w = data["width"][k]
                    h = data["height"][k]

                    # Filtre anti-bruit : ignorer les trop petites box
                    if w*h > 500:
                        new_results.append((x, y, w, h, txt, conf))

            last_results = new_results

        # --------------------------
        # DESSIN DES RÉSULTATS
        # --------------------------
        for (x, y, w, h, txt, conf) in last_results:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(display, txt, (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # --------------------------
        # AFFICHAGE HUD
        # --------------------------
        now = time.time()
        fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
        prev_t = now

        psm_names = {6: "bloc (doc)", 7: "une seule ligne", 12: "texte disperse"}
        mode_name = psm_names.get(PSM, str(PSM))

        cv2.putText(display,
                    f"OCR: {LANG} | PSM={PSM} ({mode_name}) | conf>={CONF_MIN} | {int(fps)} FPS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(display,
                    "b=bloc de texte | d=texte disperse | l=une seule ligne de texte | q=quitter",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Webcam OCR (Tesseract)", display)

        # --------------------------
        # CLAVIER : CHANGER LE PSM
        # --------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('b'):
            PSM = 6    # mode document / bloc
        elif key == ord('d'):
            PSM = 12   # texte dispersé 
        elif key == ord('l'):
            PSM = 7    # une seule ligne

    cap.release()
    cv2.destroyAllWindows()


# --------------------------
# START
# --------------------------
if __name__ == "__main__":
    run()
