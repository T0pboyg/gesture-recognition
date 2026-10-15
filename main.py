import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp


# -----------------------------------------
# Ρυθμίσεις
# -----------------------------------------

MATCH_THRESHOLD = 0.10

memes_folder = Path("memes")

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# -----------------------------------------
# Βρίσκουμε όλα τα memes
# -----------------------------------------

meme_files = [
    file
    for file in memes_folder.iterdir()
    if file.suffix.lower() in valid_extensions
]


# -----------------------------------------
# Φόρτωση Image Embedder
# -----------------------------------------

embedder = mp.tasks.vision.ImageEmbedder.create_from_model_path(
    "models/mobilenet_v3_small.tflite"
)


# -----------------------------------------
# Αποθήκευση εικόνων και embeddings
# -----------------------------------------

memes = []
meme_embeddings = []


for meme_file in meme_files:

    image = cv2.imread(str(meme_file))

    if image is None:
        print("Failed to load image:", meme_file.name)
        continue

    print("Loaded:", meme_file.name)

    # Κρατάμε την πραγματική εικόνα
    memes.append(
        (meme_file.name, image)
    )

    # Μετατροπή σε MediaPipe image
    mp_image = mp.Image.create_from_file(
        str(meme_file)
    )

    # Δημιουργία embedding
    result = embedder.embed(mp_image)

    embedding = (
        result.embeddings[0].embedding
    )

    # Αποθήκευση ονόματος + embedding
    meme_embeddings.append(
        (meme_file.name, embedding)
    )


# -----------------------------------------
# Άνοιγμα κάμερας
# -----------------------------------------

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Camera is closed")
    raise SystemExit


# Για να ξέρουμε αν έχει ανοίξει
# το παράθυρο Best Meme
best_meme_window_open = False


# -----------------------------------------
# Main loop
# -----------------------------------------

while True:

    success, frame = video.read()

    if not success:
        break


    # -------------------------------------
    # BGR -> RGB
    # -------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -------------------------------------
    # Frame -> MediaPipe image
    # -------------------------------------

    camera_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -------------------------------------
    # Embedding κάμερας
    # -------------------------------------

    camera_result = embedder.embed(
        camera_image
    )

    camera_embedding = (
        camera_result.embeddings[0].embedding
    )


    # -------------------------------------
    # Αναζήτηση καλύτερου meme
    # -------------------------------------

    best_score = -1
    best_meme = None


    for meme_name, meme_embedding in meme_embeddings:

        score = np.dot(
            camera_embedding,
            meme_embedding
        ) / (
            np.linalg.norm(camera_embedding)
            *
            np.linalg.norm(meme_embedding)
        )


        if score > best_score:

            best_score = score
            best_meme = meme_name


    # -------------------------------------
    # Αν περνάει το threshold
    # -------------------------------------

    if (
        best_meme is not None
        and best_score >= MATCH_THRESHOLD
    ):

        text = (
            f"Best match: {best_meme} "
            f"| Score: {best_score:.2f}"
        )


        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Βρίσκουμε την πραγματική εικόνα
        for meme_name, meme_image in memes:

            if meme_name == best_meme:

                height, width = (
                    meme_image.shape[:2]
                )

                max_width = 500


                # Resize αν είναι πολύ μεγάλη
                if width > max_width:

                    scale = (
                        max_width / width
                    )

                    new_width = int(
                        width * scale
                    )

                    new_height = int(
                        height * scale
                    )

                    display_meme = cv2.resize(
                        meme_image,
                        (
                            new_width,
                            new_height
                        )
                    )

                else:

                    display_meme = meme_image


                cv2.imshow(
                    "Best Meme",
                    display_meme
                )

                best_meme_window_open = True

                break


    # -------------------------------------
    # Αν ΔΕΝ περνάει το threshold
    # -------------------------------------

    else:

        text = (
            f"No good match "
            f"| Best score: {best_score:.2f}"
        )


        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Κλείνουμε το προηγούμενο meme
        if best_meme_window_open:

            cv2.destroyWindow(
                "Best Meme"
            )

            best_meme_window_open = False


    # -------------------------------------
    # Εμφάνιση κάμερας
    # -------------------------------------

    cv2.imshow(
        "Camera",
        frame
    )


    # q = έξοδος
    if cv2.waitKey(1) == ord("q"):
        break


# -----------------------------------------
# Cleanup
# -----------------------------------------

video.release()
cv2.destroyAllWindows()