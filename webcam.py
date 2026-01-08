import cv2
import numpy as np
import pickle

from deepface import DeepFace
from tensorflow.keras.models import load_model


# Runtime configuration
CONFIDENCE_THRESHOLD = 0.60
DETECTION_INTERVAL = 5
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"


def load_resources():
    """
    Load the trained encoder, classifier, and class labels.
    """
    try:
        with open("knn_model.pkl", "rb") as f:
            knn = pickle.load(f)

        with open("embeddings_128d.pkl", "rb") as f:
            data = pickle.load(f)
            class_names = data["class_names"]

        encoder = load_model(
            "autoencoder_encoder_model.keras",
            compile=False,
        )

        return encoder, knn, class_names

    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None


def run_webcam():
    """
    Perform real-time face recognition using a webcam feed.
    """
    encoder, knn, class_names = load_resources()
    if encoder is None:
        return

    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_results = []

    print("Starting recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % DETECTION_INTERVAL == 0:
            last_results.clear()

            try:
                detected_faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                )

                for face_info in detected_faces:
                    if face_info["confidence"] < 0.5:
                        continue

                    area = face_info["facial_area"]
                    box = [area["x"], area["y"], area["w"], area["h"]]

                    face_img = (face_info["face"] * 255).astype(np.uint8)

                    emb_result = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        align=False,
                    )

                    if not emb_result:
                        continue

                    emb_512 = emb_result[0]["embedding"]
                    emb_128 = encoder.predict(
                        np.asarray([emb_512], dtype=np.float32),
                        verbose=0,
                    )

                    predicted_label = knn.predict(emb_128)[0]
                    predicted_name = class_names[predicted_label]

                    probabilities = knn.predict_proba(emb_128)[0]
                    confidence = probabilities[predicted_label]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        label = f"{predicted_name} ({confidence * 100:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)

                    last_results.append(
                        {
                            "box": box,
                            "label": label,
                            "color": color,
                        }
                    )

            except Exception as e:
                print(f"Frame processing error: {e}")

        for result in last_results:
            x, y, w, h = result["box"]
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                result["color"],
                2,
            )
            cv2.putText(
                frame,
                result["label"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                result["color"],
                2,
            )

        cv2.imshow("Real-Time Face ID", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
