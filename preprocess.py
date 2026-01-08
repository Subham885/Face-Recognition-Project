import os
import cv2
import numpy as np
from deepface import DeepFace

# Dataset paths
INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"

# Face detection backend
DETECTOR_BACKEND = "opencv"


def save_original_and_flipped(face_img, output_dir, base_name, index):
    """
    Save the original face image and its horizontally flipped version.
    """
    original_path = os.path.join(output_dir, f"{base_name}_{index}_orig.jpg")
    flipped_path = os.path.join(output_dir, f"{base_name}_{index}_flip.jpg")

    cv2.imwrite(original_path, face_img)

    flipped = cv2.flip(face_img, 1)
    cv2.imwrite(flipped_path, flipped)


def process_dataset():
    """
    Detect, align, and save faces from the dataset with flip augmentation.
    """
    if not os.path.exists(INPUT_DIR):
        print(f"Error: '{INPUT_DIR}' directory not found.")
        return

    print(f"Using {DETECTOR_BACKEND} backend for face detection...")

    for person_name in os.listdir(INPUT_DIR):
        person_path = os.path.join(INPUT_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        output_person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        source_count = 0

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                # Detect and align faces
                detected_faces = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=DETECTOR_BACKEND,
                    align=True
                )

                if not detected_faces:
                    print(f"Skipping {img_path}: no face detected.")
                    continue

                # Use the first detected face
                face_rgb = detected_faces[0]["face"]

                # Convert from float RGB [0,1] to uint8 BGR [0,255]
                face_uint8 = (face_rgb * 255).astype(np.uint8)
                face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

                save_original_and_flipped(
                    face_bgr,
                    output_person_dir,
                    person_name,
                    source_count
                )

                source_count += 1

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        print(
            f"Processed {person_name}: "
            f"{source_count} source images -> {source_count * 2} total images."
        )


if __name__ == "__main__":
    process_dataset()
    print("Done.")
