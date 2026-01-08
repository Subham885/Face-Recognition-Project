import os
import pickle
import numpy as np

from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# Paths and model configuration
INPUT_DIR = "processed_dataset"
MODEL_NAME = "Facenet512"
TARGET_DIM = 128
AE_EPOCHS = 50


def build_autoencoder(input_dim, encoding_dim):
    """
    Build and compile an autoencoder for embedding dimensionality reduction.
    Returns both the full autoencoder and the encoder-only model.
    """
    input_layer = Input(shape=(input_dim,), name="input_layer")

    encoded = Dense(256, activation="relu")(input_layer)
    encoder_output = Dense(
        encoding_dim, activation="relu", name="encoder_output"
    )(encoded)

    decoded = Dense(256, activation="relu")(encoder_output)
    decoded_output = Dense(
        input_dim, activation="linear", name="decoded_output"
    )(decoded)

    autoencoder = Model(
        inputs=input_layer,
        outputs=decoded_output,
        name="autoencoder",
    )
    autoencoder.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    encoder = Model(
        inputs=input_layer,
        outputs=encoder_output,
        name="encoder",
    )

    return autoencoder, encoder


def generate_embeddings():
    """
    Generate FaceNet embeddings, reduce dimensionality with an autoencoder,
    and save the compressed embeddings and label metadata.
    """
    print(f"Loading {MODEL_NAME} and generating embeddings...")

    embeddings = []
    labels = []

    for person_name in os.listdir(INPUT_DIR):
        person_dir = os.path.join(INPUT_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            try:
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                )

                embeddings.append(result[0]["embedding"])
                labels.append(person_name)

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    if not embeddings:
        print("No embeddings generated.")
        return

    X_512 = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels)

    print(f"Generated embeddings shape: {X_512.shape}")

    autoencoder, encoder = build_autoencoder(
        input_dim=X_512.shape[1],
        encoding_dim=TARGET_DIM,
    )

    print(f"Training autoencoder for {AE_EPOCHS} epochs...")
    autoencoder.fit(
        X_512,
        X_512,
        epochs=AE_EPOCHS,
        batch_size=64,
        shuffle=True,
        verbose=1,
    )

    print("Generating compressed embeddings...")
    X_128 = encoder.predict(X_512)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    output_data = {
        "embeddings": X_128,
        "labels": y_encoded,
        "class_names": label_encoder.classes_,
    }

    with open("embeddings_128d.pkl", "wb") as f:
        pickle.dump(output_data, f)

    encoder.save("autoencoder_encoder_model.keras")

    print("✓ Encoder model saved")
    print("✓ Embeddings and labels saved")


if __name__ == "__main__":
    generate_embeddings()
