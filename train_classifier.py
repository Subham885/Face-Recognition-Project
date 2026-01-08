import pickle
from sklearn.neighbors import KNeighborsClassifier


def train_classifier():
    """
    Load embeddings and labels, train a KNN classifier,
    and save the trained model to disk.
    """
    print("Loading embeddings...")

    try:
        with open("embeddings_128d.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: embeddings_128d.pkl not found. Run the embedding generation step first.")
        return

    X = data["embeddings"]
    y = data["labels"]

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        weights="distance",
    )

    print("Training KNN classifier...")
    knn.fit(X, y)

    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)

    print("✓ KNN classifier trained and saved.")


if __name__ == "__main__":
    train_classifier()
