# Workflow

A machine Learning Project which uses Facenet model to create 512D embeddings. The embeddings is converted a 128D vector using autoencoders, since we are using KNN classifier model for the recognition task. The Data is prepossed by using opencv face detection, and augmentation is used by flipping the original image. 


# How to use

Add images of people using their name as the folder name, in the `dataset` folder.

Run the python scripts in the below order:

1. `prepocess.py`
2. `embeddings.py`
3. `train_classifier.py`
4. `webcam.py`

`preprocess.py` will crop the images and augment them and store them in the `processed dataset` folder

`embeddings.py` will use FaceNet512 model to create a 512D vector of the face, and then reduce it to 128D vector using a variational autoendcoder.

`train_classifier.py` is used for the KNN model.

Run the `webcam.py` script, which will recognise a face in the face in the frame, generate a 512D embedding, reduce it to 128D using the autoencoder and then use the KNN classifier to recognize the face.
