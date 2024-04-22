import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split

# Define the path to the dataset
path = 'Dataset'

# Initialize lists to store image data and corresponding labels
X = []
Y = []

# Iterate through the dataset directory
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            # Read and resize the image
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            # Convert image to array and reshape
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            # Append image data to X and corresponding label to Y
            X.append(im2arr)
            lbl = 0
            if name == 'Plaque':
                lbl = 1
            Y.append(lbl)
            # Print image label and path
            print(name+" "+root+"/"+directory[j]+" "+str(lbl))

# Convert lists to numpy arrays and Normalize the image data
X = np.asarray(X)
Y = np.asarray(Y)
X = X.astype('float32')
X = X/255

# Display a sample image
test = X[3]
cv2.imshow("Sample Image", test)
cv2.waitKey(0)

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Convert labels to categorical format
Y = to_categorical(Y)

# Save and Load preprocessed data
np.save('model/X.txt', X)
np.save('model/Y.txt', Y)
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Load or create a RCNN model
if os.path.exists('model/model.json'):
    # Load model from JSON and weights from H5 file
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    json_file.close()
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()
else:
    # Create a new CNN model
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
    # Print model summary
    print(classifier.summary())
    # Compile the model
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Train the model
    hist = classifier.fit(X_train, y_train, batch_size=16, epochs=10,
                          shuffle=True, verbose=2, validation_data=(X_test,y_test))
    # Save model weights and architecture
    classifier.save_weights('model/model_weights.h5')
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    # Save training history
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
