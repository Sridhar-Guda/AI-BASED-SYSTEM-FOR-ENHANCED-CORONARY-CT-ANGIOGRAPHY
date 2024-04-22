from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from skimage.metrics import structural_similarity
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

app = Flask(__name__)

file = None
text = list()
X, Y = None, None
X_train, X_test, y_train, y_test = None, None, None, None
result = None
predict = None
conf_path = None

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
class_labels = ['No Plaque', 'Plaque']
classifier = None
X_test, y_test = None, None

#function to upload dataset
def uploadDataset():
    global filename, text
    filename = r'C:\Users\ramku\Downloads\PCQ\Dataset'
    text.append(filename+" loaded")
    print(text)

def DataPreprocessing():
    global X, Y
    global filename, text


    if os.path.exists("model/X.txt.npy"):
        X = np.load("model/X.txt.npy")
        Y = np.load("model/Y.txt.npy")
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(64,64,3)
                    X.append(im2arr)
                    lbl = 0
                    if name == 'Plaque':
                        lbl = 1
                    Y.append(lbl)

        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.append("Total classes found in dataset : "+str(class_labels))
    text.append("Total images found in dataset : "+str(X.shape[0]))
    text.append("Dataset train & test split. 80% images used for training and 20% images used for testing")

    print("Dataset Preprocessed")

def runRCNN():
    global X, Y
    global text, classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/model_weights.h5")
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test,y_test))
        classifier.save_weights('model/model_weights.h5')
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print("model is ready to use")


def checkStenosis(filename):
    global result
    first = cv2.imread(filename)
    first = cv2.resize(first,(100,100))
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    result = "Non Stenosis Detected"
    for root, dirs, directory in os.walk("stent"):
        for j in range(len(directory)):
            if 'Thumbs.db' not in directory[j]:
                second = cv2.imread(root+"/"+directory[j])
                second = cv2.resize(second,(100,100))
                second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
                score, diff = structural_similarity(first_gray, second_gray, full=True)
                if score >= 0.12:
                    result = "Significant Stenosis Detected"
    print("Stenosis checked")
    return result


def classify_image(image_path):
    global text, classifier
    filename =image_path
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    result = checkStenosis(filename)
    print("image Classified")
    return class_labels[predict], result

def calculate_metrics():

    global classifier, X_train, X_test,y_train, y_test, conf_path
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,3):
        predict[i] = 0
    y_test = np.argmax(y_test, axis=-1)
    metrics ={
    'Accuracy': precision_score(y_test, predict,average='macro') * 100,
    'Precision': recall_score(y_test, predict,average='macro') * 100,
    'Recall': f1_score(y_test, predict,average='macro') * 100,
    'F1 Score': accuracy_score(y_test,predict)*100
    }
    # confusion matrix scores
    LABELS = class_labels
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize =(11, 9))
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g", annot_kws={"fontsize": 20});
    ax.set_ylim([0,2])
    plt.title("RCNN Plaque Classification Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    conf_path = 'static/confusion_matrix.png'
    plt.savefig(conf_path)
    print("calculate metrics done")
    return metrics


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route to classify images
@app.route('/classify', methods=['POST'])
def classify_image_route():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filename)
            predicted_class, stenosis_result = classify_image(filename)
            # Get precautions
            precautions = get_precautions(predicted_class, stenosis_result)
            # Pass the filename, predicted class, stenosis result, metrics, and precautions to the template
            return render_template('result.html', image_file=uploaded_file.filename, predicted_class=predicted_class, stenosis_result=stenosis_result, precautions=precautions)
    return jsonify({'error': 'No file uploaded or invalid request'})

# Function to determine precautions based on classification
def get_precautions(predicted_class, stenosis_result):
    if predicted_class == 'Plaque' and stenosis_result == 'Significant Stenosis Detected':
        return ["Consult a Healthcare Professional Immediately.","Maintain Healthy Diet.", "Avoid Junk/Outside foods","Avoid Cholestrol Foods."]
    else:
        return ["No specific precaution required."]

# Route to display statistics
@app.route('/statistics')
def display_statistics():
    # Calculate evaluation metrics
    metrics = calculate_metrics()
    # Render statistics.html template with metrics
    return render_template('statistics.html', metrics=metrics, conf_path = conf_path)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    uploadDataset()
    DataPreprocessing()
    runRCNN()

    app.run(debug=True)
