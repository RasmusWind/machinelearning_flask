#!./env/bin/python
from flask import Flask, request, render_template
import base64
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import imageio
import numpy as np
import json
from PIL import Image

app = Flask(__name__)
app.config["DEBUG"] = True

image_index = 35

img_rows, img_cols = 28, 28

def create_model(img_rows, img_cols):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train / 255
    x_test = x_test / 255

    num_classes = 10

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(
        Conv2D(
            32, 
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(img_rows, img_cols, 1)
        )
    )

    model.add(
        Conv2D(
            64, 
            (3, 3), 
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    batch_size = 128
    epochs = 20

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("test_model.h5")
    print("Saved model")
    return model

def load_model2(filepath):
    model = load_model(filepath)
    return model

def load_image(filepath, img_rows, img_cols):
    image = Image.open(filepath)
    image = image.resize((img_rows, img_cols))
    image.save(filepath)
    im = imageio.imread(filepath)
    # im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
    gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    gray = gray.reshape(1, img_rows, img_cols, 1)
    gray = gray / 255
    return gray

def ascii_shit(imgs):
    for img in imgs:
        new_img = []
        for row in img:
            print(row)
            new_row = []
            for pixel in row:
                new_pixel = " "
                if pixel > 0.2 and pixel < 0.6:
                    new_pixel = "!"
                elif pixel <= 0.2:
                    new_pixel = "#"
                elif pixel < 1:
                    new_pixel = "."
                
                new_row.append(new_pixel)
            new_img.append(new_row)
        for row in new_img:
            print("".join(row))

@app.route("/upload", methods=["POST"])
def upload():
    form = request.form
    form_file = form["file"]
    ff = form_file.split(",")[-1]
    ffb = str.encode(ff)
    
    with open("./dataimages/imgfile.png", "wb") as f:
        f.write(base64.decodebytes(ffb))

    gray = load_image("./dataimages/imgfile.png", 28, 28)
    ascii_shit(gray)
    model = load_model("./test_model.h5")
    # model = create_model(img_rows, img_cols)
    prediction = model.predict(gray)
    result = prediction.argmax()
    print("Predicted answer: ", result)
    correct_num = int(input("What was the correct answer? : "))
    cache_num = correct_num
    gray = np.reshape(np.array([gray]),(1, 28,28,1))
    correct_num = to_categorical(np.array([correct_num]), num_classes=10, dtype='int')

    model.fit(gray, correct_num, epochs=2, batch_size=128, verbose=1)
    
    model.save("test_model.h5")
        
    return json.dumps({
        "result": str(cache_num)
    })

@app.route("/")
def index():
    return render_template("index.html")