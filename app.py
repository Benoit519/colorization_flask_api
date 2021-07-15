# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from model import generator_model
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)



# flip filename 'vertical' or 'horizontal'
@app.route("/colorize", methods=["POST"])
def colorize():
    
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    img = Image.open(destination)
    img = img.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(img)
    
    
    if len(np.shape(img_array)) == 2:
        img_array = np.reshape(img_array,(224,224,1))
    
    gray = img_array[:,:,0]
    gray = np.reshape(gray,(1,224,224,1))

    # Initialize the model and load the weights
    model = generator_model()
    model.load_weights("./static/model0.h5")
    
    # Normalize the gray image and make prediction
    maximum_img = np.max(gray)
    max_divided = maximum_img/2
    gray = (gray-max_divided)/max_divided
    predicted_image_lab = model.predict(gray)
    predicted_image_lab = np.reshape(predicted_image_lab,(224,224,3))
    
    # Convert in the predicted image in RGB
    img = 127.5*predicted_image_lab + 127.5
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(img)
    
    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    
    img.save(destination)

    return send_image('temp.png')




# blend filename with stock photo and alpha parameter
@app.route("/blend", methods=["POST"])
def blend():
    # retrieve parameters from html form
    alpha = request.form['alpha']
    filename1 = request.form['image']

    # open images
    target = os.path.join(APP_ROOT, 'static/images')
    filename2 = 'blend.jpg'
    destination1 = "/".join([target, filename1])
    destination2 = "/".join([target, filename2])

    img1 = Image.open(destination1)
    img2 = Image.open(destination2)

    # resize images to max dimensions
    width = max(img1.size[0], img2.size[0])
    height = max(img1.size[1], img2.size[1])

    img1 = img1.resize((width, height), Image.ANTIALIAS)
    img2 = img2.resize((width, height), Image.ANTIALIAS)

    # if image in gray scale, convert stock image to monochrome
    if len(img1.mode) < 3:
        img2 = img2.convert('L')

    # blend and show image
    img = Image.blend(img1, img2, float(alpha)/100)

     # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image('temp.png')


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run()

