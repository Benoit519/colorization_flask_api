# Image manipulation API

from flask import Flask, render_template, send_from_directory, redirect
import os
from PIL import Image

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access redirects to API documentation
@app.route("/")
def main():
    return redirect("https://github.com/gxercavins/image-api/blob/master/README.md", code=302)




# flip filename 'vertical' or 'horizontal'
@app.route("/colorize/<mode>/<filename>", methods=["GET"])
def flip(mode, filename):

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    img = Image.open(destination)

    # check mode
    if mode == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return render_template("error.html", message="Invalid mode (vertical or horizontal)"), 400

    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image('temp.png')




# blend filename1 and filename2 with alpha parameter
@app.route("/blend/<alpha>/<filename1>/<filename2>", methods=["GET"])
def blend(alpha, filename1, filename2):

    # check for valid alpha
    alpha = float(alpha)
    if not 0 <= alpha <= 100:
        return render_template("error.html", message="Invalid alpha value (0-100)"), 400

    #open images
    target = os.path.join(APP_ROOT, 'static/images')
    destination1 = "/".join([target, filename1])
    destination2 = "/".join([target, filename2])

    img1 = Image.open(destination1)
    img2 = Image.open(destination2)

    # check for dimensions and resize to larger ones
    width = max(img1.size[0], img2.size[0])
    height = max(img1.size[1], img2.size[1])

    img1 = img1.resize((width, height), Image.ANTIALIAS)
    img2 = img2.resize((width, height), Image.ANTIALIAS)

    # if one image in gray scale, convert the other to monochrome
    if len(img1.mode) < 3:
        img2 = img2.convert('L')
    elif len(img2.mode) < 3:
        img1 = img1.convert('L')

    # blend images
    img = Image.blend(img1, img2, float(alpha)/100)

    # save and return
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
