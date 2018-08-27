from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os

STYLE_FOLDER = './static/img/style'
CONTENT_FOLDER = './static/img/content'
TRANSFER_FOLDER = './static/img/transfer'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['STYLE_FOLDER'] = STYLE_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['TRANSFER_FOLDER'] = TRANSFER_FOLDER
app.debug = True

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def style_transfer(styleimage_path,contentimage_path):
	transferimage_filename = contentimage_path.split("/")[-1]
	styleimage = image.load_img(styleimage_path, target_size = (224,224))
	contentimage = image.load_img(contentimage_path, target_size = (224,224))

	styleimage_array = image.img_to_array(styleimage)
	contentimage_array = image.img_to_array(contentimage)
	
	transferimage_array = np.concatenate((styleimage_array, contentimage_array), axis=0)

	transferimage = Image.fromarray(transferimage_array.astype('uint8'))

	transferimage_path = os.path.join(app.config['TRANSFER_FOLDER'], transferimage_filename)
	transferimage.save(transferimage_path)
	return transferimage_path

@app.route("/")
def index():
	return render_template("index.html")


@app.route("/test1/", methods=['GET','POST'])
def test1():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'styleimage' not in request.files or 'contentimage' not in request.files:
			flash('No file part')
			return redirect(request.url)
		styleimage = request.files['styleimage']
		contentimage = request.files['contentimage']
		# if user does not select file, browser also
		# submit a empty part without filename
		if styleimage.filename == '' or contentimage.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if styleimage and contentimage and allowed_file(styleimage.filename) and allowed_file(contentimage.filename):
			styleimagename = secure_filename(styleimage.filename)
			contentimagename = secure_filename(contentimage.filename)
			styleimage_path = os.path.join(app.config['STYLE_FOLDER'], styleimagename)
			contentimage_path = os.path.join(app.config['CONTENT_FOLDER'], contentimagename)
			styleimage.save(styleimage_path)
			contentimage.save(contentimage_path)
			# return redirect(url_for('uploaded_file',
			# 	filename=filename))
	# return render_template("index.html")
	transferimage_path = style_transfer(styleimage_path,contentimage_path)
	file_paths = dict()
	file_paths["contentimage_path"] = contentimage_path
	file_paths["styleimage_path"] = styleimage_path
	file_paths["transferimage_path"] = transferimage_path
	return render_template("result.html", file_paths=file_paths)
	
@app.route("/test2/", methods=['GET','POST'])
def test2():
	input1 = request.form.get('input1')
	input2 = request.form.get('input2')
	adds = int(input1)+int(input2)
	value = np.zeros(adds)
	return str(value)


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)