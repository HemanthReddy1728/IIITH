from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess

app = Flask(__name__)

uploadFolder = os.path.abspath("uploads")
app.config["uploadFolder"] = uploadFolder

allowedExtenstions = ["png", "jpg", "jpeg"]

def allowed_file(filename):
    fileName, extension = filename.split(".")
    if(extension.lower() in allowedExtenstions):
        return True
    return False

def run_script(file_path):
    data = subprocess.run(["python3", file_path], capture_output = True, text = True)
    return data

@app.route("/")
def index():
    return render_template("FrontEnd.html")

@app.route("/upload", methods=['POST'])
def upload_file():
    if "file" not in request.files:
        return render_template("FrontEnd.html", error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('FrontEnd.html', error="No selected file")

    if file and allowed_file(file.filename):
        filename = "sampleImage." + file.filename.rsplit(".", 1)[1].lower()
        fullpath = os.path.normpath(os.path.join(app.config["uploadFolder"], filename))
        if not fullpath.startswith(app.config["uploadFolder"]):
            return render_template('FrontEnd.html', error="Invalid file path.")
        file.save(fullpath)
        run_script("code1.py")
        data = run_script("analyseData.py")
        # capture_output=True,
        # return "File uploaded successfully!"
        return render_template('Output.html', data = data.stdout.strip())
    
    else:
        return render_template('FrontEnd.html', error="Invalid file format. Please upload a picture.")

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)