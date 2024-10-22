from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

class Base(DeclarativeBase):
  pass

db = SQLAlchemy(model_class=Base)

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///User.db"
# initialize the app with the extension
db.init_app(app)

class User(db.Model):
    # id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False)
    pwd1 : Mapped[str] = mapped_column(String, nullable=False)


with app.app_context():
    db.create_all()

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
    return render_template("home.html")
    

@app.route("/upload", methods=['POST'])
def upload_file():
    if "file" not in request.files:
        return render_template("upload.html", error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', error="No selected file")

    if file and allowed_file(file.filename):
        filename = "sampleImage." + file.filename.rsplit(".", 1)[1].lower()
        fullpath = os.path.normpath(os.path.join(app.config["uploadFolder"], filename))
        if not fullpath.startswith(app.config["uploadFolder"]):
            return render_template('upload.html', error="Invalid file path.")
        file.save(fullpath)
        run_script("code1.py")
        data = run_script("analyseData.py")
        data = data.stdout.strip()
        # dictData = json.loads(data)

        # capture_output=True,
        # return "File uploaded successfully!"
        # print(data)
        return render_template('Output.html', data = data)
    
    else:
        return render_template('upload.html', error="Invalid file format. Please upload a picture.")

@app.route("/register", methods=['GET', 'POST'])
def insert_user():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            new_user = User(username=username, email=email, pwd1=password)
            db.session.add(new_user)
            db.session.commit()
        except Exception as e:
            print("Error:", e)
            print("errorr")
            return render_template('Register.html', info="Enter correct detail")
        return render_template("Register.html", info=f"User {username} added to the database!")

        
    # return render_template("Register.html", info="User {{username}} added to the database!")
    return render_template("Register.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            password = request.form.get('password')
            uname = request.form.get('username')
            user = User.query.filter_by(username=uname).first()
            print(uname, user)
            if user:
                print('Login successful!', 'success')
                return render_template("upload.html")
            else:
                return render_template("login.html", info = "Invalid credentials. Please try again")
        except:
            return render_template("login.html", info = "Something went wrong. Please try again")

    return render_template('login.html')


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, port=2000)