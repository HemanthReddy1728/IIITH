from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import JSON  # change this to MySQL dialect

# Initialize flask app
app = Flask(__name__)

# add your own MySQL connection String instead of Postgres
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/flasktutorial'

# Connect SQLAlchemy engine to db and create db object
db = SQLAlchemy(app)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)


# Define the Shoe Entity
class Shoe(db.Model):
    __table_args__ = {'schema': 'flasktutorial', 'extend_existing': True}  # not needed for MySQL
    __tablename__ = 'shoes'  # Set the explicit table name

    id = db.Column(db.Integer, nullable=False, primary_key=True)
    brand = db.Column(db.String(80), nullable=False)
    name = db.Column(db.String(80), nullable=False)

    def __init__(self, id, brand, name):
        self.id = id
        self.brand = brand
        self.name = name

    def __repr__(self):
        return f'<User {self.name}>'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'brand': self.brand
        }


# Define the User Entity
class User(db.Model):
    __table_args__ = {'schema': 'flasktutorial', 'extend_existing': True}  # not needed for MySQL
    __tablename__ = 'users'  # Set the explicit table name

    id = db.Column(db.Integer, nullable=False, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    shoes = db.Column(JSON(db.Integer), nullable=True)

    def __init__(self, id, name, gender):
        self.id = id
        self.name = name
        self.gender = gender

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'gender': self.gender
        }


# Create a new Shoe
@app.route('/shoes', methods=['POST'])
def create_shoe():
    response_payload = request.get_json()

    shoe_id = response_payload.get('shoe_id')
    shoe_brand = response_payload.get('shoe_brand')
    shoe_name = response_payload.get('shoe_name')

    if shoe_id and shoe_brand and shoe_name:

        shoe = Shoe(id=shoe_id, brand=shoe_brand, name=shoe_name)
        db.session.add(shoe)
        db.session.commit()

        return jsonify({"message": "Shoe added successfully."}), 200
    else:
        return jsonify({"error": "Invalid data provided."}), 400


# Create a new User
@app.route('/users', methods=['POST'])
def create_user():
    response_payload = request.get_json()

    user_id = response_payload.get('user_id')
    user_name = response_payload.get('user_name')
    user_gender = response_payload.get('user_gender')

    if user_id and user_name and user_gender:

        user = User(id=user_id, name=user_name, gender=user_gender)
        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User added successfully."}), 200
    else:
        return jsonify({"error": "Invalid data provided."}), 400


# Add show for User
@app.route('/users/add-shoe', methods=['POST'])
def add_shoe():
    query_param = request.args
    headers = request.headers

    user_id = headers.get("user_id")
    shoe_id = query_param.get("shoe_id")

    if shoe_id and user_id:
        user = User.query.get(user_id)

        if user is None:
            return jsonify({"error": "Invalid data provided."}), 400

        if user.shoes is None:
            user.shoes = [shoe_id]
        else:
            user.shoes.append(shoe_id)

        db.session.commit()

        return jsonify({"message": "User added successfully."}), 200
    else:
        return jsonify({"error": "Invalid data provided."}), 400


# Display shoes for User
@app.route('/users/display-shoes', methods=['GET'])
def display_user_shoes():
    headers = request.headers
    user_id = headers.get("user_id")

    if user_id:
        user = User.query.get(user_id)

        if user is None:
            return jsonify({"error": "Invalid data provided."}), 400

        shoe_list = user.shoes
        if len(shoe_list) == 0:
            return jsonify({"message": "User doesn't have any shoes!"}), 200

        resp = []
        for shoe_id in shoe_list:
            shoe = Shoe.query.get(shoe_id)
            resp.append(shoe.to_dict())

        return jsonify({"user_id": user_id, "shoes": resp}), 200

    else:
        return jsonify({"error": "Invalid data provided."}), 400


# Render shoes for User
@app.route('/users/display-shoes-render', methods=['GET'])
def display_user_shoes_render():
    headers = request.headers
    user_id = headers.get("user_id")

    if user_id:
        user = User.query.get(user_id)

        if user is None:
            return jsonify({"error": "Invalid data provided."}), 400

        shoe_list = user.shoes
        if len(shoe_list) == 0:
            return jsonify({"message": "User doesn't have any shoes!"}), 200

        resp = []
        for shoe_id in shoe_list:
            shoe = Shoe.query.get(shoe_id)
            resp.append(shoe.to_dict())

        return render_template('user_shoes.html', user_id=user_id, shoes=resp)

    else:
        return jsonify({"error": "Invalid data provided."}), 400


if __name__ == '__main__':
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)
