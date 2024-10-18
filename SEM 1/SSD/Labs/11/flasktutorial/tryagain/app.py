from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shoe_inventory.db'
db = SQLAlchemy(app)

# Define SQLAlchemy models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    shoes = db.relationship('Shoe', secondary='user_shoes', back_populates='owners')

class Shoe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    owners = db.relationship('User', secondary='user_shoes', back_populates='shoes')

user_shoes = db.Table('user_shoes',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('shoe_id', db.Integer, db.ForeignKey('shoe.id'))
)

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/')
def homepage():
    return "Welcome to the Shoe Inventory App"

# Add more custom routes as needed


# API endpoint to create a new shoe
@app.route('/api/create_shoe', methods=['POST'])
def create_shoe():
    brand = request.form.get('brand')
    name = request.form.get('name')

    if brand and name:
        shoe = Shoe(brand=brand, name=name)
        db.session.add(shoe)
        db.session.commit()
        flash('Shoe created successfully', 'success')
    else:
        flash('Invalid input. Please check the form.', 'danger')

    return redirect(url_for('render_shoes'))

# API endpoint to create a new user
@app.route('/api/create_user', methods=['POST'])
def create_user():
    name = request.form.get('name')
    gender = request.form.get('gender')

    if name and gender:
        user = User(name=name, gender=gender)
        db.session.add(user)
        db.session.commit()
        flash('User created successfully', 'success')
    else:
        flash('Invalid input. Please check the form.', 'danger')

    return redirect(url_for('render_users'))

# API endpoint to add a shoe to a user
@app.route('/api/add_shoe_to_user', methods=['POST'])
def add_shoe_to_user():
    user_id = request.form.get('user_id')
    shoe_id = request.form.get('shoe_id')

    user = User.query.get(user_id)
    shoe = Shoe.query.get(shoe_id)

    if user and shoe:
        user.shoes.append(shoe)
        db.session.commit()
        return jsonify({'message': 'Shoe added to user successfully'})
    else:
        return jsonify({'error': 'Invalid user or shoe ID'})

# API endpoint to display a list of owned shoes for a user
@app.route('/api/display_shoes_for_user/<int:user_id>', methods=['GET'])
def display_shoes_for_user(user_id):
    user = User.query.get(user_id)
    if user:
        shoes = user.shoes
        shoe_data = [{'id': shoe.id, 'brand': shoe.brand, 'name': shoe.name} for shoe in shoes]
        return jsonify(shoe_data)
    else:
        return jsonify({'error': 'User not found'})

# Render user list
@app.route('/users', methods=['GET'])
def render_users():
    users = User.query.all()
    return render_template('users.html', users=users)

# Render shoe list
@app.route('/shoes', methods=['GET'])
def render_shoes():
    shoes = Shoe.query.all()
    return render_template('shoes.html', shoes=shoes)

if __name__ == '__main__':
    app.run(debug=True)
