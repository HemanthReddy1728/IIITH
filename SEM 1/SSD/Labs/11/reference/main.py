from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import JSON

# Initialize flask app
app = Flask(__name__)

# add your own MySQL connection String
# mysql://username:password@ip:port/db_name
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/flasktutorial'

# Connect SQLAlchemy engine to db and create db object
db = SQLAlchemy(app)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)


# Define the Shoe Entity
class Shoe(db.Model):
    __table_args__ = {'schema': 'flasktutorial', 'extend_existing': True}
    __tablename__ = 'shoes'  # Set the explicit table name

    id = db.Column(db.Integer, nullable=False, primary_key=True)
    brand = db.Column(db.String(80), nullable=False)
    name = db.Column(db.String(80), nullable=False)

    def __init__(self, id, brand, name):
        self.id = id
        self.brand = brand
        self.name = name

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'brand': self.brand
        }

    def __repr__(self):
        return f'<User {self.name}>'


# Define the User Entity
class User(db.Model):
    __table_args__ = {'schema': 'flasktutorial', 'extend_existing': True}
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
    app.run(debug=True)


'''
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import JSON
import json

# Initialize flask app
app = Flask(__name__)

# add your own MySQL connection String
# mysql://username:password@ip:port/db_name
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/Hotel_Shaandar'

# Connect SQLAlchemy engine to db and create db object
db = SQLAlchemy(app)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Define the Menu Entity
class Menu(db.Model):
    __table_args__ = {'schema': 'Hotel_Shaandar', 'extend_existing': True}
    __tablename__ = 'Menu'

    item_id = db.Column(db.Integer, nullable=False, primary_key=True)
    item_name = db.Column(db.String(80), nullable=False)
    item_price = db.Column(db.Integer, nullable=False)

    def __init__(self, item_id, item_name, item_price):
        self.item_id = item_id
        self.item_name = item_name
        self.item_price = item_price

    def to_dict(self):
        return {
            'item_id': self.item_id,
            'item_name': self.item_name,
            'item_price': self.item_price
        }

# Define the CustomerOrder Entity
class CustomerOrder(db.Model):
    __table_args__ = {'schema': 'Hotel_Shaandar', 'extend_existing': True}
    __tablename__ = 'CustomerOrder'

    customerId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    items = db.Column(JSON, nullable=True)

    def __init__(self, customerId, items):
        self.customerId = customerId
        self.items = items

    def to_dict(self):
        return {
            'customerId': self.customerId,
            'items': self.items
        }

# Admin API Endpoints
@app.route('/admin/menu', methods=['GET'])
def admin_view_menu():
    menu_items = Menu.query.all()
    menu_data = [item.to_dict() for item in menu_items]
    return jsonify(menu_data)

@app.route('/admin/menu', methods=['POST'])
def admin_add_item():
    data = request.get_json()
    if not data or "item_id" not in data or "item_name" not in data or "item_price" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    item_id = data["item_id"]
    item_name = data["item_name"]
    item_price = data["item_price"]

    menu_item = Menu(item_id=item_id, item_name=item_name, item_price=item_price)
    db.session.add(menu_item)
    db.session.commit()
    
    return jsonify({"message": "Item added successfully"})

@app.route('/admin/menu', methods=['PUT'])
def admin_update_item():
    data = request.get_json()
    if not data or "item_id" not in data or "item_name" not in data or "item_price" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    item_id = data["item_id"]
    item_name = data["item_name"]
    item_price = data["item_price"]

    menu_item = Menu.query.get(item_id)
    if menu_item:
        menu_item.item_name = item_name
        menu_item.item_price = item_price
        db.session.commit()
        return jsonify({"message": "Item updated successfully"})
    else:
        return jsonify({"error": "Item not found"}), 404

# Waiting Staff API Endpoints
@app.route('/staff/orders', methods=['GET'])
def staff_view_orders():
    orders = CustomerOrder.query.all()
    orders_data = [
        {
            'customerId': order.customerId,
            'items': json.loads(order.items) if order.items else []
        }
        for order in orders
    ]
    return jsonify(orders_data)

@app.route('/staff/bill', methods=['GET'])
def staff_view_bill():
    customerId = request.args.get("customerId")
    if not customerId:
        return jsonify({"error": "Missing 'customerId' query parameter"}), 400

    order = CustomerOrder.query.get(customerId)
    if order:
        items = json.loads(order.items) if order.items else []
        bill_amount = sum(item['item_price'] for item in items)
        
        response_data = {
            "customerId": customerId,
            "bill_amount": bill_amount,
            "items": items
        }

        return jsonify(response_data)
    else:
        return jsonify({"error": "Customer not found"}, 404)


# Customer API Endpoints
@app.route('/customer/order', methods=['GET'])
def customer_view_order():
    customerId = request.headers.get("customerId")
    if not customerId:
        return jsonify({"error": "Missing 'customerId' header"}), 400

    order = CustomerOrder.query.get(customerId)
    if order:
        items = json.loads(order.items) if order.items else []
        return jsonify(items)
    else:
        return jsonify([])

@app.route('/customer/order/add', methods=['POST'])
def customer_add_item_to_order():
    customerId = request.headers.get("customerId")
    if not customerId:
        return jsonify({"error": "Missing 'customerId' header"}), 400

    data = request.get_json()
    if not data or "item_id" not in data or "item_name" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    menu_item = Menu.query.get(data["item_id"])
    if menu_item:
        order = CustomerOrder.query.get(customerId)
        if not order:
            order = CustomerOrder(customerId=customerId, items=json.dumps([]))

        items = json.loads(order.items)
        items.append({
            "item_id": menu_item.item_id,
            "item_name": menu_item.item_name,
            "item_price": menu_item.item_price
        })

        order.items = json.dumps(items)
        db.session.add(order)
        db.session.commit()
        return jsonify({"message": "Item added to the order successfully"})

    return jsonify({"error": "Item not found in the menu"}), 404

@app.route('/customer/order/remove/<int:item_id>', methods=['DELETE'])
def customer_remove_item_from_order(item_id):
    customerId = request.headers.get("customerId")
    if not customerId:
        return jsonify({"error": "Missing 'customerId' header"}), 400

    order = CustomerOrder.query.get(customerId)
    if order:
        items = json.loads(order.items)
        item_to_remove = next((item for item in items if item["item_id"] == item_id), None)
        if item_to_remove:
            items.remove(item_to_remove)
            order.items = json.dumps(items)
            db.session.add(order)
            db.session.commit()
            return jsonify({"message": "Item removed from the order successfully"})

    return jsonify({"error": "Item not found in the customer's order"}), 404


@app.route('/customer/bill', methods=['GET'])
def customer_render_bill():
    customerId = request.headers.get("customerId")
    if not customerId:
        return jsonify({"error": "Missing 'customerId' header"}), 400

    order = CustomerOrder.query.get(customerId)
    if order:
        items = json.loads(order.items) if order.items else []
        bill_amount = sum(item['item_price'] for item in items)
        
        # Render the HTML bill template with customer-specific information
        return render_template("bill.html", customerId=customerId, items=items, total_amount=bill_amount)
    else:
        return jsonify({"error": "Customer not found"}, 404)
    
if __name__ == '__main__':
    app.run(debug=True)

'''