from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Session, attributes
import json
import os
# Initialize flask app
app = Flask(__name__)

# Add your own MySQL connection String
# mysql://username:password@ip:port/db_name
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/Hotel_Shaandar'

# Connect SQLAlchemy engine to the database and create the database object
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

    customer_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    items = db.Column(JSON, nullable=True)

    def __init__(self, customer_id, items):
        self.customer_id = customer_id
        self.items = items

    def to_dict(self):
        return {
            'customer_id': self.customer_id,
            'items': self.items
        }

# Custom error response
def custom_error_response(error_message):
    response = jsonify({"error": error_message})
    response.status_code = 418  # I'm a teapot error
    return response

# Check headers for required information
def check_headers():
    root_org = request.headers.get("rootOrg")
    org = request.headers.get("org")
    if root_org != "Restuarant" or org != "Shaandar":
        return False # return True -> In browser, admin and staff apis only work. If False, nothing works
    return True

# Admin API Endpoints
@app.route('/admin/menu', methods=['GET'])
def admin_view_menu():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    menu_items = Menu.query.all()
    menu_data = [item.to_dict() for item in menu_items]
    return jsonify(menu_data)

@app.route('/admin/menu', methods=['POST'])
def admin_add_item():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    data = request.get_json()
    if not data or "item_id" not in data or "item_name" not in data or "item_price" not in data:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Invalid request data")

    item_id = data["item_id"]
    item_name = data["item_name"]
    item_price = data["item_price"]

    menu_item = Menu(item_id=item_id, item_name=item_name, item_price=item_price)
    db.session.add(menu_item)
    db.session.commit()
    
    return jsonify({"message": "Item added successfully"})

@app.route('/admin/menu', methods=['PUT'])
def admin_update_item():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    data = request.get_json()
    if not data or "item_id" not in data or "item_name" not in data or "item_price" not in data:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Invalid request data")

    item_id = data["item_id"]
    item_name = data["item_name"]
    item_price = data["item_price"]

    menu_item = db.session.get(Menu, item_id)
    if menu_item:
        menu_item.item_name = item_name
        menu_item.item_price = item_price
        db.session.commit()
        return jsonify({"message": "Item updated successfully"})
    else:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Item not found"), 404

# Waiting Staff API Endpoints
@app.route('/staff/orders', methods=['GET'])
def staff_view_orders():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    orders = CustomerOrder.query.all()
    orders_data = [
        {
            'customer_id': order.customer_id,
            'items': json.loads(order.items) if order.items else []
        }
        for order in orders
    ]
    return jsonify(orders_data)

@app.route('/staff/bill', methods=['GET'])
def staff_view_bill():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    customer_id = request.args.get("customer_id")
    if not customer_id:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing 'customer_id' query parameter"), 400

    order = db.session.get(CustomerOrder, customer_id)
    if order:
        items = json.loads(order.items) if order.items else []
        bill_amount = sum(item['item_price'] for item in items)
        
        response_data = {
            "customer_id": customer_id,
            "bill_amount": bill_amount,
            "items": items
        }

        return jsonify(response_data)
    else:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Customer not found"), 404

# Customer API Endpoints
@app.route('/customer/order', methods=['GET'])
def customer_view_order():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    customer_id = request.headers.get("customerId")
    if not customer_id:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing 'customer_id' header"), 400

    order = db.session.get(CustomerOrder, customer_id)
    if order:
        items = json.loads(order.items) if order.items else []
        return jsonify(items)
    else:
        return jsonify([])

@app.route('/customer/order/add', methods=['POST'])
def customer_add_item_to_order():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    customer_id = request.headers.get("customerId")
    if not customer_id:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing 'customer_id' header"), 400

    data = request.get_json()
    if not data or "item_id" not in data or "item_name" not in data:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Invalid request data"), 400

    menu_item = db.session.get(Menu, data["item_id"])
    if menu_item:
        if menu_item.item_name != data["item_name"]:
            return custom_error_response("There is some issue. Please try again") # return custom_error_response("Item name does not match the selected item"), 400
        
        order = db.session.get(CustomerOrder, customer_id)
        if not order:
            order = CustomerOrder(customer_id=customer_id, items=json.dumps([]))

        items = json.loads(order.items)
        items.append({
            "item_id": menu_item.item_id,
            "item_name": menu_item.item_name,
            "item_price": menu_item.item_price
        })
        # attributes.flag_modified(CustomerOrder, 'items')
        order.items = json.dumps(items)
        db.session.add(order)
        db.session.commit()
        return jsonify({"message": "Item added to the order successfully"})

    return custom_error_response("There is some issue. Please try again") # return custom_error_response("Item not found in the menu"), 404



@app.route('/customer/order/remove/<int:item_id>', methods=['DELETE'])
def customer_remove_item_from_order(item_id):
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    customer_id = request.headers.get("customerId")
    if not customer_id:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing 'customer_id' header"), 400

    order = db.session.get(CustomerOrder, customer_id)
    if order:
        items = json.loads(order.items)
        item_to_remove = next((item for item in items if item["item_id"] == item_id), None)
        if item_to_remove:
            items.remove(item_to_remove)
            order.items = json.dumps(items)
            db.session.add(order)
            db.session.commit()
            return jsonify({"message": "Item removed from the order successfully"})

    return custom_error_response("There is some issue. Please try again") # return custom_error_response("Item not found in the customer's order"), 404

@app.route('/customer/bill', methods=['GET'])
def customer_render_bill():
    if not check_headers():
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing or incorrect headers. No DB/API operations allowed.")
    
    customer_id = request.headers.get("customerId")
    if not customer_id:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Missing 'customer_id' header"), 400

    order = db.session.get(CustomerOrder, customer_id)
    if order:
        items = json.loads(order.items) if order.items else []
        bill_amount = sum(item['item_price'] for item in items)
        
        return render_template("bill.html", customer_id=customer_id, items=items, total_amount=bill_amount)
    else:
        return custom_error_response("There is some issue. Please try again") # return custom_error_response("Customer not found"), 404

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)
