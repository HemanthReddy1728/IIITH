# Hotel Shaandar Restaurant API

This is a Python Flask-based API for managing a restaurant's menu, orders, and bills. The API has three main user roles: Admin, Waiting Staff, and Customers, each with their own set of endpoints.

## Prerequisites

Before running this application, ensure you have the following installed:

- Python 3
- Flask
- Flask-SQLAlchemy
- MySQL (or any other relational database)

requirements:
python3 -m venv .nv
source .nv/bin/activate
pip install wheel Flask-Cors Flask-SQLAlchemy mysqlclient pipdeptree
pip install:
blinker==1.7.0
click==8.1.7
Flask==3.0.0
Flask-Cors==4.0.0
Flask-SQLAlchemy==3.1.1
greenlet==3.0.1
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.3
mysqlclient==2.2.0
pipdeptree==2.13.0
SQLAlchemy==2.0.23
typing_extensions==4.8.0
Werkzeug==3.0.1


## Installation

1. Clone this repository.
2. Install the required Python packages:

```
pip install flask flask-cors flask-sqlalchemy
```

3. Update the database connection string in the `app.config['SQLALCHEMY_DATABASE_URI']` variable in `app.py` with your own database credentials.

## Usage

### Admin

- View the menu: `GET /admin/menu`
- Add an item to the menu: `POST /admin/menu`
- Update an item in the menu: `PUT /admin/menu`

### Waiting Staff

- View customer orders: `GET /staff/orders`
- View a customer's bill: `GET /staff/bill?customerId={customer_id}`

### Customer

- View your order: `GET /customer/order`
- Add an item to your order: `POST /customer/order/add`
- Remove an item from your order: `DELETE /customer/order/remove/{item_id}`
- View your bill: `GET /customer/bill`

## Custom Error Handling

The API has custom error responses with the `418 I'm a teapot` status code and meaningful error messages. If headers are missing or incorrect, no database or API operations are allowed.

