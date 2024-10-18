from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy.dialects.mysql import JSON

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/flask_tut'
db = SQLAlchemy(app)
CORS(app)

class Shoe(db.Model):

    __table_args__ = {'extend_existing': True}
    __tablename__ = 'shoes'

    id = db.Column(db.Integer, nullable=False, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Integer)
    description = db.Column(db.String(1000))
    image = db.Column(db.String(1000))

    def __repr__(self):
        return '<Shoe %r>' % self.name