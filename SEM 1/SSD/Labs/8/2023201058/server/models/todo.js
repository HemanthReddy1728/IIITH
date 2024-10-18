// models/ToDo.js

const mongoose = require('mongoose');

// Define the schema for the To-Do model
const toDoSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true,
    },
    description: String,
    date: {
        type: Date,
        required: true,
    },
    status: {
        type: Boolean,
    }
});

// Create the To-Do model
const ToDoModel = mongoose.model('ToDo', toDoSchema);

module.exports = ToDoModel;
