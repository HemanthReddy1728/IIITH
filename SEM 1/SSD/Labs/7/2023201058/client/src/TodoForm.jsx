import React, { useState } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import 'bootstrap/dist/css/bootstrap.min.css';

function TodoForm() {
    const [title, setTitle] = useState("");
    const [description, setDescription] = useState("");
    const [dueDate, setDueDate] = useState(new Date());

    const handleCreateTodo = () => {
        // Check if the due date is in the future
        const currentDate = new Date();
        if (dueDate <= currentDate) {
            alert("Please select a future due date for the To-Do item.");
            return; // Do not proceed with creation
        }

        // Create a To-Do item object
        const todoItem = {
            title,
            description,
            dueDate,
        };

        // Display the entered details in an alert box
        const detailsText = `To-Do Item Details:\nTitle: ${todoItem.title}\nDescription: ${todoItem.description}\nDue Date: ${todoItem.dueDate}`;
        alert(detailsText);

        // You can handle the creation of the To-Do item here
        // For this example, let's log the data to the console
        console.log("New To-Do Item:", todoItem);

        // You can also send the data to your server for storage

        // Reset the form fields after creating the item
        setTitle("");
        setDescription("");
        setDueDate(new Date());
    };

    return (
        <div className="container">
            <h2>Create To-Do Item</h2>
            <form>
                <div className="mb-3">
                    <label htmlFor="title" className="form-label">Title:</label>
                    <input
                        type="text"
                        id="title"
                        className="form-control"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                    />
                </div>
                <div className="mb-3">
                    <label htmlFor="description" className="form-label">Description:</label>
                    <textarea
                        id="description"
                        className="form-control"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                    />
                </div>
                <div className="mb-3">
                    <label htmlFor="dueDate" className="form-label">Due Date:</label>
                    <DatePicker
                        id="dueDate"
                        className="form-control"
                        selected={dueDate}
                        onChange={(date) => setDueDate(date)}
                    />
                </div>
                <button type="button" className="btn btn-primary" onClick={handleCreateTodo}>
                    Create To-Do
                </button>
            </form>
        </div>
    );
}

export default TodoForm;
