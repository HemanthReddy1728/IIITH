import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function PendingToDo() {
    const [incompleteTodos, setIncompleteTodos] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        // Fetch incomplete To-Do items for the logged-in user
        const fetchData = async () => {
            try {
                // Replace 'userId' with the actual ID of the logged-in user
                const response = await axios.get(`http://localhost:3001/incomplete-todos?userId=${userId}`);
                if (response.data.Status === 'Success') {
                    setIncompleteTodos(response.data.todos);
                }
            } catch (error) {
                console.error('Error fetching incomplete To-Do items:', error);
            }
        };

        fetchData();
    }, [userId]);

    const markTodoDone = async (id) => {
        try {
            const response = await axios.put(`http://localhost:3001/mark-todo-done/${id}`);
            if (response.data.Status === 'Success') {
                // Update the UI to reflect the status change
                setIncompleteTodos((todos) => todos.filter((todo) => todo._id !== id));
            }
        } catch (error) {
            console.error('Error marking To-Do item as done:', error);
        }
    };

    const deleteTodo = async (id) => {
        try {
            const response = await axios.delete(`http://localhost:3001/delete-todo/${id}`);
            if (response.data.Status === 'Success') {
                // Remove the deleted To-Do item from the UI
                setIncompleteTodos((todos) => todos.filter((todo) => todo._id !== id));
            }
        } catch (error) {
            console.error('Error deleting To-Do item:', error);
        }
    };

    const handleLogout = () => {
        // Handle the Logout action
        // For example, remove the token from cookies and navigate to the login page
        document.cookie = 'token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
        navigate('/login');
    };

    return (
        <div className="pending-todo-container">
            <h1>Pending To-Do Items</h1>
            <ul>
                {incompleteTodos.map((todo) => (
                    <li key={todo._id}>
                        <div>
                            <strong>Title:</strong>
                            <input type="text" value={todo.title} readOnly />
                        </div>
                        <div>
                            <strong>Description:</strong>
                            <input type="text" value={todo.description} readOnly />
                        </div>
                        <div>
                            <strong>Due Date:</strong>
                            <input type="text" value={todo.DueDate} readOnly />
                        </div>
                        <div>
                            <strong>Status:</strong>
                            <input type="text" value="Incomplete" readOnly />
                        </div>
                        <button onClick={() => markTodoDone(todo._id)}>Done</button>
                        <button onClick={() => deleteTodo(todo._id)}>Delete</button>
                    </li>
                ))}
            </ul>
            <button onClick={() => navigate('/dashboard')}>Go to Dashboard</button>
            <button onClick={() => navigate('/complete-todo')}>Go to Complete To-Do</button>
            <button onClick={handleLogout}>Logout</button>
        </div>
    );
}

export default PendingToDo;
