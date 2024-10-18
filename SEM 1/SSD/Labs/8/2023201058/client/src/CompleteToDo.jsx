import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // Import useNavigate

function CompleteToDo() {
    const [completedTodos, setCompletedTodos] = useState([]);
    const navigate = useNavigate(); // Create a navigate function

    useEffect(() => {
        // Fetch completed To-Do items for the logged-in user
        const fetchData = async () => {
            try {
                // Replace 'userId' with the actual ID of the logged-in user
                const response = await axios.get(`http://localhost:3001/completed-todos?userId=${userId}`);
                if (response.data.Status === 'Success') {
                    setCompletedTodos(response.data.todos);
                }
            } catch (error) {
                console.error('Error fetching completed To-Do items:', error);
            }
        };

        fetchData();
    }, [userId]);

    const handleLogout = () => {
        // Handle the Logout action
        // For example, remove the token from cookies and navigate to the login page
        document.cookie = 'token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
        navigate('/login');
    };

    return (
        <div className="complete-todo-container">
            <h1>Completed To-Do Items</h1>
            <ul>
                {completedTodos.map((todo) => (
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
                            <input type="text" value="Complete" readOnly />
                        </div>
                    </li>
                ))}
            </ul>
            <button onClick={() => navigate('/dashboard')}>Go to Dashboard</button>
            <button onClick={() => navigate('/pending-todo')}>Go to Pending To-Do</button>
            <button onClick={handleLogout}>Logout</button> {/* Add the Logout button */}
        </div>
    );
}

export default CompleteToDo;
