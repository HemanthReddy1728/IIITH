import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Dashboard() {
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [date, setDate] = useState('');
    const [alertMessage, setAlertMessage] = useState('');
    const currentDate = new Date();
    const navigate = useNavigate();

    useEffect(() => {
        // Check for the presence of a cookie containing authentication information here
        const token = document.cookie.split('; ').find(row => row.startsWith('token='));

        if (!token) {
            // Redirect the user to the login page if the token is missing
            navigate('/login');
        }
    }, [navigate]);

    const handleCreate = async () => {
        const currentDate = new Date().toISOString().slice(0, 10);

        if (date < currentDate) {
            setAlertMessage('Please select a future date for completion.');
        } else {
            try {
                const response = await axios.post('http://localhost:3001/dashboard', {
                    title,
                    description,
                    date,
                    status: 1,
                });

                if (response.data.Status === 'Success') {
                    setAlertMessage(`To-Do Item Created:
        Title: ${title}
        Description: ${description}
        Completion Date: ${date}`);
                    setTitle('');
                    setDescription('');
                    setDate('');
                }
            } catch (error) {
                console.error('Error creating To-Do item:', error);
                setAlertMessage('Error creating To-Do item');
            }
        }
    };

    const handleLogout = () => {
        document.cookie = 'token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
        navigate('/login');
    };

    return (
        <div className="todo-container">
            <h1>Create To-Do Item</h1>
            <span>
                <button className="logout-button" onClick={handleLogout}>
                    Logout
                </button>
                <button onClick={() => navigate('/complete-todo')}>Go to Complete To-Do</button>
                <button onClick={() => navigate('/pending-todo')}>Go to Pending To-Do</button>
            </span>
            <div className="form-group">
                <label htmlFor="title">Title:</label>
                <input
                    type="text"
                    id="title"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    required
                />
            </div>
            <div className="form-group">
                <label htmlFor="description">Description:</label>
                <input
                    type="text"
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                />
            </div>
            <div className="form-group">
                <label htmlFor="date">Completion Date:</label>
                <input
                    type="date"
                    id="date"
                    value={date}
                    min={currentDate.toISOString().split('T')[0]}
                    onChange={(e) => setDate(e.target.value)}
                    required
                />
            </div>
            <button className="create-button" onClick={handleCreate}>
                Create
            </button>

            {alertMessage && (
                <div className="alert-box">{alertMessage}</div>
            )}
        </div>
    );
}

export default Dashboard;
