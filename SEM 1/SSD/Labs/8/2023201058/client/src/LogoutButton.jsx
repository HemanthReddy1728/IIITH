// LogoutButton.jsx
import React from 'react';
import { useHistory } from 'react-router-dom';
import axios from 'axios';

function LogoutButton() {
    const history = useHistory();

    const handleLogout = async () => {
        try {
            const response = await axios.get('http://localhost:3001/logout');
            if (response.data.Status === 'Success') {
                // Clear the authentication token from cookies
                document.cookie = 'token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';

                // Redirect to the login page
                history.push('/login');
            }
        } catch (error) {
            console.error('Error logging out:', error);
        }
    };

    return (
        <button onClick={handleLogout}>Logout</button>
    );
}

export default LogoutButton;
