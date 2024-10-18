import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from 'axios';

import './Login.css';

function Login() {
    const [name, setName] = useState(""); // Change 'email' to 'name'
    const [password, setPassword] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const navigate = useNavigate();

    axios.defaults.withCredentials = true;

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await axios.post('http://localhost:3001/login', { name, password }); // Change 'email' to 'name'

            if (response.data.Status === "Success") {
                if (response.data.role === "admin") {
                    navigate('/admin');
                } else {
                    navigate('/dashboard');
                }
            }
        } catch (error) {
            if (error.response && error.response.status === 800) {
                setErrorMessage('Password is incorrect!');
            } else {
                console.error(error);
                setErrorMessage('No such record');
            }
        }
    };

    return (
        <div className="login-container">
            <div className="login-box">
                <h2 className="login-title">Login</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="name" className="login-label"> {/* Change 'email' to 'name' */}
                            Name
                        </label>
                        <input
                            type="text"
                            placeholder="Enter Name"
                            autoComplete="off"
                            name="name"
                            className="form-control login-input"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            required
                        />

                    </div>
                    <div className="form-group">
                        <label htmlFor="password" className="login-label">
                            Password
                        </label>
                        <input
                            type="password"
                            placeholder="Enter Password"
                            name="password"
                            className="form-control login-input"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <button type="submit" className="btn btn-primary login-button">
                        Login
                    </button>
                    {errorMessage && (
                        <div className="alert alert-danger mt-3" role="alert">
                            {errorMessage}
                        </div>
                    )}
                </form>
                <p className="login-text">Already Have an Account?</p>
                <Link to="/register" className="btn btn-secondary login-signup-button">
                    Sign Up
                </Link>
            </div>
        </div>
    );
}

export default Login;
