import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";

import "./Signup.css";

function Signup() {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("visitor");
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();

    // Check if password and confirm password match
    if (password !== confirmPassword) {
      setErrorMessage("Password and Confirm Password do not match");
      return;
    }

    axios
      .post("http://localhost:3001/register", {
        name,
        password,
        role,
      })
      .then((res) => {
        navigate("/login");
      })
      .catch((error) => {
        if (error.response && error.response.status === 400) {
          // Handle the case where the name is already in use
          setErrorMessage("Name is already in use. Please use a different name.");
        } else {
          // Handle other errors
          console.error(error);
          setErrorMessage("An error occurred while registering. Please try again later.");
        }
      });
  };

  return (
    <div className="signup-container">
      <div className="signup-box">
        <h2 className="signup-title">Register</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name" className="signup-label">
              Name
            </label>
            <input
              type="text"
              placeholder="Enter Name"
              autoComplete="off"
              name="name"
              className="form-control signup-input"
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password" className="signup-label">
              Password
            </label>
            <input
              type="password"
              placeholder="Enter Password"
              name="password"
              className="form-control signup-input"
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword" className="signup-label">
              Confirm Password
            </label>
            <input
              type="password"
              placeholder="Confirm Password"
              name="confirmPassword"
              className="form-control signup-input"
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="btn btn-success signup-button">
            Register
          </button>
          {errorMessage && (
            <div className="alert alert-danger mt-3" role="alert">
              {errorMessage}
            </div>
          )}
        </form>
        <Link to="/login" className="btn btn-success signup-login-button">
          Login
        </Link>
      </div>
    </div>
  );
}

export default Signup;
