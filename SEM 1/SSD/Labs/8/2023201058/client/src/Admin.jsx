import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import 'bootstrap/dist/css/bootstrap.min.css'
import axios from 'axios'
import './admin.css';
import Cookies from 'js-cookie';

function Admin() {
  const [questions, setQuestion] = useState("");
  const [answers, setAnswer] = useState("");
  const [successMessage, setSuccessMessage] = useState(""); // New state for success message
  const navigate = useNavigate();


 
  const handleSubmit = (e) => {
    e.preventDefault();
    axios.post('http://localhost:3001/admin', { questions, answers })
      .then(res => {
        console.log("posting: " + res.data);
        if (res.data.Status === "Success") {
          setSuccessMessage("Question added successfully!"); // Set success message
          // Clear the input fields
          setQuestion("");
          setAnswer("");
        }
      })
      .catch(err => console.log(err))
  }
  

  const handleLogout = () => {
    // Implement your logout logic here, such as clearing the user's session or token.
    // Example: Clearing a JWT token stored in localStorage
    localStorage.removeItem("token");

    // Navigate to the home page after logout
    navigate("/");
  };

  return (
   
    <div className="container mt-5">
      
      <h2 className="mb-4">Add Questions and Answers (Admin)</h2>
      {successMessage && ( // Render the success message if it exists
        <div className="alert alert-success">{successMessage}</div>
      )}
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label htmlFor="question" className="form-label">
            Question:
          </label>
          <textarea
            id="question"
            className="form-control"
            rows="4"
            value={questions}
            onChange={(e) => setQuestion(e.target.value)}
            required
          ></textarea>
        </div>
        <div className="mb-3">
          <label htmlFor="answer" className="form-label">
            Answer:
          </label>
          <textarea
            id="answer"
            className="form-control"
            rows="4"
            value={answers}
            onChange={(e) => setAnswer(e.target.value)}
            required
          ></textarea>
        </div>
        <button type="submit" className="btn btn-primary">
          Add Q&A
        </button>
        <div>
      <div className="top-right" >
        <button type="button" className="btn btn-dark" onClick={handleLogout}>Logout</button>
      </div>
      {/* Your content here */}
    </div>


      </form>
    </div>
  );
}

export default Admin;
