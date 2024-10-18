import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Home.css"; // Import the CSS file

function Home() {
  const pageStyle = {
    backgroundColor: "#00d4ff",
    minHeight: "100vh",
  };

  const containerStyle = {
    display: "flex",
    justifyContent: "space-between", // Adjust alignment
    alignItems: "center", // Center vertically
    padding: "1rem",
  };

  const navigate = useNavigate();

  const handleLoginClick = () => {
    navigate("/login");
  };

  return (
    <div style={pageStyle} className="home-container">
      <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container" style={containerStyle}>
          <Link className="navbar-brand" to="/">
            <span className="brand-title">SSD LAB</span>
          </Link>
          <ul className="navbar-nav ml-auto">
            <li className="nav-item">
              <button className="btn btn-primary" onClick={handleLoginClick}>
                Login
              </button>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/register">
                <button className="btn btn-success">Register</button>
              </Link>
            </li>
          </ul>
        </div>
      </nav>
    </div>
  );
}

export default Home;
