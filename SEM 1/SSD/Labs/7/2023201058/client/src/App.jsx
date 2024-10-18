import React from "react";
import Login from "./Login";
import Signup from "./Signup";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./Dashboard";
import Home from "./Home";
import TodoForm from "./TodoForm"; // Import the TodoForm component

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/register" element={<Signup />} />
                <Route path="/login" element={<Login />} />
                <Route path="/dashboard" element={<Dashboard />} />
                {/* Add a new route for the TodoForm */}
                <Route path="/create-todo" element={<TodoForm />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
