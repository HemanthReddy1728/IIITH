import React, { useState } from 'react'
import './style.css'
import 'bootstrap/dist/css/bootstrap.min.css'
import { Link, useNavigate } from 'react-router-dom'
import axios from 'axios'

function Register() {
    const [username, setUsername] = useState()
    const [role, setRole] = useState()
    const [email, setEmail] = useState()
    const [password, setPassword] = useState()
    const navigate = useNavigate()

    const handleSubmit = (e) => {
        e.preventDefault()
        axios.post('http://localhost:3001/register', { username, role, email, password })
            .then(res => navigate('/login'))
            .catch(err => console.log(err))
    }

    return (
        <div className='signup_container d-flex justify-content-center align-items-center bg-secondary vh-100'>
            <div className='signup_form bg-white p-3 rounded w-25'>
                <h2>Sign Up</h2>
                <br />
                <form onSubmit={handleSubmit}>
                    <div className="mb-3">
                        <label htmlFor="name">Username:</label> <br />
                        <input type="text" placeholder='Enter Username'
                            onChange={e => setUsername(e.target.value)} />
                    </div>
                    <br />
                    <div className="mb-3">
                        <label htmlFor="role">Role:</label><br />
                        <input type="text" placeholder='Enter Role'
                            onChange={e => setRole(e.target.value)} />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="email">Email:</label><br />
                        <input type="email" placeholder='Enter Email'
                            onChange={e => setEmail(e.target.value)} />
                    </div>
                    <br />
                    <div className="mb-3">
                        <label htmlFor="password">Password:</label><br />
                        <input type="password" placeholder='********'
                            onChange={e => setPassword(e.target.value)} />
                    </div>
                    <button className='signup_btn btn btn-success w-100 rounded-0'>Sign up</button>
                </form>
                <br></br>
                <p>Already have account?</p>
                <Link to="/login" className="btn btn-default border w-100 bg-light rounded-0 "><button>Login</button></Link>
            </div>
        </div>
    );
}

export default Register