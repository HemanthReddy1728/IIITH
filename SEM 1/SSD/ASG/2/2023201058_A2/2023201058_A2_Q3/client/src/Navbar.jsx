import React, { useContext } from 'react'
import './style.css'
import { Link, useNavigate } from 'react-router-dom'
import { userContext } from './App'
import axios from 'axios'

function Navbar() {
    const user = useContext(userContext)
    const navigate = useNavigate()

    const handleLogout = () => {
        axios.get('http://localhost:3001/logout')
            .then(res => {
                if (res.data === "Success") {
                    // navigate(0)
                    window.location.href = "/register"
                }

            }).catch(err => console.log(err))
    }

    return (
        <div className='navbar-header'>
            <div><h3>Chat GPT Lite App</h3></div>
            <div>
                <Link to="/" className='link'>Home</Link>
                {
                    user.username ?
                        <Link to="/create" className='link'>Create</Link>
                        : <></>
                }
                {/* <a href="" className='link'>Contact</a> */}
            </div>
            <div>
            {
                user.username ?
                    <div>
                        <input type="button" onClick={handleLogout} value="Logout" className='btn_input' />
                    </div>
                    :
                    <div>
                        <h5><Link to="/register" className="link">Register</Link></h5>
                        {/* <span> | </span> */}|
                        <h5><Link to="/login" className="link">Login</Link></h5>
                    </div>
            }
            </div>
        </div>
    )
}

export default Navbar