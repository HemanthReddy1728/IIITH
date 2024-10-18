import React, { useEffect } from "react";
import axios from 'axios';
import { useNavigate } from "react-router-dom";

function Home() {
    axios.defaults.withCredentials = true; 
    const navigate = useNavigate();
    useEffect(() => {
        axios.get('http://localhost:3001/home')
        .then(result => { console.log(result); 
            if (result.data !== "Success") {
                navigate('/login');
            }
            // navigate('/'); 
        })
        .catch(err => console.log(err));
    }, []);
    return (
        <h2>Home page component</h2>
    );
}

export default Home;