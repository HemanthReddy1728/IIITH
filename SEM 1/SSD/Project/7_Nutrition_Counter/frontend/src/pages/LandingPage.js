import React, { useState } from "react";
import SignInForm from "../components/SignIn";
import SignUpForm from "../components/SignUp";

import "../css/styles.css";

export default function LandingPage() {
    const [type, setType] = useState("signIn");
    const handleOnClick = text => {
        if (text !== type) {
            setType(text);
            return;
        }
    };
    const containerClass =
        "container " + (type === "signUp" ? "right-panel-active" : "");
    return (
        <div className="App">
            <div className={containerClass} id="container">
                <SignUpForm />
                <SignInForm />
                <div className="overlay-container">
                    <div className="overlay">
                        <div class="blur-overlay"></div>
                        <div className="overlay-panel overlay-left">
                            <h1>Hello, Welcome to Nutrition Counter!</h1>
                            <h5>Already have an account?</h5>
                            <h5>Access it by clicking SIGN IN button below.</h5>
                            <button
                                className="ghost"
                                id="signIn"
                                onClick={() => handleOnClick("signIn")}
                                style={{ width: '200px', margin: '15px' }}
                            >
                                Sign In
                            </button>
                        </div>
                        <div className="overlay-panel overlay-right">
                            <h1>Hello, Welcome to Nutrition Counter!</h1>
                            <h5>Don't have an account yet?</h5>
                            <h5>Create one by clicking SIGN UP button below.</h5>

                            <button
                                className="ghost "
                                id="signUp"
                                onClick={() => handleOnClick("signUp")}
                                style={{ width: '200px', margin: '15px' }}
                            >
                                Sign Up
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}