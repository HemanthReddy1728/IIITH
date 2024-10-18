import React from "react";
import { useState } from "react";
import { useLogin } from "../hooks/useLogin";
import TextField from "@mui/material/TextField";
import { createTheme, ThemeProvider } from "@mui/material/styles";

import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";

const theme = createTheme({
    palette: {
        primary: {
            main: "#1aac83",
        },
    },
});

const SignInForm = () => {
    //For email id
    const [email, setEmail] = useState("");
    const [isEmailFocused, setEmailFocus] = useState(false);
    const [isEmailValid, setEmailValid] = useState(false);

    //For password
    const [password, setPassword] = useState("");
    const [isPasswordFocused, setPasswordFocus] = useState(false);
    const [showPassword, setShowPassword] = useState(false);

    //For custom login hook
    const { login, error, isLoading } = useLogin();

    const handleSubmit = async (e) => {
        e.preventDefault();

        await login(email, password);
    };

    const handleEmailChange = (event) => {
        const newEmail = event.target.value;
        setEmail(newEmail);

        // Custom email format validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        const isValidEmail = emailRegex.test(newEmail);
        setEmailValid(isValidEmail);
    };

    return (
        <div className="form-container sign-in-container">
            <form onSubmit={handleSubmit}>
                <div class="image-container" style={{ width: '300px', height: '280px' }}></div>
                <h1>SIGN IN</h1>
                <ThemeProvider theme={theme}>
                    <TextField
                        margin="normal"
                        fullWidth
                        id="email"
                        name="email"
                        label="Email Address"
                        value={email.trim()}
                        onChange={handleEmailChange}
                        onFocus={() => setEmailFocus(true)}
                        error={isEmailFocused && (email.trim() === "" || !isEmailValid)}
                        helperText={
                            !isEmailFocused
                                ? ""
                                : email.trim() !== ""
                                    ? isEmailValid
                                        ? ""
                                        : "Please enter Vaild Email ID"
                                    : "Email ID required"
                        }
                    />

                    <TextField
                        margin="normal"
                        fullWidth
                        type={showPassword ? "text" : "password"}
                        id="password"
                        name="password"
                        label="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        onFocus={() => setPasswordFocus(true)}
                        error={isPasswordFocused && password === ""}
                        helperText={
                            isPasswordFocused && password === "" ? "Password required" : ""
                        }
                        InputProps={{
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton
                                        onClick={() => setShowPassword(!showPassword)}
                                        edge="end"
                                    >
                                        {showPassword ? <Visibility /> : <VisibilityOff />}
                                    </IconButton>
                                </InputAdornment>
                            ),
                        }}
                    />
                </ThemeProvider>

                <button disabled={isLoading} className="custom-button" style={{ width: "200px", margin: "15px" }}>Sign In</button>
                {error && <div className="error">{error}</div>}
            </form>
        </div>
    );
};

export default SignInForm;
