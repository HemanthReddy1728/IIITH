import React from "react";
import { useState } from "react";
import { useSignup } from "../hooks/useSignup";
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

const SignUpForm = () => {
    //For Name
    const [name, setName] = useState("");
    const [isNameFocused, setNameFocus] = useState(false);

    //For email id
    const [email, setEmail] = useState("");
    const [isEmailFocused, setEmailFocus] = useState(false);
    const [isEmailValid, setEmailValid] = useState(false);

    //For first password
    const [firstPassword, setFirstPassword] = useState("");
    const [isFirstPasswordFocused, setFirstPasswordFocus] = useState(false);
    const [showFirstPassword, setShowFirstPassword] = useState(false);

    //For second password
    const [secondPassword, setSecondPassword] = useState("");
    const [isSecondPasswordFocused, setSecondPasswordFocus] = useState(false);
    const [showSecondPassword, setShowSecondPassword] = useState(false);

    //For custom login hook
    const { signup, error, isLoading } = useSignup();

    const handleSubmit = async (e) => {
        e.preventDefault();

        await signup(name.trim(), email, firstPassword);
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
        <div className="form-container sign-up-container">
            <form onSubmit={handleSubmit}>
                <div class="inline-container">
                    <div
                        class="image-container"
                        style={{ width: "200px", height: "160px" }}
                    ></div>
                    <h1>CREATE ACCOUNT</h1>
                    <div
                        class="image-container"
                        style={{ width: "200px", height: "160px" }}
                    ></div>
                </div>
                <ThemeProvider theme={theme}>
                    <TextField
                        margin="normal"
                        fullWidth
                        id="name"
                        name="name"
                        label="Name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        onFocus={() => setNameFocus(true)}
                        error={isNameFocused && name.trim() === ""}
                        helperText={isNameFocused && name.trim() === "" ? "Enter Name" : ""}
                    />

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

                    {/* For First Password */}
                    <TextField
                        margin="normal"
                        fullWidth
                        type={showFirstPassword ? "text" : "password"}
                        id="firstPassword"
                        name="firstPassword"
                        label="Password"
                        value={firstPassword}
                        onChange={(e) => setFirstPassword(e.target.value)}
                        onFocus={() => setFirstPasswordFocus(true)}
                        error={isFirstPasswordFocused && firstPassword === ""}
                        helperText={
                            isFirstPasswordFocused && firstPassword === ""
                                ? "Password required"
                                : ""
                        }
                        InputProps={{
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton
                                        onClick={() => setShowFirstPassword(!showFirstPassword)}
                                        edge="end"
                                    >
                                        {showFirstPassword ? <Visibility /> : <VisibilityOff />}
                                    </IconButton>
                                </InputAdornment>
                            ),
                        }}
                    />

                    {/* For Second Password */}
                    <TextField
                        margin="normal"
                        fullWidth
                        type={showSecondPassword ? "text" : "password"}
                        id="secondPassword"
                        name="secondPassword"
                        label="Re Enter Password"
                        value={secondPassword}
                        onChange={(e) => setSecondPassword(e.target.value)}
                        onFocus={() => setSecondPasswordFocus(true)}
                        error={
                            isSecondPasswordFocused &&
                            (secondPassword === "" || secondPassword !== firstPassword)
                        }
                        helperText={
                            !isSecondPasswordFocused
                                ? ""
                                : secondPassword !== ""
                                    ? secondPassword === firstPassword
                                        ? ""
                                        : "Passwords doesn't Match"
                                    : "Re Enter Password required"
                        }
                        InputProps={{
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton
                                        onClick={() => setShowSecondPassword(!showSecondPassword)}
                                        edge="end"
                                    >
                                        {showSecondPassword ? <Visibility /> : <VisibilityOff />}
                                    </IconButton>
                                </InputAdornment>
                            ),
                        }}
                    />
                </ThemeProvider>

                <button disabled={isLoading} className="custom-button" style={{ width: "200px", margin: "15px" }}>Sign UP</button>
                {error && <div className="error">{error}</div>}
            </form>
        </div>
    );
};

export default SignUpForm;
