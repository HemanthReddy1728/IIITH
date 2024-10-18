require('dotenv').config();
const express = require("express");
const mongoose = require('mongoose');
const cors = require("cors");
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cookieParser = require('cookie-parser');
const UserModel = require('./models/User');

const app = express();
app.use(express.json());
app.use(cors({
    origin: ["http://localhost:5173"],
    methods: ["GET", "POST"],
    credentials: true
}));
app.use(cookieParser());

// Use environment variables for sensitive information
const { MONGODB_URI, JWT_SECRET } = process.env;

mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true });

const verifyUser = (req, res, next) => {
    const token = req.cookies.token;
    if (!token) {
        return res.status(401).json({ error: "Token is missing" });
    } else {
        jwt.verify(token, JWT_SECRET, (err, decoded) => {
            if (err) {
                return res.status(401).json({ error: "Error with token" });
            } else {
                req.user = decoded;
                next();
            }
        });
    }
};

app.get('/create-todo', verifyUser, (req, res) => {
    res.json("Success");
});

app.post('/register', (req, res) => {
    const { name, email, password } = req.body;
    bcrypt.hash(password, 10)
        .then(hash => {
            UserModel.create({ name, email, password: hash })
                .then(user => {
                    const token = jwt.sign({ email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '1d' });
                    res.cookie('token', token);
                    return res.json({ Status: "Success", role: user.role });
                })
                .catch(err => {
                    console.error(err); // Log the error
                    return res.status(500).json({ error: "An error occurred while creating the user." });
                });
        })
        .catch(err => {
            console.error(err); // Log the error
            return res.status(500).json({ error: "An error occurred while hashing the password." });
        });
});


app.post('/login', (req, res) => {
    const { email, password } = req.body;
    UserModel.findOne({ email: email })
        .then(user => {
            if (user) {
                bcrypt.compare(password, user.password, (err, response) => {
                    if (response) {
                        const token = jwt.sign({ email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '1d' });
                        res.cookie('token', token);
                        return res.json({ Status: "Success", role: user.role });
                    } else {
                        return res.status(401).json({ error: "The password is incorrect" });
                    }
                });
            } else {
                return res.status(404).json({ error: "No record existed" });
            }
        })
        .catch(err => {
            console.error(err); // Log the error
            return res.status(500).json({ error: "An error occurred while logging in." });
        });
});


const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
