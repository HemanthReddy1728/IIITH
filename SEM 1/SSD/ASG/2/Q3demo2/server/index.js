const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const EmployeeModel = require('./models/Employee');

const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const cookieParser = require('cookie-parser');

const app = express();
app.use(express.json());
app.use(cors({
    origin: ["http://localhost:5173"],
    methods: ["GET", "POST"],
    credentials: true
}));
app.use(cookieParser());

mongoose.connect('mongodb://localhost:27017/employee');

const algorithm = 'aes-256-ctr';
const secretKey = 'vOVH6sdmpNWjRRIqCc7rdxs01lwHzfr3';

const encrypt = (text) => {
    const cipher = crypto.createCipher(algorithm, secretKey);
    const encrypted = Buffer.concat([cipher.update(text), cipher.final()]);
    return encrypted.toString('hex');
};

const decrypt = (hash) => {
    const decipher = crypto.createDecipher(algorithm, secretKey);
    const decrypted = Buffer.concat([decipher.update(Buffer.from(hash, 'hex')), decipher.final()]);
    return decrypted.toString();
};

const verifyUser = (req, res, next) => {
    const encryptedToken = req.cookies.token;
    const token = decrypt(encryptedToken);
    console.log(token);

    if(!token) {
        return res.json("No Token Available");
    }
    else
    {
        jwt.verify(token, "jwt-secret-key", (err, decoded) => {
            if (err) {
                return res.json("Invalid Wrong Token");
            } else {
                // req.decoded = decoded;
                next();
            }
        });
    }
}

app.get('/home', verifyUser, (req, res) => {
    return res.json("Success");
});

app.post('/register', (req, res) => {
    // EmployeeModel.create(req.body)
    // .then(employees => res.json(employees))
    // .catch(err => res.json(err));
    const { name, email, password } = req.body;
    bcrypt.hash(password, 10)
    .then(hash => {
        EmployeeModel.create({ name, email, password:hash })
        .then(employees => res.json(employees))
        .catch(err => res.json(err));
    }).catch(err => console.log(err.message));
});

app.post('/login', (req, res) => {
    const { email, password } = req.body;
    EmployeeModel.findOne({ email })
    .then(user => {
        // if (user)
        // {
        //     if (user.password === password) {
        //         res.json("Success");
        //     } else {
        //         res.json("Invalid Password");
        //     }
        // }
        if (user) {
            bcrypt.compare(password, user.password, (err, response) => {
                if (err) {
                    // res.json("Invalid Password");
                } 
                if (response) {
                    const token = jwt.sign({ email: user.email }, "jwt-secret-key", { expiresIn: "1d" });
                    const encryptedToken = encrypt(token);
                    res.cookie('token', encryptedToken);
                    res.json("Success");
                }
                else {
                    res.json("Invalid Password");
                }
            })
        }
        else {
            res.json("Invalid Credentials");
        }
    })
    
}); 

app.listen(3001, () => {
    console.log('Server running on port 3001');
});