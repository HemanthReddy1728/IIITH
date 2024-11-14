const express = require("express")
const mongoose = require('mongoose')
const cors = require("cors")
const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const cookieParser = require('cookie-parser')
const UserModel = require('./models/User')
const questionModel = require('./models/questions')
const HistoryModel = require('./models/Hist')
const ToDoModel = require('./models/todo')
const lusca = require('lusca');
const app = express()
app.use(express.json())
app.use(cors({
    origin: ["http://localhost:5173"],
    methods: ["GET", "POST", "DELETE"],
    credentials: true
}))
app.use(cookieParser())
app.use(lusca.csrf())
mongoose.connect('mongodb://127.0.0.1:27017/GPTLITEN');


const verifyUser = (req, res, next) => {
    const token = req.cookies.token;
    if (!token) {
        return res.json("Token is missing");
    } else {
        jwt.verify(token, "jwt-secret-key", (err, decoded) => {
            if (err) {
                return res.json("Error with token");
            } else {
                // Allow access for all roles
                req.user = decoded
                next();
            }
        });
    }
};


app.get("/questions", async (req, res) => {
    try {
        const questions = await questionModel.find();
        res.json(questions);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Server error" });
    }
});


app.post('/register', async (req, res) => {
    const { name, password, role } = req.body;

    try {
        // Check if a user with the same name already exists
        const existingUser = await UserModel.findOne({ name });

        if (existingUser) {
            return res.status(400).json({ error: 'Name is already in use' });
        }

        // Hash the password
        const hash = await bcrypt.hash(password, 10);

        // Create a new user and save it to the database
        const newUser = await UserModel.create({ name, password: hash, role });

        res.json("Success");
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Server error' });
    }
});


app.post('/dashboard', verifyUser, async (req, res) => {
    const { title, description, date, status } = req.body;

    try {
        // Create a new To-Do item and save it to the database
        const newTodo = await ToDoModel.create({ title, description, date, status });

        res.json({ Status: 'Success', message: 'To-Do Item Created and Saved to Database' });
    } catch (error) {
        console.error('Error creating the To-Do item:', error);
        res.status(500).json({ Status: 'Error', message: 'Server error' });
    }
});

app.get('/completed-todos', verifyUser, async (req, res) => {
    try {
        // Fetch completed To-Do items for the current user
        const completedTodos = await ToDoModel.find({ status: 0, userId: req.user.email });

        res.json({ Status: 'Success', todos: completedTodos });
    } catch (error) {
        console.error('Error fetching completed To-Do items:', error);
        res.status(500).json({ Status: 'Error', message: 'Server error' });
    }
});

// Fetch incomplete To-Do items for the logged-in user
app.get('/incomplete-todos', verifyUser, async (req, res) => {
    try {
        const incompleteTodos = await ToDoModel.find({ Status: 1, userId: req.user.email });
        res.json({ Status: 'Success', todos: incompleteTodos });
    } catch (error) {
        console.error('Error fetching incomplete To-Do items:', error);
        res.status(500).json({ Status: 'Error', message: 'Server error' });
    }
});

// Mark a To-Do item as done (status = 0 for complete)
app.put('/mark-todo-done/:id', verifyUser, async (req, res) => {
    try {
        const todoId = req.params.id;
        const updatedTodo = await ToDoModel.findByIdAndUpdate(todoId, { Status: 0 }, { new: true });

        if (updatedTodo) {
            res.json({ Status: 'Success', message: 'To-Do item marked as done' });
        } else {
            res.status(404).json({ Status: 'Error', message: 'To-Do item not found' });
        }
    } catch (error) {
        console.error('Error marking To-Do item as done:', error);
        res.status(500).json({ Status: 'Error', message: 'Server error' });
    }
});

// Delete a To-Do item
app.delete('/delete-todo/:id', verifyUser, async (req, res) => {
    try {
        const todoId = req.params.id;
        const deletedTodo = await ToDoModel.findByIdAndRemove(todoId);

        if (deletedTodo) {
            res.json({ Status: 'Success', message: 'To-Do item deleted' });
        } else {
            res.status(404).json({ Status: 'Error', message: 'To-Do item not found' });
        }
    } catch (error) {
        console.error('Error deleting To-Do item:', error);
        res.status(500).json({ Status: 'Error', message: 'Server error' });
    }
});



app.post('/admin', (req, res) => {
    const { questions, answers } = req.body;

    // Check if the question already exists in the database
    questionModel.findOne({ questions })
        .then((existingQuestion) => {
            if (existingQuestion) {
                // If the question exists, update its answer
                existingQuestion.answers = answers;
                existingQuestion.save()
                    .then(() => {
                        res.json({ Status: 'Success', message: 'QnA updated successfully' });
                    })
                    .catch((err) => {
                        res.status(500).json({ Status: 'Error', message: err.message });
                    });
            } else {
                // If the question does not exist, create a new QnA document
                questionModel.create({ questions, answers })
                    .then(() => {
                        res.json({ Status: 'Success', message: 'QnA added successfully' });
                    })
                    .catch((err) => {
                        res.status(500).json({ Status: 'Error', message: err.message });
                    });
            }
        })
        .catch((err) => {
            res.status(500).json({ Status: 'Error', message: err.message });
        });
});


app.post('/interactions', (req, res) => {
    const { question, answer } = req.body;
    const token = req.cookies.token; // Get the JWT token from cookies

    // Verify and decode the token to get user information
    jwt.verify(token, 'jwt-secret-key', (err, decoded) => {
        if (err) {
            return res.status(401).json({ Status: 'Error', message: 'Invalid token' });
        }

        const { email, role } = decoded;

        // Save the interaction data to the History model along with user information
        const interaction = new HistoryModel({ userId: email, question, answer, role });

        interaction.save()
            .then(() => {
                res.json({ Status: 'Success', message: 'Interaction recorded successfully' });
            })
            .catch((err) => {
                res.status(500).json({ Status: 'Error', message: err.message });
            });
    });
});

// Route to fetch interaction history
app.get('/history', verifyUser, async (req, res) => {
    try {
        // Fetch interaction history for the current user
        const history = await HistoryModel.find({ userId: req.user.email });
        res.json(history);
    } catch (error) {
        console.error("Error fetching interaction history:", error);
        res.status(500).json({ error: "Server error" });
    }
});

app.delete('/history', verifyUser, async (req, res) => {
    try {
        // Delete interaction history for the current user
        await HistoryModel.deleteMany({ userId: req.user.email });
        res.json({ message: 'Interaction history deleted successfully' });
    } catch (error) {
        console.error('Error deleting interaction history:', error);
        res.status(500).json({ error: 'Server error' });
    }
});




app.post('/login', (req, res) => {
    const { name, password } = req.body;

    UserModel.findOne({ name: { $eq: name } })
        .then(user => {
            if (user) {
                bcrypt.compare(password, user.password, (err, response) => {
                    if (response) {
                        const token = jwt.sign({ email: user.email, role: user.role },
                            "jwt-secret-key", { expiresIn: '1d' });
                        res.cookie('token', token);
                        return res.json({ Status: "Success", role: user.role });
                    } else {
                        return res.status(800).json("The password is incorrect");
                    }
                });
            } else {
                return res.status(900).json("No record existed");
            }
        });
});

app.get('/logout', verifyUser, (req, res) => {
    try {
        res.clearCookie('token'); 
        res.json({ Status: 'Success', message: 'Logged out' });
    } catch (error) {
        console.error('Error logging out:', error);
        res.status(500).json({ Status: 'Error', message: 'Server error' });
    }
});


app.listen(3001, () => {
    console.log("Server is Running")
})