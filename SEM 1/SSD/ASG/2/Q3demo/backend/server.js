require('dotenv').config();

const express = require('express');
//mongoose app
const mongoose = require('mongoose');
const workoutRoutes = require('./routes/workouts');
//express app
const app = express();
//middleware
app.use(express.json());

app.use((req, res, next) => {
    console.log('%s %s', req.path, req.method);
    next();
});

// routes
app.use('/api/workouts', workoutRoutes);

//connect to mongodb
mongoose.connect(process.env.MONGO_URI)
    .then(() => {
        console.log('Connected to MongoDB');
        // app.get('/', (req, res) => { res.json({ message: 'Hello World' }) });
//         // app.listen(5000, () => console.log('Server running and listening on port 5000'));
        app.listen(process.env.PORT, () => console.log('Server running and listening on port', process.env.PORT));
    })
    .catch((err) => console.log(err));

// console.log('Connected to MongoDB');
// app.get('/', (req, res) => { res.json({ message: 'Hello World' }) });
// app.listen(5000, () => console.log('Server running and listening on port 5000'));
// app.listen(process.env.PORT, () => console.log('Server running and listening on port', process.env.PORT));