require('dotenv').config()

const express = require('express')
const mongoose = require('mongoose')
const analysisRoutes = require('./routes/analysis')
const userRoutes = require('./routes/user')

// express app
const app = express()

// Increase payload size limit for JSON requests
app.use(express.json({ limit: '10mb' }));

// Increase payload size limit for URL-encoded data
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// middleware
app.use(express.json())

app.use((req, res, next) => {
    console.log('%s %s', req.path, req.method)
    next()
})

// routes
app.use('/api/analysis', analysisRoutes)
app.use('/api/user', userRoutes)

// connect to db
mongoose.connect(process.env.MONGO_URI)
    .then(() => {
        // listen for requests
        app.listen(process.env.PORT, () => {
            console.log('connected to db & listening on port', process.env.PORT)
        })
    })
    .catch((error) => {
        console.log(error)
    })