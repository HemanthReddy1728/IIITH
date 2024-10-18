const mongoose = require('mongoose')

const Schema = mongoose.Schema

const analysisSchema = new Schema({

    name: {
        type: String,
        required: true
    },

    size: {
        type: Number,
        required: true
    },

    type: {
        type: String,
        required: true
    },

    file: {
        type: Buffer,  // Use Buffer type for binary data
        required: true
    },

    user_id: {
        type: String,
        required: true
    }
}, { timestamps: true })

module.exports = mongoose.model('Analysis', analysisSchema)