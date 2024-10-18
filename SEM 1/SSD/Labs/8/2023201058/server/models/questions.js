const mongoose = require('mongoose')

const QuestionSchema = new mongoose.Schema({

questions: String,
answers: String,


})

const questionModel = mongoose.model("questions",QuestionSchema)
module.exports = questionModel