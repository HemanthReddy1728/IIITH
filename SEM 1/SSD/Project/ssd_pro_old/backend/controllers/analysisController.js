const Analysis = require('../models/analysisModel')
const mongoose = require('mongoose')

// get all analysis by a user
const getAllAnalysis = async (req, res) => {
    const user_id = req.user._id

    const analysis = await Analysis.find({ user_id }).sort({ createdAt: -1 })

    res.status(200).json(analysis)
}

// get a single analysis
const getSingleAnalysis = async (req, res) => {
    const { id } = req.params

    if (!mongoose.Types.ObjectId.isValid(id)) {
        return res.status(404).json({ error: 'No such Analysis' })
    }

    const analysis = await Analysis.findById(id)

    if (!analysis) {
        return res.status(404).json({ error: 'No such Analysis' })
    }

    res.status(200).json(analysis)
}


// create new analysis
const createAnalysis = async (req, res) => {
    const { name, size, type, file } = req.body

    //We won't need it
    // let emptyFields = []

    // if(!type) {
    //   emptyFields.push('type')
    // }
    // if(!file) {
    //   emptyFields.push('file')
    // }
    // if(emptyFields.length > 0) {
    //   return res.status(400).json({ error: 'Please fill in all the fields', emptyFields })
    // }

    // add doc to db
    try {
        const user_id = req.user._id
        console.log({ name, size, type, file, user_id })
        const analysis = await Analysis.create({ name, size, type, file, user_id })
        res.status(200).json(analysis)
    } catch (error) {
        res.status(400).json({ error: error.message })
    }
}

// delete an analysis
const deleteAnalysis = async (req, res) => {
    const { id } = req.params

    if (!mongoose.Types.ObjectId.isValid(id)) {
        return res.status(404).json({ error: 'No such Analysis' })
    }

    const analysis = await Analysis.findOneAndDelete({ _id: id })

    if (!analysis) {
        return res.status(400).json({ error: 'No such Analysis' })
    }

    res.status(200).json(analysis)
}


//We won't need it
// update a workout
// const updateWorkout = async (req, res) => {
//   const { id } = req.params

//   if (!mongoose.Types.ObjectId.isValid(id)) {
//     return res.status(404).json({error: 'No such workout'})
//   }

//   const workout = await Workout.findOneAndUpdate({_id: id}, {
//     ...req.body
//   })

//   if (!workout) {
//     return res.status(400).json({error: 'No such workout'})
//   }

//   res.status(200).json(workout)
// }


module.exports = {
    getAllAnalysis,
    getSingleAnalysis,
    createAnalysis,
    deleteAnalysis
}