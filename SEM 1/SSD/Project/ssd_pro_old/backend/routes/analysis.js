const express = require('express')

const {
    getAllAnalysis,
    getSingleAnalysis,
    createAnalysis,
    deleteAnalysis
} = require('../controllers/analysisController')

//This is used for protecting the api routes
const requireAuth = require('../middleware/requireAuth')

const router = express.Router()

// require auth for all workout routes
router.use(requireAuth)

// GET all analysis
router.get('/', getAllAnalysis)

//GET a single analysis
router.get('/:id', getSingleAnalysis)

// POST a new analysis
router.post('/', createAnalysis)

// DELETE an analysis
router.delete('/:id', deleteAnalysis)

// We don't need it
// UPDATE a workout
// router.patch('/:id', updateWorkout)


module.exports = router