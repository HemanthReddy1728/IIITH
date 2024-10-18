const express = require('express');
// const Workout = require('../models/workoutModel');
const { getWorkout, getWorkouts, createWorkout, deleteWorkout, updateWorkout } = require('../controllers/workoutController');
const router = express.Router();

// // GET all workouts
// router.get('/', (req, res) => {
//     res.json({message: 'GET all workouts'});
// });

// // GET a single workouts
// router.get('/:id', (req, res) => {
//     res.json({ message: 'GET a single workout' });
// });

// // POST a new workouts
// router.post('/', async (req, res) => {
//     //req.body;
//     const { title, reps, load } = req.body;

//     // res.json({ message: 'POST a new workout' });
//     try {
//         const workout = await Workout.create({ title, reps, load });
//         res.status(200).json(workout);
//     } catch (error) {
//         res.status(400).json({ message: error.message });
//     }
    
// });

// // DELETE a workout
// router.delete('/:id', (req, res) => {
//     res.json({ message: 'DELETE a workout' });
// });

// // UPDATE a workout
// router.patch('/:id', (req, res) => {
//     res.json({ message: 'UPDATE a workout' });
// });

// GET all workouts
router.get('/', getWorkouts);

// GET a single workouts
router.get('/:id', getWorkout);

// POST a new workouts
router.post('/', createWorkout);

// DELETE a workout
router.delete('/:id', deleteWorkout);

// UPDATE a workout
router.patch('/:id', updateWorkout);

module.exports = router;