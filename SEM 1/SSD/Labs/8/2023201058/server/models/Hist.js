const mongoose = require('mongoose');

const HistSchema = new mongoose.Schema({
  userId: String, // User identifier (e.g., email)
  question: String,
  answer: String,
  role: String, // User role (e.g., 'admin' or 'visitor')
}, {
    collection: 'interactionHistory' // Set the collection name explicitly
  });

const HistoryModel = mongoose.model('interactionHistory',HistSchema);
module.exports = HistoryModel;
