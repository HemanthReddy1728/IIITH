CREATE TABLE `users` (
  `id` int NOT NULL,
  `name` varchar(80) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `shoes` json DEFAULT NULL,
  PRIMARY KEY (`id`)
)