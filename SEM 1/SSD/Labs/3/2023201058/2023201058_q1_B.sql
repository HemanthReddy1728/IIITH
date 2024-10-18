-- SELECT * FROM SSDLab.menu;
ALTER TABLE SSDLab.menu ADD food_type VARCHAR(255); 
update SSDLab.menu set food_type = 'non-veg' where dish_name like '%Chicken%';
update SSDLab.menu set food_type = 'veg' where dish_name like '%Paneer%';
SELECT * FROM SSDLab.menu;