-- SELECT * FROM SSDLab.menu;
update SSDLab.menu set dish_name = REPLACE(dish_name, '0', '');
update SSDLab.menu set dish_name = TRIM(' ' FROM dish_name);
SELECT * FROM SSDLab.menu;