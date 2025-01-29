import pygame as pg
import numpy as np
import sys
from math import ceil

pg.font.init()

def get_x_values(y_values_len):
	return np.linspace(0, surfrect.w, y_values_len)

def moving_average(data, window_size=50):
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    moving_avg = np.concatenate((np.zeros(window_size - 1), moving_avg))
    
    return moving_avg

def read_file(file_path):
	with open(file_path, 'r') as file:
		data = file.read().strip().split(',')
		data = data[:-1]
		data = np.array(list(map(float, data)))
		
	return data

def draw_gridx(surface, font, values, y_start, y_stop, x_grid_spacing):
    for i, x in enumerate(values):
        text = font.render(str(i * x_grid_spacing), 1, 'black')
        text = pg.transform.flip(text, 0, 1)
        text = pg.transform.rotate(text, 90)
        text_y = y_stop - text.get_rect().h
        start = (x, y_start)
        stop = (x, y_stop)
        pg.draw.line(surface, 'yellow', start, stop)
        surface.blit(text, (x, text_y))

def draw_gridy(surface, font, values, x_start, x_stop, y_origin, y_scale):
    for y in values:
        text = font.render(str(y), 1, 'black')
        text = pg.transform.flip(text, 0, 1)
        y_ = (y + y_origin) * y_scale
        start = (x_start, y_)
        stop = (x_stop, y_)
        pg.draw.line(surface, 'yellow', start, stop)
        surface.blit(text, start)

def draw_grid(surface, x_values, y_values, start, stop, y_origin, y_scale, x_grid_spacing):
    font = pg.font.SysFont(None, 16)
    x_start, y_start = start
    x_stop, y_stop = stop
    draw_gridx(surface, font, x_values, y_start, y_stop, x_grid_spacing)
    draw_gridy(surface, font, y_values, x_start, x_stop, y_origin, y_scale)


saves_folder = ''
x_grid_spacing = 25
y_grid_spacing = 5
if len(sys.argv) > 1:
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        arg_value = sys.argv[i+1]
        if arg == '-x':
            x_grid_spacing = int(arg_value)
        elif arg == '-y':
            y_grid_spacing = int(arg_value)
        elif arg == '-s':
            saves_folder = arg_value
        else:
            raise Exception('Invalid command line arguments')
            
    if saves_folder in ['0', '1']:
        saves_folder = ''

#surface = pg.display.set_mode((1300, 660), pg.FULLSCREEN)
surface = pg.display.set_mode((1300, 660))
surfrect = surface.get_rect()

file = f'./saves{saves_folder}/rewards.txt'
image_file = f'./saves{saves_folder}/rewards_plot.png'

y_values = read_file(file)
x_values = get_x_values(len(y_values))
m_avg_values = moving_average(y_values)
min_y = min(y_values)
max_y = max(y_values)
min_y_index = np.argmin(y_values)
max_y_index = np.argmax(y_values)
min_m_avg = min(m_avg_values)
max_m_avg = max(m_avg_values)
min_m_avg_index = np.argmin(m_avg_values)
max_m_avg_index = np.argmax(m_avg_values)
x_grid_values = [x for i, x in enumerate(x_values) if i % x_grid_spacing == 0]
y_grid_values = [i for i in range(min(0, int(min_y // y_grid_spacing * y_grid_spacing)) ,ceil(max_y) + y_grid_spacing, y_grid_spacing)]
y_scale = surfrect.h / (max_y + y_grid_spacing + abs(min_y))
origin = (0, abs(min(0, min_y)))
print(f'Length:\t{len(y_values)} \
       \nFirst:\t{y_values[0]} \
       \nLast:\t{y_values[-1]} \
       \nMin:\t{min_y} Episode:{min_y_index} \
       \nMax:\t{max_y} Episode:{max_y_index} \
       \nMean:\t{round(np.mean(y_values), 2)} \
       \nMoving Avg. Last:\t{round(m_avg_values[-1], 2)} \
       \nMoving Avg. Min:\t{round(min_m_avg, 2)} Episode:{min_m_avg_index} \
       \nMoving Avg. Max:\t{round(max_m_avg, 2)} Episode:{max_m_avg_index}')
bg_color = 'white'
lines_color = 'black'
m_avg_lines_color = 'blue'

points = []
m_avg_points = []

for x, y, y2 in zip(x_values, y_values, m_avg_values):
    y_ = (y + origin[1]) * y_scale
    y2_ = (y2 + origin[1]) * y_scale
    points.append((x, y_))
    m_avg_points.append((x, y2_))

surface.fill(bg_color)
line_start = origin[0], origin[1] * y_scale
line_stop = surfrect.w, origin[1] * y_scale
draw_grid(surface, x_grid_values, y_grid_values, (0, 0), (surfrect.w, surfrect.h), origin[1], y_scale, x_grid_spacing)
pg.draw.line(surface, 'red', line_start, line_stop, width=1)

pg.draw.lines(surface, lines_color, False, points)
pg.draw.lines(surface, m_avg_lines_color, False, m_avg_points)
surface.blit(pg.transform.flip(surface, 0, 1), (0, 0))
pg.display.flip()

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()
        elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
            sys.exit()
        elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
            pg.image.save(surface, image_file)