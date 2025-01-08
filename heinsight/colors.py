from random import randint
from itertools import chain

COLOR_DICT = {
    -10: "No color",
    0: "red",
    10: "red-orange",
    20: "orange",
    30: "yellow",
    40: "yellow-green",
    50: "green",
    60: "green",
    70: "green-aqua",
    80: "aqua",
    90: "cyan",
    100: "cyan-blue",
    110: "blue",
    120: "blue-violet",
    130: "violet",
    140: "violet-red",
    150: "magenta",
    160: "magenta-red",
    170: "red",
}


def register_colors(model_list):
    color_palette = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (0, 255, 255),  # Cyan
        (128, 128, 0),  # Olive
    ]
    name_color_dict = {}

    names = [model.names.values() for model in model_list if model is not None]
    names = set(chain.from_iterable(names))
    if len(names) > len(color_palette):
        color_palette.extend([(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(len(names) - len(color_palette))])
    for index, name in enumerate(names):
        name_color_dict[name] = color_palette[index]
    return name_color_dict
