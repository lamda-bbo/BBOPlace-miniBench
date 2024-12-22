from utils.constant import INF
from utils.debug import *

def comp_res(macro_pos, placedb):
    if len(macro_pos) == 0:
        return INF

    hpwl = 0.0
    for net_name in placedb.net_info:
        max_x = 0.0
        min_x = placedb.canvas_width * 1.1
        max_y = 0.0
        min_y = placedb.canvas_height * 1.1
        for macro in placedb.net_info[net_name]["nodes"]:
            size_x = placedb.node_info[macro]["size_x"]
            size_y = placedb.node_info[macro]["size_y"]
            pin_x = macro_pos[macro][0] + size_x / 2 + placedb.net_info[net_name]["nodes"][macro]["x_offset"]
            pin_y = macro_pos[macro][1] + size_y / 2 + placedb.net_info[net_name]["nodes"][macro]["y_offset"]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)

        hpwl_temp = (max_x - min_x) + (max_y - min_y)
        
        if "weight" in placedb.net_info[net_name]:
            hpwl_temp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_temp
    return hpwl
