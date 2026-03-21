import math

X_M_PER_UNIT          =  1.3   # metres per llmptctrl grid unit (x axis)
Y_M_PER_UNIT          =  0.9   # metres per llmptctrl grid unit (y axis)

def convert_coordinates_to_angles(curr_x, curr_y, new_x, new_y, laser_scan, current_angles):
    """
    Convert pixel coordinates to pan-tilt angle differences in radians.
    
    Args:
        curr_x (int): Current x coordinate (pixel).
        curr_y (int): Current y coordinate (pixel).
        new_x (int): New x coordinate (pixel).
        new_y (int): New y coordinate (pixel).
    
    Returns:
        tuple: (dx_rad, dy_rad) angles in radians.
    """

    # Update the radians based on dx and dy and the distance from the image center

    dx, dy = new_x - curr_x, new_y - curr_y

    curr_x_rad, curr_y_rad = current_angles

    len_scan = len(laser_scan.ranges)
    mid = len_scan * 3 // 4
    dist_x = laser_scan.ranges[mid]

    print(f"[coord_convert] len_scan: {len_scan}")
    print(f"[coord_convert] Distance: {laser_scan.ranges[mid+2]} at {mid+2}")
    print(f"[coord_convert] Distance: {laser_scan.ranges[mid+1]} at {mid+1}")
    print(f"[coord_convert] Distance : {dist_x} at {mid}")
    print(f"[coord_convert] Distance: {laser_scan.ranges[mid-1]} at {mid-1}")
    print(f"[coord_convert] Distance: {laser_scan.ranges[mid-2]} at {mid-2}")

    a = len_scan / (2 * math.pi)
    y_idx = curr_y_rad * a + mid
    dist_y = laser_scan.ranges[int(y_idx)]

    rad_x = math.atan(math.tan(curr_x_rad) + (X_M_PER_UNIT*dx)/dist_x)
    rad_y = math.atan(math.tan(curr_y_rad) + (Y_M_PER_UNIT*dy)/dist_y)

    return rad_x, rad_y
