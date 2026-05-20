import math

from ...lidar_scan_utils import finite_range_at_angle, process_scan_for_rover

X_M_PER_UNIT          =  1.1   # metres per llmptctrl grid unit (x axis)
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
    if laser_scan is None:
        return current_angles

    scan = process_scan_for_rover(laser_scan)
    if scan is None:
        return current_angles
    
    if new_y == 0:
        return 0.0, 0.0

    dx, dy = new_x - curr_x, new_y - curr_y
    curr_x_rad, curr_y_rad = current_angles

    dist_x = finite_range_at_angle(scan, -90.0)
    dist_x_0 = finite_range_at_angle(scan, 0.0)
    dist_x_90 = finite_range_at_angle(scan, 90.0)

    # Guard against division by zero or invalid range metadata.
    dist_x = max(dist_x, 1e-6)

    rad_x = math.atan(math.tan(curr_x_rad) + (X_M_PER_UNIT * dx) / dist_x)
    rad_y = math.atan(math.tan(curr_y_rad) + (Y_M_PER_UNIT * dy) / dist_x)

    return rad_x, rad_y