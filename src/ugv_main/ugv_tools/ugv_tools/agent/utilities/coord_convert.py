import math

def convert_coordinates_to_angles(curr_x, curr_y, new_x, new_y):
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

    dx = curr_x - new_x
    dy = new_y - curr_y

    # Update the radians based on dx and dy and the distance from the image center
    dx = curr_x - new_x
    dy = curr_y - new_y
    dx_rad = -float(dx) * math.pi / 18
    dy_rad = float(dy) * math.pi / 18
    return dx_rad, dy_rad
