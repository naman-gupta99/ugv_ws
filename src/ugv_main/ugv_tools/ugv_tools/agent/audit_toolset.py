from langchain_core.tools import tool
from typing import Any, Dict

from .utilities.coord_convert import convert_coordinates_to_angles

class State:

    def __generate_goal_coordinates(self):
        goals = set()
        for x in range(self.target_area["x_min"], self.target_area["x_max"] + 1):
            for y in range(self.target_area["y_min"], self.target_area["y_max"] + 1):
                goals.add((x, y))
        return goals

    def __init__(self):
        self.current_coordinates = {"x": 0, "y": 0}
        self.target_area = {"x_min": -3, "x_max": 2, "y_min": -2, "y_max": 2}
        self.remaining_coordinates = self.__generate_goal_coordinates()
        self.path = [{"x": 0, "y": 0}]
        self.update_rover_state_func = None

    def move_ahead(self):
        self.current_coordinates["y"] += 1
        self._update_coordinates()

    def move_right(self):
        self.current_coordinates["x"] += 1
        self._update_coordinates()

    def move_back(self):
        self.current_coordinates["y"] -= 1
        self._update_coordinates()

    def move_left(self):
        self.current_coordinates["x"] -= 1
        self._update_coordinates()

    def _update_coordinates(self):
        self.path.append(self.current_coordinates.copy())
        
        dx_rad, dy_rad = convert_coordinates_to_angles(self.path[-2]["x"], self.path[-2]["y"],
                                                       self.current_coordinates["x"], self.current_coordinates["y"])
        if self.update_rover_state_func:
            self.update_rover_state_func(dx_rad, dy_rad)
        else:
            print("[audit_toolset] Warning: update_rover_state_func not set.")

        if (self.current_coordinates["x"], self.current_coordinates["y"]) in self.remaining_coordinates:
            self.remaining_coordinates.remove((self.current_coordinates["x"], self.current_coordinates["y"]))

    def is_mission_complete(self):
        return not self.remaining_coordinates

    def get_state(self):
        return {
            "current_coordinates": self.current_coordinates,
            "target_area": self.target_area,
            "current_path_taken": self.path,
            "mission_complete": self.is_mission_complete(),
        }

    def plot_path(self):
        pass

    def print_metrics(self):
        print("=== Rover Mission Metrics ===")
        try:
            # Basic counts
            total_steps = max(0, len(self.path) - 1)
            start_xy = (self.path[0]["x"], self.path[0]["y"]) if self.path else (None, None)
            end_xy = (self.path[-1]["x"], self.path[-1]["y"]) if self.path else (None, None)

            # Target area
            x_min, x_max = self.target_area["x_min"], self.target_area["x_max"]
            y_min, y_max = self.target_area["y_min"], self.target_area["y_max"]
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            total_goals = width * height

            # Unique positions visited and coverage within target
            visited = {(p["x"], p["y"]) for p in self.path}
            goals = {(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)}
            visited_goals = visited & goals
            visited_goals_count = len(visited_goals)
            remaining_goals_count = total_goals - visited_goals_count
            coverage_pct = (visited_goals_count / total_goals * 100.0) if total_goals > 0 else 0.0

            # Revisits (steps that returned to an already-visited cell)
            unique_positions_count = len(visited)
            revisits = max(0, total_steps - max(0, unique_positions_count - 1))

            # Manhattan entry cost to the rectangle (0 if start inside)
            def dist_to_interval(v: int, a: int, b: int) -> int:
                if a <= v <= b:
                    return 0
                return min(abs(v - a), abs(v - b))

            sx, sy = start_xy
            entry_dx = dist_to_interval(sx, x_min, x_max) if sx is not None else 0
            entry_dy = dist_to_interval(sy, y_min, y_max) if sy is not None else 0
            entry_cost = entry_dx + entry_dy

            # Lower-bound minimal steps to cover all goals starting from start:
            # - Need to reach the rectangle (entry_cost)
            # - Then traverse N cells with a Hamiltonian path: at least (N - 1) moves
            minimal_required_steps_lb = (total_goals - 1) + entry_cost if total_goals > 0 else 0

            steps_remaining_lb = max(0, minimal_required_steps_lb - total_steps)

            # Print summary
            print(f"Start: {start_xy} | End: {end_xy}")
            print(f"Target Area: x=[{x_min},{x_max}] y=[{y_min},{y_max}]  (W={width}, H={height}, Goals={total_goals})")
            print(f"Steps taken: {total_steps}")
            print(f"Unique positions visited: {unique_positions_count}  (Revisits: {revisits})")
            print(f"Goals visited: {visited_goals_count}/{total_goals}  ({coverage_pct:.2f}%)")
            print(f"Mission complete: {self.is_mission_complete()}  | Remaining goals: {remaining_goals_count}")
            print("-- Optimality Lower-Bound (Manhattan-based) --")
            print(f"Entry cost to reach target rectangle: {entry_cost}")
            print(f"Minimum steps required (lower bound): {minimal_required_steps_lb}")
            print(f"Lower-bound remaining steps to finish: {steps_remaining_lb}")
        except Exception as e:
            print(f"[print_metrics] Error computing metrics: {e}")
        

audit_state_instance = State()

@tool
def move_ahead() -> Dict[str, Any]:
    """
    Moves the rover one step forward (in the positive y-direction).

    This function updates the rover's current coordinates by incrementing the 'y' value by 1.
    It then returns the complete current state of the rover's environment.

    Returns:
        Dict[str, Any]: A dictionary representing the current state, including current coordinates,
                        goal coordinates, visited and unvisited goals, and mission completion status.
    """
    audit_state_instance.move_ahead()
    return audit_state_instance.get_state()


@tool
def move_back() -> Dict[str, Any]:
    """
    Moves the rover one step backward (in the negative y-direction).

    This function updates the rover's current coordinates by decrementing the 'y' value by 1.
    It then returns the complete current state of the rover's environment.

    Returns:
        Dict[str, Any]: A dictionary representing the current state, including current coordinates,
                        goal coordinates, visited and unvisited goals, and mission completion status.
    """
    audit_state_instance.move_back()
    return audit_state_instance.get_state()


@tool
def move_left() -> Dict[str, Any]:
    """
    Moves the rover one step left (in the negative x-direction).

    This function updates the rover's current coordinates by decrementing the 'x' value by 1.
    It then returns the complete current state of the rover's environment.

    Returns:
        Dict[str, Any]: A dictionary representing the current state, including current coordinates,
                        goal coordinates, visited and unvisited goals, and mission completion status.
    """
    audit_state_instance.move_left()
    return audit_state_instance.get_state()


@tool
def move_right() -> Dict[str, Any]:
    """
    Moves the rover one step right (in the positive x-direction).

    This function updates the rover's current coordinates by incrementing the 'x' value by 1.
    It then returns the complete current state of the rover's environment.

    Returns:
        Dict[str, Any]: A dictionary representing the current state, including current coordinates,
                        goal coordinates, visited and unvisited goals, and mission completion status.
    """
    audit_state_instance.move_right()
    return audit_state_instance.get_state()
