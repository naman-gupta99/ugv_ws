import time

from .audit_toolset import audit_state_instance


def execute_greedy_audit(metrics, print_agent_metrics):
    print("=== Starting Greedy Audit Bot Mission ===")
    metrics["start_time"] = time.time()

    def _move_to_x(target_x):
        while audit_state_instance.current_coordinates["x"] < target_x:
            metrics["tools_transitions"] += 1
            audit_state_instance.move_right()
        while audit_state_instance.current_coordinates["x"] > target_x:
            metrics["tools_transitions"] += 1
            audit_state_instance.move_left()

    def _move_to_y(target_y):
        while audit_state_instance.current_coordinates["y"] < target_y:
            metrics["tools_transitions"] += 1
            audit_state_instance.move_ahead()
        while audit_state_instance.current_coordinates["y"] > target_y:
            metrics["tools_transitions"] += 1
            audit_state_instance.move_back()

    try:
        target = audit_state_instance.target_area
        x_min = target["x_min"]
        x_max = target["x_max"]
        y_min = target["y_min"]
        y_max = target["y_max"]

        _move_to_x(x_min)
        _move_to_y(y_min)

        for x in range(x_min, x_max + 1):
            if (x - x_min) % 2 == 0:
                _move_to_y(y_max)
            else:
                _move_to_y(y_min)

            if x < x_max:
                _move_to_x(x + 1)

        if audit_state_instance.is_mission_complete():
            metrics["missions_completed"] += 1
        else:
            print("Greedy baseline finished path but mission is not complete.")
    except Exception as e:
        print(f"Error running greedy audit bot: {e}")
    finally:
        metrics["end_time"] = time.time()
        metrics["duration_sec"] = (
            round(metrics["end_time"] - metrics["start_time"], 3)
            if metrics["start_time"]
            else None
        )
        audit_state_instance.plot_path()
        audit_state_instance.print_metrics()
        print_agent_metrics()
