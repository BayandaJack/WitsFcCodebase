import numpy as np

# Global variable to track forward movement
formation_increment = 0.0  # How much the formation has moved forward

def GenerateBasicFormation(step_size=0.1):

    base_formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-9, 0]),    # Right Defender
        np.array([3, -4]),     # Left Defender
        np.array([3, 4]),      # Forward Left
        np.array([9, 0])       # Forward Right
    ]
    """
    Generates the formation with a small forward x-axis increment each tick.
    step_size: how much to move forward each tick
    
    global formation_increment
    formation_increment += step_size  # increment for this tick

    # Base formation (your current hardcoded setup)
    base_formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-9, 0]),    # Right Defender
        np.array([3, -4]),     # Left Defender
        np.array([3, 4]),      # Forward Left
        np.array([9, 0])       # Forward Right
    ]

    # Move everyone except goalkeeper forward along x-axis
    dynamic_formation = []
    for i, pos in enumerate(base_formation):
        if i == 0:
            # Keep goalkeeper fixed
            dynamic_formation.append(pos.copy())
        else:
            new_pos = pos.copy()
            new_pos[0] += formation_increment
            dynamic_formation.append(new_pos)

    return dynamic_formation
    """
    return base_formation
