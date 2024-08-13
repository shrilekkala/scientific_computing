def is_a_number(var):
    """
    Checks if a variable is of type integer or float
    Returns true or false
    """
    if isinstance(var, bool):
        # Edge case, as otherwise a boolean counts as an integer (0 or 1)
        return False
    elif isinstance(var, int) or isinstance(var, float):
        return True
    else:
        return False
