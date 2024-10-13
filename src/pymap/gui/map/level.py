"""Level utilities."""


def level_to_info(level: int) -> str:
    """Converts a level to a string with information.

    Args:
        level (int): The level.

    Returns:
        str: The information.
    """
    x, y = level % 4, level // 4

    x_to_collision = {0: 'Passable', 1: 'Obstacle', 2: '??? (2)', 3: '??? (3)'}

    match y:
        case int() if 2 < y < 15:
            return f'Level {hex(y)}, {x_to_collision[x]}'
        case 0:
            x_to_collision = {
                0: 'Connect Levels',
                1: 'Obstacle',
                2: '??? (2)',
                3: '??? (3)',
            }
            return f'{x_to_collision[x]}'
        case 1:
            return f'Water, {x_to_collision[x]}'
        case 15:
            return f'Bridge, {x_to_collision[x]}'
        case _:
            return f'??? (y={y})'
