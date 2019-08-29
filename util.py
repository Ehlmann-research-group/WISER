

def closest_value(values, desired):
    '''
    Given a list of numeric values and a desired value, returns the value in
    the list that is closest to the desired value.

    If the list is empty, the function will return None.

    If multiple values are "closest," the first one will be returned.

    Note that the type of the desired value and the type of the closest value
    may not match, even if they are the same value.  For example, calling
    closest_value([1.0, 2.0, 3.0], 2) will return 2.0, not 2.
    '''

    min_dist = None
    closest = None
    i_closest = None

    for (i, current) in enumerate(values):
        current_dist = abs(current - desired)
        if closest is None or current_dist < min_dist:
            i_closest = i
            closest = current
            min_dist = current_dist

    return (i_closest, closest)
