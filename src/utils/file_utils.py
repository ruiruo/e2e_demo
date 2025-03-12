from collections import defaultdict


# Define a recursive defaultdict function
def nested_dict():
    return defaultdict(nested_dict)
