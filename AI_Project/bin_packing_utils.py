"""Bin packing dataset utilities for OR dataset (binpack5-8)."""

import re


def parse_binpack_file(filepath: str) -> dict:
    """
    Parse OR bin packing dataset file.
    
    Format per instance:
        instance_name
        capacity num_items optimal
        item1_size
        item2_size
        ...
        itemN_size
    
    Returns:
        dict: {instance_name: {"capacity": float, "items": tuple, "num_items": int, "optimal": int}}
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    instances = {}
    lines = content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if it's a number (num_instances) or instance name
        if line.isdigit():
            # First line is total number of instances
            i += 1
            continue
        
        # Instance name line
        instance_name = line
        i += 1
        
        if i >= len(lines):
            break
        
        # Header line: capacity num_items optimal
        header_match = re.match(r'([\d.]+)\s+(\d+)\s+(\d+)', lines[i].strip())
        if not header_match:
            i += 1
            continue
        
        capacity = float(header_match.group(1))
        num_items = int(header_match.group(2))
        optimal = int(header_match.group(3))
        i += 1
        
        # Read item sizes
        items = []
        for _ in range(num_items):
            if i < len(lines):
                try:
                    item = float(lines[i].strip())
                    items.append(item)
                except ValueError:
                    pass
                i += 1
        
        instances[instance_name] = {
            "capacity": capacity,
            "items": tuple(items),
            "num_items": num_items,
            "optimal": optimal
        }
    
    return instances


# Load the OR3 dataset (subset for testing)
def get_or3_instances():
    """Return the first 2 instances from binpack8 as OR3 format."""
    all_instances = parse_binpack_file('binpack8.txt')
    # Select first 2 instances
    instance_names = list(all_instances.keys())[:2]
    return {name: all_instances[name] for name in instance_names}
