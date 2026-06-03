"""
Utility Functions
"""

import math

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def normalize_value(value, min_val, max_val):
    """Normalize value between 0 and 1"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def format_time(seconds):
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def smooth_value(current, target, factor=0.3):
    """Smooth transition between values"""
    return current * (1 - factor) + target * factor

if __name__ == "__main__":
    print("="*50)
    print("Testing utils.py")
    print("="*50)
    
    dist = calculate_distance(0, 0, 3, 4)
    print(f"Distance (0,0) to (3,4): {dist}")
    
    norm = normalize_value(50, 0, 100)
    print(f"Normalize 50 between 0-100: {norm}")
    
    clamped = clamp(150, 0, 100)
    print(f"Clamp 150 between 0-100: {clamped}")
    
    time_str = format_time(125)
    print(f"Format 125 seconds: {time_str}")
    
    smooth = smooth_value(0, 1, 0.3)
    print(f"Smooth 0->1: {smooth:.2f}")
    
    print("✅ utils.py working!")