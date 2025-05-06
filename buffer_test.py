from enum import Enum
from collections import deque, Counter

# Define your Enum
class MyEnum(Enum):
    A = 1
    B = 2
    C = 3

# Create a circular buffer of size 16
buffer = deque(maxlen=16)

# Function to add to the buffer
def add_to_buffer(value: MyEnum):
    buffer.append(value)

# Function to get the most frequent Enum in the buffer
def most_frequent_enum():
    if not buffer:
        return None
    count = Counter(buffer)
    most_common = count.most_common(1)
    
    print(most_common)
    print(most_common[0][1])
    return most_common[0][0] if most_common else None

# Example usage:
add_to_buffer(MyEnum.A)
add_to_buffer(MyEnum.B)
add_to_buffer(MyEnum.A)
add_to_buffer(MyEnum.C)
add_to_buffer(MyEnum.A)

print("Most frequent enum:", most_frequent_enum())
