import sys
import pandas as pd

input_path = sys.argv[1]
print(input_path)
data = pd.read_csv(input_path)
print(data)