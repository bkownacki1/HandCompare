import csv
import numpy as np
from pandas import read_csv

arr = np.arange(120000)
arr = np.append(arr, "index")
print(arr)

df = read_csv('img_pixels.csv')
df.columns = arr
df.to_csv('img_pixels.csv')

# f = open("img_pixels.csv", "w")
# writer = csv.DictWriter(
#      f, fieldnames=arr)
# writer.writeheader()
# f.close()