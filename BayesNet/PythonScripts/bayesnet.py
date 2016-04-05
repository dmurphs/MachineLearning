import pandas as pd
from helpers import k2

dataFrame = pd.read_csv('../Data/forestFireData.csv')

dimensions = dataFrame.columns

network = k2(dataFrame,3,dimensions)
print network
