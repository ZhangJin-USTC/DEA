import os
import pandas as pd
import numpy as np
from abessdag import abessdag

# Import real data
marks = pd.read_csv(os.getcwd()+"\\marks.csv")
marks = marks.drop(marks.columns[0], axis=1)

# ABESS-DAG
result = abessdag(marks)
print(result[1])

#  From    To
#   ALG  VECT
#   ALG   ANL
#   ALG  STAT
#  VECT  MECH
#   ALG  MECH