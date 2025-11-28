import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS

# Load data
df = pd.read_stata('ps6/JEC.dta')
df['l_quantity'] = np.log(df['quantity'])
df['l_price'] = np.log(df['price'])

# Define variables
dep = df['l_quantity']
endog = df['l_price']
exog_data = df[[f'seas{i}' for i in range(1, 13)]]
# Note: we assume the user has seas1..seas12 columns. 
# If the library requires a constant and we give 12 dummies, it might work if they sum to 1. 
# But usually we just pass what we want as exogenous.

# Run IV regression (iii) - Cartel + Ice
iv3 = IV2SLS(dep, exog_data, endog, df[['cartel', 'ice']]).fit(cov_type='robust')
print(iv3.summary)

