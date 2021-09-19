import pandas as pan
import pickle
from sklearn.linear_model import LinearRegression

data = pan.read_excel('BP_simple_data.xlsx')

x1 = data.iloc[:,1:]
x2 = data.iloc[:,0:1]

reg = LinearRegression()
reg.fit(x1,x2)

prediction = reg.predict(x1)
pickle.dump(reg, open('model.pkl','wb'))
