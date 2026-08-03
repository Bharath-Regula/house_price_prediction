import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
# load the dataset
df = pd.read_csv("house_p.csv")
x = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']
# create a logistic regression model
model = LinearRegression()
model.fit(x, y)
# save model using pickle
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model trained and saved successfully.")    
