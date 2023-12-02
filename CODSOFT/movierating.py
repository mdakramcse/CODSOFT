import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load the movie dataset
movie_data = pd.read_csv('IMDb Movies India.csv')

# Display the first few rows of the dataset
print(movie_data.head())

# Data cleaning and preprocessing
# Fill missing values or replace with averages
movie_data['Rating'].fillna(movie_data['Rating'].mean(), inplace=True)
movie_data['Year'].fillna(movie_data['Year'].mode()[0], inplace=True)
movie_data['Length'].fillna(movie_data['Length'].mean(), inplace=True)

# Trend Analysis
# Year with the best average rating
avg_rating_per_year = movie_data.groupby('Year')['Rating'].mean().reset_index()
best_year = avg_rating_per_year.loc[avg_rating_per_year['Rating'].idxmax()]

# Relationship between movie length and rating
sns.scatterplot(x='Length', y='Rating', data=movie_data)
plt.title('Movie Length vs. Rating')
plt.show()

# Top 10 movies according to rating per year and overall
top_movies_per_year = movie_data.groupby('Year').apply(lambda x: x.nlargest(10, 'Rating')).reset_index(drop=True)
top_movies_overall = movie_data.nlargest(10, 'Rating')

# Number of popular movies released each year
popular_movies_count = movie_data.groupby('Year')['Votes'].count().reset_index()

# Counting the number of votes for movies that performed better in rating per year and overall
top_votes_per_year = movie_data.groupby('Year').apply(lambda x: x.nlargest(1, 'Rating')).reset_index(drop=True)
top_votes_overall = movie_data.nlargest(1, 'Rating')

# Linear Regression Model
X = movie_data[['Year', 'Length', 'Votes']]
y = movie_data['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print trends and analysis results
print("\nYear with the Best Average Rating:")
print(best_year)

print("\nTop 10 Movies According to Rating per Year:")
print(top_movies_per_year)

print("\nTop 10 Movies Overall:")
print(top_movies_overall)

print("\nNumber of Popular Movies Released Each Year:")
print(popular_movies_count)

print("\nNumber of Votes for Movies with Best Rating per Year:")
print(top_votes_per_year)

print("\nNumber of Votes for Movie with Best Rating Overall:")
print(top_votes_overall)

# Print the coefficients of the linear regression model
print("\nLinear Regression Model Coefficients:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
