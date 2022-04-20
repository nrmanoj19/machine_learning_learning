import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math
import matplotlib.pyplot as plt


df = pd.read_csv('hiring.csv')
df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)
mean_score = math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(mean_score)
print(mean_score)
print(df.head(10))
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])
print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))
plt.scatter(df['experience'], df['salary($)'], color='red', marker='*')
plt.scatter(df['test_score(out of 10)'], df['salary($)'], color='blue', marker='*')
plt.scatter(df['interview_score(out of 10)'], df['salary($)'], color='green', marker='*')
plt.xlabel('candidate_profile: Red-experience Blue-test_score Green-interview_score')
plt.ylabel('salary($)')
plt.show()
plt.savefig('salary_prediction.png')

