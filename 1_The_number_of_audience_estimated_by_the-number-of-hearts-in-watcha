# Edwith 강의에서 선형회귀를 설명하면서, 하나의 예시를 계속해서 보여주신다.
#'왓챠 '보고싶어요' 수로 예상한 '옥자' 관객 수'
#옥자는 2017년 여름에 개봉한 영화인데, 몇몇 상영관에서만 개봉을 했고 일반 영화관에서는 개봉 안한걸로 알고 있다.
#빅데이터를 이용한 머신러닝 툴로 옥자 관람객 구할 수 있다는 것 ! 아마 정확도는.. ? 모르겠다.
#아 '옥자가 다른 영화처럼 일반 개봉했을 때 '라는 전제가 들어간다.
#이미 예상 관람객 수가 나와있지만, 다른 영화 데이터를 이용해서 어떤 수가 나올지 궁금해서 정말 기본적인 sklearn 모듈의 linear_model로 구해봤다.

​#The teacher demostarates linear regression in the online lecture "Edwith" and continues to show an example
#The number of audience of movie "Okja" predicted by the number of hearts in Watcha
#Okja is the movie released in the summer of 2017, it was released only in some theaters and I remember that it had not been released in general cinemas.
#It is possible to get the predicted number of audience of Okja if Okja had been released in the genearl cinema by using other movie Data
#But I'm not sure about the accuracy because I'm gonna use very simple and basic linear regression.
#I already have an estimated number of viewers, but I was wondering what number would come from using other movie data, so I got it with the linear_model of the basic sklearn module.


import pandas as pd
import numpy as np
title=["Martion", "Kingsman", "Captin America", "Interstella"]
num_heart=[8759, 10132, 12078, 16430]
actual=[4870000, 6120000, 8660000, 10300000]
data=pd.DataFrame({'title':title, 'num_heart':num_heart, 'actual':actual})

X=data['num_heart'].values.reshape(-1, 1)
y=data['actual'].values


#원래 갖고 있던 데이터로 데이터프레임을 만든다.
#간단한 선형회귀를 시키기 위해 sklearn 을 임포트

#Make a dataframe using sklearn

from sklearn import linear_model
sk_lr=linear_model.LinearRegression()
sk_lr.fit(X,y)


Ok=np.array([12008])
Ok_data=Ok.reshape(-1, 1)

predict_result=sk_lr.predict(Ok_data)
data['predict_actual']=sk_lr.predict(X)

#predict 했음

new_data=pd.DataFrame({'title':'Okza', 'num_heart':Ok, 'predict_actual':predict_result})
data=data.append(new_data, ignore_index=True, sort=False)

#하트를 바탕으로 예상한 관람객 수까지 포함한 데이터 셋
#The data including the number of audience predicted by the number of hearts of each movie in watcha
data

#예상 관람객 수를 기준으로 sorting
#Sorting by estimated number of visitors


data=data.sort_values(by=['predict_actual'])
data


#그래프까지 그려 보았다.
#Make a graph
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
fig=plt.figure()
ax=plt.axes()
ax.plot(data['title'], data['predict_actual'],label='the predicted number of audience')
ax.plot(data['title'], data['actual'], 'o', color='red', label='the actual number of audience')
ax.legend()
plt.title("The Actul number of Audiences \npredicted by the number of hearts in Watcha")
plt.show()
