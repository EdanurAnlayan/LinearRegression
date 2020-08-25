# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:35:10 2020

@author: User
"""
#Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('studentpor.csv',';') #csv uzantılı datasetin okunması 
dataset=dataset.drop('guardian',axis=1)#işe yaramayan bir kolonun datasetten çıkarılması

from sklearn.preprocessing import OneHotEncoder #katagorik olan özellikler, makinenin anlaması için numeric'e çevrilir.
hoten=OneHotEncoder(categorical_features='all') #bunu da one hot encoder ile sağlarız.

#Aşağıdaki kodlarda datasette katagorik olan özellikler hot encoder ile numeric değerlere çevrildi.
cinsiyet=dataset.iloc[:,1:2].values
cinsiyet=hoten.fit_transform(cinsiyet).toarray()
cinsiyet=pd.DataFrame(cinsiyet[:,1:2],range(649),['cinsiyet'])

#Yalnızca 2 seçenek olduğu için tek bir kolon halinde gösterdim.
adres=dataset.iloc[:,3:4].values
adres=hoten.fit_transform(adres).toarray()
adres=pd.DataFrame(adres[:,0:1],range(649),['adres'])

mjob=dataset.iloc[:,8:9].values
mjob=hoten.fit_transform(mjob).toarray()
mjob=pd.DataFrame(mjob,range(649),['at_home_mjob','healt_mjob','other_mjob','services_mjob','teacher_mjob'])

fjob=dataset.iloc[:,9:10].values
fjob=hoten.fit_transform(fjob).toarray()
fjob=pd.DataFrame(fjob,range(649),['athome_fjob','other_fjob','services_fjob','health_fjob','teacher_fjob'])

okul=dataset.iloc[:,0:1].values
okul=hoten.fit_transform(okul).toarray()
okul=pd.DataFrame(okul[:,0:1],range(649),['okul']) 

famsize=dataset.iloc[:,4:5].values
famsize=hoten.fit_transform(famsize).toarray()
famsize=pd.DataFrame(famsize[:,0:1],range(649),['famsize'])

pstatus=dataset.iloc[:,5:6].values
pstatus=hoten.fit_transform(pstatus).toarray()
pstatus=pd.DataFrame(pstatus[:,0:1],range(649),['Pstatus'])

reason=dataset.iloc[:,10:11].values
reason=hoten.fit_transform(reason).toarray()
reason=pd.DataFrame(reason,range(649),['course_reason','home_reason','other_reason','reputition_reason'])

schoolsup=dataset.iloc[:,14:15].values
schoolsup=hoten.fit_transform(schoolsup).toarray()
schoolsup=pd.DataFrame(schoolsup[:,0:1],range(649),['schoolsup'])

famsup=dataset.iloc[:,15:16].values
famsup=hoten.fit_transform(famsup).toarray()
famsup=pd.DataFrame(famsup[:,0:1],range(649),['famsup'])

paid=dataset.iloc[:,16:17].values
paid=hoten.fit_transform(paid).toarray()
paid=pd.DataFrame(paid[:,0:1],range(649),['paid'])

activities=dataset.iloc[:,17:18].values
activities=hoten.fit_transform(activities).toarray()
activities=pd.DataFrame(activities[:,0:1],range(649),['activities'])

nursery=dataset.iloc[:,18:19].values
nursery=hoten.fit_transform(nursery).toarray()
nursery=pd.DataFrame(nursery[:,0:1],range(649),['nursery'])

higher=dataset.iloc[:,19:20].values
higher=hoten.fit_transform(higher).toarray()
higher=pd.DataFrame(higher[:,0:1],range(649),['higher'])

internet=dataset.iloc[:,20:21].values
internet=hoten.fit_transform(internet).toarray()
internet=pd.DataFrame(internet[:,0:1],range(649),['internet'])

romantic=dataset.iloc[:,20:21].values
romantic=hoten.fit_transform(romantic).toarray()
romantic=pd.DataFrame(romantic[:,0:1],range(649),['romantic'])

#Datasetten katagorik olan kolonlar silinir. Ki numeric olanlar ile birleştirebilelim ve karışıklık olmasın.
dataset=dataset.drop('school',axis=1)
dataset=dataset.drop('sex',axis=1)
dataset=dataset.drop('address',axis=1)
dataset=dataset.drop('famsize',axis=1)
dataset=dataset.drop('Pstatus',axis=1)
dataset=dataset.drop('Mjob',axis=1)
dataset=dataset.drop('Fjob',axis=1)
dataset=dataset.drop('reason',axis=1)
dataset=dataset.drop('schoolsup',axis=1)
dataset=dataset.drop('famsup',axis=1)
dataset=dataset.drop('paid',axis=1)
dataset=dataset.drop('activities',axis=1)
dataset=dataset.drop('nursery',axis=1)
dataset=dataset.drop('higher',axis=1)
dataset=dataset.drop('internet',axis=1)
dataset=dataset.drop('romantic',axis=1)

#Oluşturulan numeric özellikler dataset ile birleştirilir.
data_end=pd.concat([okul,cinsiyet,adres,famsize,pstatus,mjob,fjob,reason,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,dataset],axis=1)
output=data_end[['G3']] #G3 hedef output olarak belirlendi.
data_end=data_end.drop('G3',axis=1)#G3 bizim hedef outputumuz olduğu için datasetten çıkardık.Datasette yalnızca inputlar kalmalı.

#Dataset Train ve test olarak 2'ye bölünür.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_end,output,test_size=0.20)#%80i train için %20test için ayrıldı.

#Probleme Lineer Regresyon uygulanacağı için;
from sklearn.linear_model import LinearRegression
lr=LinearRegression() #lr değişkenini Linear Regresyondan bir obje olarak tanımladım
lr.fit(x_train,y_train)#Dataset lr'ye göre eğittim
pred=lr.predict(x_test)#Tahmin değerleri oluşturdum.

#p değerini 0.5olarak belirledim ve bu değere göre fazla olan özellikleri çıkartarak backward elimination işlemleri gerçekleştirdim.
import statsmodels.formula.api as sm
x_l=data_end.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#12 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#22 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#3 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#8 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#18 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,4,5,6,7,9,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#9 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,4,5,6,7,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#33 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,4,5,6,7,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#37 numaralı özellik çıkarılır
x_l=data_end.iloc[:,[0,1,2,4,5,6,7,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,38,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#aynı işlemler en uygun p değerleri kalana kadar devam eder
#...

x_l=data_end.iloc[:,[0,1,10,17,30,32,40,41]].values #p değerleri 0.5 in altında kalan özelliklerle tahmin işlemine devam edilecek.
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#Backward elimination'dan sonra yeni datasetimizi train ve test olarak 2'ye böldük ve işlemleri gerçekleştirdik.
x2_train,x2_test,y2_train,y2_test=train_test_split(x_l,output,test_size=0.20)
lr2=LinearRegression()
lr2.fit(x2_train,y2_train)
pred2=lr2.predict(x2_test)

#R-squared 0.854 olup 0.8'den büyük olduğu için ve p değerleri 0.5'in altında olduğu için optimal tahminlere ulaşılır.































