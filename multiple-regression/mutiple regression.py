# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:38:38 2022

@author: My
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


data = np.loadtxt('data.txt',dtype=float, delimiter=',') #还是要浮点数，因为后期预测可能存在小数
#如何第一行是列名，则加入skiprows=1
x=data[:,0:2]#两个自变量,第一个为房子的面积，第二个是卧室的个数
y=data[:,2]#房子的大小
m,n=x.shape


def cost_function(x,y,w,b):
    cost=0
    for i in range(m):
        y_i=np.dot(w,x[i])+b #所谓的vector了，简化运算
        cost=cost+(y_i-y[i])**2
    
    return cost/(2*m)


def gradient_descent(x,y,w,b):
    dw=np.zeros(n,)   #记住w是有n个个数，建立一个numpy列表，从而针对每单个数进行更新
    db=0.  
    
    for j in range(m):
        part=np.dot(w,x[j])+b-y[j]
        for k in range(n):
            dw[k]=dw[k]+ part * x[j,k]   #这一步的理解很重要，因为该数据只需要三个参数w1,w2,b, 所以三个偏导数值
            db=db+part
    
    return dw/m , db/m

#其实可以不用管w的结构了，因为后面调用的时候会用到一个numpy结构的数据
def gradient_function(x,y,w_in,b_in,alpha,iters):
    w=w_in
    b=b_in
    cost_history=[]
    for l in range(iters):
        dw,db=gradient_descent(x,y,w,b)
        w=w-alpha*dw  #因为在后面w调用时，就是一个numpy数组,不必构建一个zeros类型的numpy
        b=b-alpha*db
        
        if l<1000:
            cost_history.append(cost_function(x,y,w,b))
    
    return w , b ,cost_history 



#调用
x_initial=x
y_initial=y
alpha=1.0e-9
iters=1000
w_initial=np.zeros(n,)

w_final,b_final,cost_final = gradient_function(x_initial,y_initial,w_initial,0,alpha,iters)

print(f'最终第一个参数为{w_final[0]},第二个参数为{w_final[1]},参数b为{b_final}')
for i in range(m):
    print(f"prediction: {np.dot(x[i], w_final) + b_final:0.2f}, target value: {y[i]}")




plt.plot(cost_final[:1000],label='cost',linewidth=2)
plt.xlabel('iters')
plt.ylabel('cost')
plt.title('iterations of cost')
plt.legend()
plt.show()



print(f'预测房子为2000平方英尺，3个卧室时的价格为{2000*w_final[0]+3*w_final[1]}')
print(f'预测房子为3000平方英尺，4个卧室时的价格为{3000*w_final[0]+4*w_final[1]}')
print(f'预测房子为4000平方英尺，5个卧室时的价格为{4000*w_final[0]+5*w_final[1]}')


#%%
import numpy as np
np.set_printoptions(precision=2) #输出的np都是两位小数
from sklearn.linear_model import  SGDRegressor  #梯度下降最小二乘法
from sklearn.preprocessing import StandardScaler   #这个库中的一类standardScaler
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')

data=np.loadtxt('data.txt',dtype=float, delimiter=',')
x=data[:,0:2]
y=data[:,2]


#标准化
scaler=StandardScaler()
x_norm=scaler.fit_transform(x)  
#有三种：fit()----作用是求均值方差最大最小，训练集固有的属性,
#transform()----在fit基础上进行标准化，降维，归一化PAC,standardScaler
#fit_transform()-----包含了transform的作用

#拟合模型函数
sgdr=SGDRegressor(max_iter=1000)
sgdr.fit(x_norm,y)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm=sgdr.intercept_
w_norm=sgdr.coef_
print(f'b:{b_norm},w:{w_norm}')



#用sgdr预测
y_pred_sgd = sgdr.predict(x_norm)
#或者用w,b预测
y_pred = np.dot(x_norm, w_norm) + b_norm 

x_feature=['size of house','number of bedroom']
fig,ax=plt.subplots(1,2,figsize=(12,6))
for i in range(len(ax)):
    ax[i].scatter(x[:,i],y,label='train')
    ax[i].set_xlabel(x_feature[i])
    ax[i].scatter(x[:,i],y_pred,label='predict')
ax[0].set_ylabel('price')
ax[0].legend()
fig.suptitle('housing prediction',size=30)
plt.show()


#%%
#还有一种用LinearRegression的做法，不用去迭代
import numpy as np
np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression #单纯最小二乘
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')


data=np.loadtxt('data.txt',dtype=float, delimiter=',')
x_train=data[:,0:2]
y_train=data[:,2]
x_feature=['the size of house','the number of bedroom']

#拟合函数，求w,b
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)

b=linear_model.intercept_
w=linear_model.coef_

#预测
print(f'the prediction is {linear_model.predict(x_train)[:4]}')#这个直接用机器学习函数预测
print(f'the prediction is {(w @ x_train + b)[:4]}')#用w,b来预测

x_house=np.array([1200,3]).reshape(-1,2)
predict=linear_model.predict(x_house)[0]
print(f'房子为1200平方英尺，卧室有三个，房子的价格为{predict}')












