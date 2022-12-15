# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:16:02 2022

@author: My
"""

#问题：根据公司的4个指标预测企业是否破产，0为破产，1为未破产
import numpy as np
import matplotlib.pyplot as plt
import math
data=np.loadtxt('中小企业破产判别.txt',skiprows=1,encoding='GBK')
x = data[:,0:4] #指标有：总负债率，收益性指标，短期支付能力，生产效率指标
y = data[:,4]
y[y==1]=0 #企业破产
y[y==2]=1 #企业没有破产，经营状况良好
m,n=x.shape
def logistic_function(x,w,b):
    z=np.dot(x,w)+b
    y=1/(1+np.exp(-z))
    
    return y

def cost_function(x,y,w,b):
    cost=0
    for i in range(m):
        y_=logistic_function(x[i], w, b)
        loss=-y[i]*math.log(y_)-(1-y[i])*math.log(1-y_)
        cost=cost+loss
        
    cost=cost/m
    return cost

def gradient_function(x,y,w,b):
    #求代价函数的偏导数
    dw=np.zeros(n,)
    db=0
    for i in range(m):
       y_=logistic_function(x[i], w, b)
       dw_=y_-y[i]
       
       for j in range(n):
           dw[j] = dw_ * x[i,j]/m + dw[j]
    
       db=db+dw_

    return dw,db

def gradient(x,y,w,b,alpha,iters):
    J_history=[]
    for i in range(iters):
        dw,db=gradient_function(x,y,w,b)
        for j in range(n):
            w[j]=w[j]-alpha*dw[j]
        b=b-alpha*db
        
        if i < 1000:
            J_history.append(cost_function(x,y,w,b))
        
    return w,b,J_history

#调用
w=np.zeros(n,)
b=0
alpha=1.0e-1
iters=1000
w_final,b_final,J_history=gradient(x, y, w, b, alpha, iters)
        
print(f'w为{w_final},b为{b_final}')

#画出cost的迭代值

fig,ax=plt.subplots(1,1)
ax.plot(J_history,label='cost')
ax.set_xlabel('iters')
ax.set_ylabel('cost')
ax.set_title('the process of iters')
ax.legend()
ax.annotate("convergency", xy= [200,0.4], xycoords='data',    
                 xytext=[250,0.45], ha="left", va="center",
                   arrowprops={'arrowstyle': '->'})
#判断是否在200出收敛
difference=J_history[199]-J_history[200]
if difference<0.001:
    print('差值小于0.001，函数已经收敛')
        

#预测值和实际值相比较，精度测算
y_=logistic_function(x, w_final , b_final)
for h in range(y_.shape[0]):
    if y_[h]>0.5:
        y_[h]=1
    else:
        y_[h]=0
        
q=x.shape[0]
a=0
for g in range(y_.shape[0]):
    if y_[g]==y[g]:
        a+=1
p=a/q#即预测精度就为这么多
print(f'正确预测的精度为{p}')
        
#发现那些变量偏差时，分类明显
x_name=['x1','x2','x3','x4']
positive=y==1
negative=y==0
positive_=y_==1
negative_=y_==0
fig,ax = plt.subplots(2,3,figsize=(12,6))
for i in range(1):
    ax[1,i+2].scatter(x[positive,2],x[positive,i+3],color='blue',s=16,label='y=normal')
    ax[1,i+2].scatter(x[negative,2],x[negative,i+3],marker='x',color='red',s=16,label='y=bankruptcy')
    ax[1,i+2].set_xlabel(x_name[i+2])
    ax[1,i+2].set_ylabel(x_name[i+3])
    ax[1,i+2].legend()
    for j in range(2):
        ax[1,j].scatter(x[positive,1],x[positive,j+2],color='blue',s=16,label='y=normal')
        ax[1,j].scatter(x[negative,1],x[negative,j+2],marker='x',color='red',s=16,label='y=bankruptcy')
        ax[1,j].set_xlabel(x_name[1])
        ax[1,j].set_ylabel(x_name[j+2])
        for k in range(3):
            ax[0,k].scatter(x[positive,0],x[positive,k+1],color='blue',s=16,label='y=nromal')
            ax[0,k].scatter(x[negative,0],x[negative,k+1],marker='x',color='red',s=16,label='y=bankruptcy')
            ax[0,k].set_xlabel(x_name[0])
            ax[0,k].set_ylabel(x_name[k+1])
fig.suptitle('the relationship among the different variables',fontsize=25)
#图片看出，x1,x3也就是总负债率和短期支付能力对破产的分类影响较大；总负债率适当越大，短期支付能力越大，企业不容易破产.
#同时通过w的大小也可以看出，w1和w3值较大，对分类贡献较大

#regularization正则化，消除overfit
def adjust_cost_function(x,y,w,b,lamda):
    cost=cost_function(x, y, w, b)
    sum=0
    for i in range(x.shape[1]):
        sum+=w[i]**2
    adjust_cost=cost+(lamda/(2*x.shape[0]))*sum
    
    return adjust_cost


def adjust_gradient_function(x,y,w,b,lamda):
    #求代价函数的偏导数
    dw,db=gradient_function(x, y, w, b)
    dw_=np.zeros(n,)
    for i in range(x.shape[1]):
        dw_[i]=dw[i]+(lamda/m)*w[i]

    return dw_,db


def adjust_gradient(x,y,w,b,alpha,iters):
    K_history=[]
    for i in range(iters):
        dw_,db=adjust_gradient_function(x,y,w,b,lamda)
        for j in range(n):
            w[j]=w[j]-alpha*dw_[j]
        b=b-alpha*db
        
        if i < 1000:
            K_history.append(adjust_cost_function(x,y,w,b,lamda))
        
    return w,b,K_history

w=np.zeros(n,)
b=0
alpha=1.0e-1
iters=1000
lamda=0.1   #取值很重要,在w=w-alpha*dw中，被化简为w*(1-alpha*lambda/m),所以如果lambda很大,w就很小,导致underfit;如果lambda很小，w就很大,不满足cost最小，而overfit
adjust_w_final,b_final,K_history=adjust_gradient(x, y, w, b, alpha, iters)
        
print(f'正则化后的w为{adjust_w_final},b为{b_final}')

#比较正则化后的cost
fig,ax=plt.subplots(1,1)
ax.plot(J_history,label='original',color='blue')
ax.plot(K_history,label='regularization',color='red')
ax.set_xlabel('iters')
ax.set_ylabel('cost')
ax.legend()
ax.set_title('comparison')
ax.annotate('convergency speed',xy=(120,0.43),xycoords='data',xytext=(200,0.48),ha='left',va='center',arrowprops={'arrowstyle':'->'})
#说明正则化后收敛加快，因为cost减小小于0.001先到达


#正则化后预测y
adjust_y_=logistic_function(x, adjust_w_final , b_final)
#和上面步骤相同
for h in range(adjust_y_.shape[0]):
    if adjust_y_[h]>0.5:
        adjust_y_[h]=1
    else:
        adjust_y_[h]=0
        
q=x.shape[0]
a=0
for g in range(adjust_y_.shape[0]):
    if adjust_y_[g]==y[g]:
        a+=1
p=a/q#即预测精度就为这么多
print(f'正则化后的正确预测的精度为{p}')

#预测剩余未知破产的公司
data1=np.loadtxt('预测的数据集.txt')
predict_x=data1[:,0:4]
results=logistic_function(predict_x, adjust_w_final, b_final)
print(results)

#%%
import numpy as np
from sklearn.linear_model import LogisticRegression

data_group=np.loadtxt('中小企业破产判别.txt',skiprows=1)
X=data_group[:,0:4]
Y=data_group[:,4]
Y[Y==1]=0 #企业破产
Y[Y==2]=1 #企业没有破产，经营状况良好
lr_model = LogisticRegression() 
lr_model.fit(X, Y)

y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X, y))  #预测值和实际值相符合的概率







