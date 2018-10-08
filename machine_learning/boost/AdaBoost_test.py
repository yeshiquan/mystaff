#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def f(a,x):
    return np.exp(-a*x)

def print_t(title,v):
    print title
    print v
    print ""

def G1(x):
    return np.where(x<2.5, 1, -1)

def G2(x):
    return np.where(x<8.5, 1, -1)

def G3(x):
    return np.where(x<5.5, -1, 1)

def G(a1, a2, a3, x):
    return a1*G1(x) + a2*G2(x) + a3*G3(x)

def sign(x):
    return np.where(x<0, -1, 1)

X = np.array([0,1,2,3,4,5,6,7,8,9])
Y = np.array([1,1,1,-1,-1,-1,1,1,1,-1])

YGx_title = "YGx: 判断分类是否正确，分类正确取值1，分类错误取值-1"

D1 = np.full((1,10),0.1)

print("迭代1：根据2.5分类")
print("====================")
Gx = G1(X)
# x为7,8,9时分类错误
e1 = D1[0][7] + D1[0][8] + D1[0][9]
a1 = 1.0/2.0*np.log((1-e1)/e1)
print_t("a1", a1)

YGx = Y*Gx
fa =  f(a1,YGx)
sum_fa = np.sum(D1*fa)
D2 = D1 * fa / sum_fa
print_t(YGx_title, YGx)
title = "D2: 被分错的样本7 8 9的权值变大，其它被分对的样本的权值变小"
print_t(title, D2)

print("迭代2：根据8.5分类")
print("====================")
Gx = np.array([1,1,1,1,1,1,1,1,1,-1])
# x为3,4,5时分类错误
e2 = D2[0][3]*1 + D2[0][4]*1 + D2[0][5]*1
print_t("e2", e2)
a2 = 1.0/2.0*np.log((1-e2)/e2)
print_t("a2", a2)

YGx = Y*Gx
print_t(YGx_title, YGx)
fa =  f(a2,YGx)
sum_fa = np.sum(D2*fa)
D3 = D2 * fa / sum_fa
title = "D3: 被分错的样本3 4 5的权值变大，其它被分对的样本的权值变小"
print_t(title, D3)

print("迭代3：根据5.5分类")
print("====================")
Gx = np.array([-1,-1,-1,-1,-1,-1,1,1,1,1])
# x为0,1,2,9时分类错误
e3 = D3[0][0]*1 + D3[0][1]*1 + D3[0][2]*1 + D3[0][9]*1
print_t("e3", e3)
a3 = 1.0/2.0*np.log((1-e3)/e3)
print_t("a3", a3)

YGx = Y*Gx
print_t(YGx_title, YGx)
fa =  f(a3,YGx)
sum_fa = np.sum(D3*fa)
D4 = D3 * fa / sum_fa
title = "D4: 被分错的样本0 1 2 9的权值变大，其它被分对的样本的权值变小"
print_t(title, D4)

print("验证模型")
print("====================")
fx =  G(a1,a2,a3,X)
print_t("fx", fx)
Gx = sign(fx)
YGx = Y*Gx
print_t(YGx_title, YGx)
