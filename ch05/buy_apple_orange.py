#!/usr/bin/env python
# coding: utf-8
from layer_native import *


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(apple_orange_price, tax)

print("FORWARD---")
print("apple_price = {}".format(apple_price))
print("orange_price = {}".format(orange_price))
print("apple_orange_price = {}".format(apple_orange_price))
print("price = {}".format(price))


dprice = 1
dapple_orange_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dapple_orange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print("BACKWARD---")
print("dapple, dapple_num = {}, {}".format(dapple, dapple_num))
print("dorange, dorange_num = {}, {}".format(dorange, dorange_num))
print("dapple_price, dorange_price = {}, {}".format(dapple_price, dorange_price))
print("dapple_orange_price, dtax = {}, {}".format(dapple_orange_price, dtax))
