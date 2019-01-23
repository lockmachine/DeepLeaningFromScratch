from layer_native import MulLayer
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple = MulLayer()
mul_tax = MulLayer()

# forward
apple_price = mul_apple.forward(apple, apple_num)
price = mul_tax.forward(apple_price, tax)

print("price = {}".format(price))

# backward
dprice = 1
dapple_price, dtax = mul_tax.backward(dprice)
dapple, dapple_num = mul_apple.backward(dapple_price)

print("dapple = {}\ndapple_num = {}\ndapple_price = {}\ndtax = {}".format(dapple, dapple_num, dapple_price, dtax))
