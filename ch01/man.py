class Man:
    def __init__(self, name):   #コンストラクタ
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Goodbye" + self.name + "!")


m = Man("David")
m.hello()
m.goodbye()

n = Man("Sam")
n.hello()
n.goodbye()
