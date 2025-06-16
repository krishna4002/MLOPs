import demo_pb2
import sys

flower = demo_pb2.Flower()
flower.sepal_length = 2.3
flower.sepal_width = 4.5
flower.petal_length = 6.9
flower.petal_width = 0.5

flower_dict={}
flower_dict["sepal_width"] ="Baishik"
flower_dict["sepal_length"] =2.3
flower_dict["petal_length"] ="Poddar"
flower_dict["petal_width"] =7.4

print(sys.getsizeof(flower_dict))
print(sys.getsizeof(flower))
print("\n",flower.SerializeToString())
print("\n--------------------------------\n")
dflower= demo_pb2.Flower()
dflower.ParseFromString(b'\r33\x13@\x15\xcd\xcc\xdc@\x1d\x00\x00\x90@%\x00\x00\x00?')
print(dflower)