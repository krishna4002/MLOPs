import grpc
import example_pb2 as msgs
import example_pb2_grpc as msgs2

def run():
    print("Getting the Prediction...")
    with grpc.insecure_channel("localhost:5001") as channel:
        stub = msgs2.PredictionServiceStub(channel)
        print(stub)
        flower = msgs.Flower()
        flower.sepal_length = 7.2
        flower.sepal_width = 4.2
        flower.petal_length = 8.2
        flower.petal_width = 1.2
        response = stub.GetPrediction(flower)
    print(response.pred)

if __name__=="__main__":
    run()