from example_pb2_grpc import PredictionServiceServicer
import example_pb2_grpc
import example_pb2 as msgs
import grpc
from concurrent import futures

class PredictionService(PredictionServiceServicer):
    def GetPrediction(self,request,context):
        prediction = msgs.Prediction()
        prediction.pred = 106.9
        return prediction

class PredictionService2(PredictionServiceServicer):
    def GetPrediction(self,request,context):
        prediction = msgs.Prediction()
        prediction.pred = 96.9
        return prediction

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    example_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionService2(),server)
    server.add_insecure_port("localhost:5001")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()