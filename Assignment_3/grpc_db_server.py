# grpc_server/grpc_db_server.py
from concurrent import futures
import grpc
import iris_pb2
import iris_pb2_grpc
import os
import csv

BASE_DIR = os.path.join(os.getcwd(), 'database')
os.makedirs(BASE_DIR, exist_ok=True)

CLS_CSV_PATH = os.path.join(BASE_DIR, 'classification_predictions.csv')
REG_CSV_PATH = os.path.join(BASE_DIR, 'regression_predictions.csv')


class IrisServiceServicer(iris_pb2_grpc.IrisServiceServicer):
    def SavePrediction(self, request, context):
        try:
            # Choose file based on model_type
            if request.model_type.lower() == "classification":
                file_path = CLS_CSV_PATH
            elif request.model_type.lower() == "regression":
                file_path = REG_CSV_PATH
            else:
                return iris_pb2.PredictionResponse(message="Unknown model type")

            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(file_path)

            # Write to CSV
            with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['model_type', 'input_data', 'prediction'])
                writer.writerow([request.model_type, request.input_data, request.prediction])

            return iris_pb2.PredictionResponse(message="Saved to CSV")

        except Exception as e:
            return iris_pb2.PredictionResponse(message=f"Failed to save: {str(e)}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    iris_pb2_grpc.add_IrisServiceServicer_to_server(IrisServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC Server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
