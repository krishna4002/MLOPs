from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def publish_prediction(model_type, data, prediction):
    message = {
        'model_type': model_type,
        'input': data,
        'prediction': prediction
    }
    producer.send('iris_predictions', message)
    producer.flush()
    print(f"[Kafka] Published to topic: {message}")