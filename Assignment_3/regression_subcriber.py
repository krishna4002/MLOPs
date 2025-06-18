from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'iris_predictions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='regression-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("[Regression Subscriber] Listening...")
for msg in consumer:
    data = msg.value
    if data['model_type'].lower() == 'regression':
        print(f"[Regression] Received: {data}")