from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'iris_predictions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='classification-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("[Classification Subscriber] Listening...")
for msg in consumer:
    data = msg.value
    if data['model_type'].lower() == 'classification':
        print(f"[Classification] Received: {data}")