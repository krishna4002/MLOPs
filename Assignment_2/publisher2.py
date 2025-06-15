<<<<<<< HEAD
#!/usr/bin/env python

from random import choice
from confluent_kafka import Producer

if __name__ == '__main__':

    config = {
        # User-specific properties that you must set
        'bootstrap.servers': 'localhost:9092',

        # Fixed properties
        'acks': 'all'
    }

    # Create Producer instance
    producer1=Producer(config)
    producer2 = Producer(config)

    # Optional per-message delivery callback (triggered by poll() or flush())
    # when a message has been successfully delivered or permanently
    # failed delivery (after retries).
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))

    # Produce data by selecting random values from these lists.
    topic =[ "dummy","channel"]
    user_ids = '42'
    products = 'Pranab'
    producer1.produce(topic[0], products, user_ids, callback=delivery_callback)
    user_ids_2 = '60'
    products_2 = 'Debraj'
    producer2.produce(topic[1], products_2, user_ids_2, callback=delivery_callback)

    # Block until the messages are sent.
    producer1.poll(10000)
    producer1.flush()
    producer2.poll(10000)
=======
#!/usr/bin/env python

from random import choice
from confluent_kafka import Producer

if __name__ == '__main__':

    config = {
        # User-specific properties that you must set
        'bootstrap.servers': 'localhost:9092',

        # Fixed properties
        'acks': 'all'
    }

    # Create Producer instance
    producer1=Producer(config)
    producer2 = Producer(config)

    # Optional per-message delivery callback (triggered by poll() or flush())
    # when a message has been successfully delivered or permanently
    # failed delivery (after retries).
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))

    # Produce data by selecting random values from these lists.
    topic =[ "dummy","channel"]
    user_ids = '42'
    products = 'Pranab'
    producer1.produce(topic[0], products, user_ids, callback=delivery_callback)
    user_ids_2 = '60'
    products_2 = 'Debraj'
    producer2.produce(topic[1], products_2, user_ids_2, callback=delivery_callback)

    # Block until the messages are sent.
    producer1.poll(10000)
    producer1.flush()
    producer2.poll(10000)
>>>>>>> 03585fa (Adding Assignment_2 work)
    producer2.flush()