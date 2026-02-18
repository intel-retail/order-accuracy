"""
RabbitMQ Message Queue Module for Order Processing

Design decisions:
- Transient queue: Messages lost on restart (intentional for sample app)
- Auto-delete: Queue removed when consumers disconnect  
- Manual ack: Only acknowledge after successful VLM processing
- Retry with backoff: Requeue failed messages with delay
- No dead letter: Failed messages eventually discarded after max retries
"""
import os
import json
import time
import logging
import threading
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, asdict

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

logger = logging.getLogger(__name__)

# Configuration from environment
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', '5672'))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS', 'guest')
USE_RABBITMQ = os.environ.get('USE_RABBITMQ', 'true').lower() == 'true'

# Queue names
ORDER_QUEUE = 'order_processing'
EXCHANGE_NAME = ''  # Default exchange (direct routing)

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0  # seconds


@dataclass
class OrderMessage:
    """Message structure for order processing requests"""
    order_id: str
    station_id: str
    retry_count: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'OrderMessage':
        return cls(**json.loads(data))


class RabbitMQConnection:
    """Thread-safe RabbitMQ connection manager"""
    
    def __init__(self):
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """Establish connection to RabbitMQ"""
        with self._lock:
            if self._connection and self._connection.is_open:
                return True
            
            try:
                credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
                parameters = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )
                
                logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")
                self._connection = pika.BlockingConnection(parameters)
                self._channel = self._connection.channel()
                
                # Declare transient, auto-delete queue
                # - durable=False: Queue won't survive broker restart
                # - auto_delete=True: Queue deleted when last consumer disconnects
                self._channel.queue_declare(
                    queue=ORDER_QUEUE,
                    durable=False,
                    auto_delete=True,
                    arguments={
                        'x-message-ttl': 300000,  # 5 minutes TTL for messages
                    }
                )
                
                logger.info(f"Connected to RabbitMQ, queue '{ORDER_QUEUE}' ready")
                return True
                
            except AMQPConnectionError as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
                self._connection = None
                self._channel = None
                return False
    
    def close(self):
        """Close connection"""
        with self._lock:
            if self._connection and self._connection.is_open:
                self._connection.close()
            self._connection = None
            self._channel = None
    
    @property
    def channel(self) -> Optional[pika.channel.Channel]:
        """Get channel, reconnecting if necessary"""
        if not self._channel or self._channel.is_closed:
            self.connect()
        return self._channel
    
    @property 
    def is_connected(self) -> bool:
        return self._connection is not None and self._connection.is_open


class OrderProducer:
    """Producer for publishing order processing requests"""
    
    def __init__(self):
        self._conn = RabbitMQConnection()
        
    def publish(self, order_id: str, station_id: str) -> bool:
        """
        Publish order for processing.
        Returns True if published successfully.
        """
        if not USE_RABBITMQ:
            logger.debug("RabbitMQ disabled, skipping publish")
            return False
            
        message = OrderMessage(order_id=order_id, station_id=station_id)
        
        for attempt in range(3):
            try:
                if not self._conn.connect():
                    time.sleep(1)
                    continue
                    
                self._conn.channel.basic_publish(
                    exchange=EXCHANGE_NAME,
                    routing_key=ORDER_QUEUE,
                    body=message.to_json(),
                    properties=pika.BasicProperties(
                        delivery_mode=1,  # Transient (non-persistent)
                        content_type='application/json',
                    )
                )
                
                logger.info(f"[QUEUE] Published order: order_id={order_id}, station_id={station_id}")
                return True
                
            except (AMQPConnectionError, AMQPChannelError) as e:
                logger.warning(f"[QUEUE] Publish attempt {attempt+1} failed: {e}")
                self._conn.close()
                time.sleep(0.5)
        
        logger.error(f"[QUEUE] Failed to publish order after 3 attempts: order_id={order_id}")
        return False
    
    def close(self):
        """Close producer connection"""
        self._conn.close()


class OrderConsumer:
    """
    Consumer for processing orders from queue.
    
    Features:
    - Manual acknowledgment after successful processing
    - Automatic retry with exponential backoff on failure
    - Graceful shutdown support
    """
    
    def __init__(self, process_callback: Callable[[str, str], bool]):
        """
        Args:
            process_callback: Function(order_id, station_id) -> bool
                             Returns True on success, False on failure
        """
        self._conn = RabbitMQConnection()
        self._process_callback = process_callback
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start consuming messages in background thread"""
        if not USE_RABBITMQ:
            logger.info("RabbitMQ disabled, consumer not started")
            return
            
        if self._running:
            logger.warning("Consumer already running")
            return
            
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            name="OrderConsumer",
            daemon=True
        )
        self._consumer_thread.start()
        logger.info("[QUEUE] Order consumer started")
    
    def stop(self):
        """Stop consuming messages"""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5)
        self._conn.close()
        logger.info("[QUEUE] Order consumer stopped")
    
    def _consume_loop(self):
        """Main consumption loop with reconnection support"""
        while self._running:
            try:
                if not self._conn.connect():
                    logger.warning("[QUEUE] Connection failed, retrying in 5s...")
                    time.sleep(5)
                    continue
                
                # Set prefetch to 1 for fair dispatch
                self._conn.channel.basic_qos(prefetch_count=1)
                
                # Start consuming with manual ack
                self._conn.channel.basic_consume(
                    queue=ORDER_QUEUE,
                    on_message_callback=self._on_message,
                    auto_ack=False
                )
                
                logger.info(f"[QUEUE] Waiting for messages on queue '{ORDER_QUEUE}'...")
                
                while self._running and self._conn.is_connected:
                    self._conn._connection.process_data_events(time_limit=1)
                    
            except (AMQPConnectionError, AMQPChannelError) as e:
                logger.warning(f"[QUEUE] Connection lost: {e}, reconnecting...")
                self._conn.close()
                time.sleep(2)
            except Exception as e:
                logger.error(f"[QUEUE] Unexpected error: {e}", exc_info=True)
                time.sleep(5)
    
    def _on_message(self, channel, method, properties, body):
        """Handle incoming message"""
        try:
            message = OrderMessage.from_json(body.decode('utf-8'))
            logger.info(f"[QUEUE] Received order: order_id={message.order_id}, "
                       f"station_id={message.station_id}, retry={message.retry_count}")
            
            # Process the order
            success = self._process_callback(message.order_id, message.station_id)
            
            if success:
                # Acknowledge successful processing
                channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(f"[QUEUE] Order processed successfully: order_id={message.order_id}")
            else:
                # Retry logic
                self._handle_failure(channel, method, message)
                
        except json.JSONDecodeError as e:
            logger.error(f"[QUEUE] Invalid message format: {e}")
            channel.basic_ack(delivery_tag=method.delivery_tag)  # Discard malformed
        except Exception as e:
            logger.error(f"[QUEUE] Processing error: {e}", exc_info=True)
            # Requeue on unexpected errors
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def _handle_failure(self, channel, method, message: OrderMessage):
        """Handle failed processing with retry logic"""
        message.retry_count += 1
        
        if message.retry_count >= MAX_RETRIES:
            logger.error(f"[QUEUE] Max retries ({MAX_RETRIES}) exceeded for order_id={message.order_id}, discarding")
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        # Calculate backoff delay
        delay = min(INITIAL_RETRY_DELAY * (2 ** (message.retry_count - 1)), MAX_RETRY_DELAY)
        logger.warning(f"[QUEUE] Order failed, retry {message.retry_count}/{MAX_RETRIES} "
                      f"in {delay}s: order_id={message.order_id}")
        
        # Acknowledge original message
        channel.basic_ack(delivery_tag=method.delivery_tag)
        
        # Wait and republish with updated retry count
        time.sleep(delay)
        
        try:
            channel.basic_publish(
                exchange=EXCHANGE_NAME,
                routing_key=ORDER_QUEUE,
                body=message.to_json(),
                properties=pika.BasicProperties(
                    delivery_mode=1,
                    content_type='application/json',
                )
            )
            logger.info(f"[QUEUE] Requeued order for retry: order_id={message.order_id}")
        except Exception as e:
            logger.error(f"[QUEUE] Failed to requeue order: {e}")


# Singleton instances for easy access
_producer: Optional[OrderProducer] = None


def get_producer() -> OrderProducer:
    """Get singleton producer instance"""
    global _producer
    if _producer is None:
        _producer = OrderProducer()
    return _producer


def publish_order(order_id: str, station_id: str) -> bool:
    """Convenience function to publish an order"""
    return get_producer().publish(order_id, station_id)
