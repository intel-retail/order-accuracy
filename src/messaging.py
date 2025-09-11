"""RabbitMQ/MQTT messaging functionality for metadata collection."""

import json
import time
from typing import List, Dict, Any, Tuple

import pika

from config import (
    RABBITMQ_HOST,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_TOPIC,
    RABBITMQ_EXCHANGE,
    RABBITMQ_ROUTING_KEY,
    logger
)


def test_rabbitmq_connection() -> Tuple[bool, str]:
    """Test the RabbitMQ connection and topic binding."""
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
        params = pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        
        # Test if amq.topic exchange exists
        try:
            channel.exchange_declare(exchange=RABBITMQ_EXCHANGE, exchange_type='topic', passive=True)
            logger.info(f"Exchange {RABBITMQ_EXCHANGE} exists and is accessible")
        except Exception as e:
            logger.error(f"Exchange {RABBITMQ_EXCHANGE} is not accessible: {e}")
            return False, f"Exchange {RABBITMQ_EXCHANGE} not accessible: {e}"
        
        # Create a test queue and binding
        test_queue = f"test_queue_{int(time.time())}"
        channel.queue_declare(queue=test_queue, auto_delete=True)
        channel.queue_bind(exchange=RABBITMQ_EXCHANGE, queue=test_queue, routing_key=RABBITMQ_TOPIC)
        
        logger.info(f"Successfully bound test queue to {RABBITMQ_EXCHANGE} with topic {RABBITMQ_TOPIC}")
        
        # Clean up
        channel.queue_delete(queue=test_queue)
        connection.close()
        
        return True, "RabbitMQ MQTT connection test successful"
        
    except Exception as e:
        logger.exception(f"RabbitMQ connection test failed: {e}")
        return False, f"RabbitMQ connection test failed: {e}"


def fetch_rabbitmq_metadata(limit_frames: int = 3, timeout: int = 30) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Fetch metadata messages from RabbitMQ MQTT topic.
    Uses a callback-based consumer to collect messages in real-time.
    """
    frames_data = []
    
    def message_callback(ch, method, properties, body):
        try:
            payload = json.loads(body.decode('utf-8'))
            logger.info(f"Raw message received: {payload}")
            
            # Parse the new MQTT message structure
            evam_identifier = payload.get('evamIdentifier', 'unknown')
            chunk_id = payload.get('chunkId', 0)
            frames = payload.get('frames', [])
            
            # Only process the first chunk (chunk_id = 1)
            if chunk_id != 1:
                logger.info(f"Skipping chunk {chunk_id} - only processing first chunk (chunk_id=1)")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            logger.info(f"Processing chunk {chunk_id} with {len(frames)} frames from EVAM {evam_identifier}")
            
            # Process each frame in the frames array
            for frame in frames:
                frame_id = frame.get('frameId', 'unknown')
                chunk_frame = frame.get('chunkFrame', 0)
                image_uri = frame.get('imageUri', '')
                metadata = frame.get('metadata', {})
                
                # Extract object information from metadata
                objects = metadata.get('objects', [])
                object_count = len(objects)
                detected_objects = []
                
                # Parse detected objects
                for obj in objects:
                    detection = obj.get('detection', {})
                    label = detection.get('label', 'unknown')
                    confidence = detection.get('confidence', 0.0)
                    bounding_box = detection.get('bounding_box', {})
                    roi_type = obj.get('roi_type', label)
                    region_id = obj.get('region_id', 0)
                    
                    detected_objects.append({
                        'label': label,
                        'roi_type': roi_type,
                        'confidence': confidence,
                        'region_id': region_id,
                        'bounding_box': bounding_box,
                        'position': {
                            'x': obj.get('x', 0),
                            'y': obj.get('y', 0),
                            'w': obj.get('w', 0),
                            'h': obj.get('h', 0)
                        }
                    })
                
                # Extract additional metadata
                frame_timestamp = metadata.get('frame_timestamp', 0.0)
                img_format = metadata.get('img_format', 'BGR')
                resolution = metadata.get('resolution', {})
                
                # Generate frame name for better identification
                frame_name = f"chunk_{chunk_id}_frame_{frame_id}"
                
                frame_data = {
                    'frame_number': frame_id,
                    'object_count': object_count,
                    'frame_id': frame_id,
                    'chunk': chunk_id,
                    'chunk_frame': chunk_frame,
                    'frame_name': frame_name,
                    'image_uri': image_uri,
                    'frame_timestamp': frame_timestamp,
                    'img_format': img_format,
                    'resolution': resolution,
                    'detected_objects': detected_objects,
                    'evam_identifier': evam_identifier,
                    'raw_payload': payload  # Keep original for debugging
                }
                
                frames_data.append(frame_data)
                logger.info(f"Processed {frame_name} with {object_count} objects at timestamp {frame_timestamp}s")
                
                # Log object details if available
                if detected_objects and len(detected_objects) > 0:
                    object_labels = [obj['label'] for obj in detected_objects[:3]]
                    logger.debug(f"Objects in {frame_name}: {object_labels}")
                    if len(detected_objects) > 3:
                        logger.debug(f"... and {len(detected_objects) - 3} more objects")
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(f"Raw message body: {body}")
            
            # Try to extract at least basic frame info even if parsing fails
            try:
                payload = json.loads(body.decode('utf-8'))
                evam_id = payload.get('evamIdentifier', 'error_evam')
                chunk_id = payload.get('chunkId', 0)
                fallback_data = {
                    'frame_number': None,
                    'object_count': 0,
                    'frame_id': 'error_frame',
                    'chunk': chunk_id,
                    'frame_name': f"error_chunk_{chunk_id}",
                    'detected_objects': [],
                    'evam_identifier': evam_id,
                    'raw_payload': payload,
                    'parse_error': str(e)
                }
                frames_data.append(fallback_data)
                logger.warning(f"Added fallback frame data for failed parse: chunk {chunk_id}")
            except Exception as fallback_error:
                logger.error(f"Complete parsing failure, skipping message: {fallback_error}")
            
            # Still acknowledge to avoid redelivery
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST, 
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        
        # Create a named queue that can persist between connections
        queue_name = f"video_metadata_consumer_{int(time.time())}"
        channel.queue_declare(queue=queue_name, durable=False, auto_delete=True, exclusive=False)
        
        # Bind the queue to the amq.topic exchange with the specified topic
        logger.info(f"Binding queue {queue_name} to exchange {RABBITMQ_EXCHANGE} with routing key {RABBITMQ_ROUTING_KEY}")
        channel.queue_bind(exchange=RABBITMQ_EXCHANGE, queue=queue_name, routing_key=RABBITMQ_ROUTING_KEY)
        
        # Also try binding with multiple patterns to catch messages
        routing_patterns = [
            RABBITMQ_TOPIC,  # Original format: topic/video_stream
            RABBITMQ_ROUTING_KEY,  # Dot format: topic.video_stream
            f"{RABBITMQ_ROUTING_KEY}.*",  # topic.video_stream.*
            f"*.{RABBITMQ_ROUTING_KEY.split('.')[-1]}",  # *.video_stream
            "topic.*",  # topic.*
            "*.video_stream",  # *.video_stream  
            "#"  # catch-all (use with caution)
        ]
        
        for pattern in routing_patterns:
            try:
                channel.queue_bind(exchange=RABBITMQ_EXCHANGE, queue=queue_name, routing_key=pattern)
                logger.debug(f"Bound queue to pattern: {pattern}")
            except Exception as e:
                logger.debug(f"Failed to bind to pattern {pattern}: {e}")
        
        # Set up the consumer
        channel.basic_qos(prefetch_count=10)
        channel.basic_consume(queue=queue_name, on_message_callback=message_callback, auto_ack=False)
        
        logger.info(f"Starting to consume messages for {timeout} seconds...")
        start_time = time.time()
        
        # Consume messages with timeout
        while time.time() - start_time < timeout and len(frames_data) < limit_frames * 4:  # Collect more frames for better selection
            try:
                connection.process_data_events(time_limit=1)
                # Stop early if we have enough frames with good object counts
                if len(frames_data) >= limit_frames * 2:
                    # Check if we have good quality frames (with decent object counts)
                    frames_with_objects = [f for f in frames_data if f.get('object_count', 0) > 0]
                    if len(frames_with_objects) >= limit_frames:
                        logger.info(f"Collected enough frames with objects ({len(frames_with_objects)}), stopping early")
                        break
            except Exception as e:
                logger.error(f"Error during message consumption: {e}")
                break
        
        # Clean up
        try:
            channel.stop_consuming()
            channel.queue_delete(queue=queue_name)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        connection.close()
        
        if not frames_data:
            logger.warning(f"No metadata messages consumed from MQTT topic {RABBITMQ_TOPIC} after {timeout} seconds")
            return False, [], f"No metadata messages consumed from MQTT topic {RABBITMQ_TOPIC}. Check if publisher is active."
        
        # Sort by object count (descending) to get frames with most objects first
        frames_data.sort(key=lambda x: (x.get('object_count') or 0), reverse=True)
        
        # Log details about all frames for analysis
        logger.info(f"Total frames collected: {len(frames_data)}")
        logger.info("Ranking frames by object count (descending):")
        for i, frame in enumerate(frames_data):
            object_count = frame.get('object_count', 0)
            frame_name = frame.get('frame_name', f"chunk_{frame.get('chunk', 0)}_frame_{frame.get('frame_id', 'unknown')}")
            logger.info(f"  #{i+1}: {frame_name} - {object_count} objects")
        
        # Get top frames with most objects for detailed logging
        top_frames_to_log = min(3, len(frames_data))
        logger.info(f"=== TOP {top_frames_to_log} FRAMES WITH MOST OBJECTS ===")
        for i, frame in enumerate(frames_data[:top_frames_to_log], 1):
            frame_name = frame.get('frame_name', f"chunk_{frame.get('chunk', 0)}_frame_{frame.get('frame_id', 'unknown')}")
            object_count = frame.get('object_count', 0)
            detected_objects = frame.get('detected_objects', [])
            frame_timestamp = frame.get('frame_timestamp', 0.0)
            image_uri = frame.get('image_uri', '')
            evam_id = frame.get('evam_identifier', 'unknown')
            
            logger.info(f"#{i}. Frame: {frame_name}")
            logger.info(f"    Object Count: {object_count}")
            logger.info(f"    Frame ID: {frame.get('frame_id', 'unknown')}")
            logger.info(f"    Chunk: {frame.get('chunk', 'unknown')}")
            logger.info(f"    Timestamp: {frame_timestamp}s")
            logger.info(f"    Image URI: {image_uri}")
            logger.info(f"    EVAM ID: {evam_id}")
            
            if detected_objects:
                # Group objects by type for better logging
                object_groups = {}
                for obj in detected_objects:
                    label = obj.get('label', 'unknown')
                    confidence = obj.get('confidence', 0.0)
                    if label in object_groups:
                        object_groups[label].append(confidence)
                    else:
                        object_groups[label] = [confidence]
                
                object_summary = []
                for label, confidences in object_groups.items():
                    count = len(confidences)
                    avg_conf = sum(confidences) / count
                    if count > 1:
                        object_summary.append(f"{count}x {label} (avg: {avg_conf:.2f})")
                    else:
                        object_summary.append(f"{label} ({avg_conf:.2f})")
                
                logger.info(f"    Objects: {', '.join(object_summary[:5])}")
                if len(object_summary) > 5:
                    logger.info(f"    ... and {len(object_summary) - 5} more object types")
        
        # Return top frames based on object count (already sorted)
        selected_frames = frames_data[:limit_frames]
        logger.info(f"Returning top {len(selected_frames)} frames (out of {len(frames_data)} total) with highest object counts for further analysis")
        return True, selected_frames, ""
        
    except Exception as e:
        logger.exception(f"RabbitMQ MQTT error: {e}")
        return False, [], f"RabbitMQ MQTT error: {e}"


def get_top_frames_summary(frames_data: List[Dict[str, Any]], top_n: int = 3) -> str:
    """
    Generate a summary of the top N frames with the most objects.
    Returns a formatted string with frame details.
    """
    if not frames_data:
        return "No frames available for analysis."
    
    # Sort frames by object count (descending)
    sorted_frames = sorted(frames_data, key=lambda x: (x.get('object_count') or 0), reverse=True)
    top_frames = sorted_frames[:top_n]
    
    summary_lines = [f"Top {len(top_frames)} frames with most objects:"]
    
    for i, frame in enumerate(top_frames, 1):
        frame_name = frame.get('frame_name', f"chunk_{frame.get('chunk', 0)}_frame_{frame.get('frame_id', 'unknown')}")
        object_count = frame.get('object_count', 0)
        detected_objects = frame.get('detected_objects', [])
        frame_timestamp = frame.get('frame_timestamp', 0.0)
        image_uri = frame.get('image_uri', '')
        
        summary_lines.append(f"{i}. {frame_name}: {object_count} objects at {frame_timestamp}s")
        if image_uri:
            summary_lines.append(f"   Image: {image_uri}")
        
        if detected_objects:
            # Group objects by label for better summary
            object_counts = {}
            for obj in detected_objects:
                label = obj.get('label', obj.get('roi_type', 'unknown'))
                confidence = obj.get('confidence', 0.0)
                if label in object_counts:
                    object_counts[label]['count'] += 1
                    object_counts[label]['avg_confidence'] = (object_counts[label]['avg_confidence'] + confidence) / 2
                else:
                    object_counts[label] = {'count': 1, 'avg_confidence': confidence}
            
            # Show object summary
            object_summary = []
            for label, info in object_counts.items():
                count = info['count']
                avg_conf = info['avg_confidence']
                if count > 1:
                    object_summary.append(f"{count}x {label} ({avg_conf:.2f})")
                else:
                    object_summary.append(f"{label} ({avg_conf:.2f})")
            
            summary_lines.append(f"   Objects: {', '.join(object_summary[:5])}")
            if len(object_summary) > 5:
                summary_lines.append(f"   ... and {len(object_summary) - 5} more object types")
    
    return "\n".join(summary_lines)
