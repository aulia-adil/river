from flask import Flask, request, jsonify
from confluent_kafka import Consumer, KafkaException
from river.datasets import synth
from river.tree import HoeffdingTreeClassifier
import pickle
import threading
import time
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Kafka configuration
kafka_conf = {
    'bootstrap.servers': 'broker:9092',
    'group.id': f"{time.time()}",
    'auto.offset.reset': 'latest' 
}

# Global inference tree
inference_tree = None
tree_lock = threading.Lock()

def initialize_tree():
    """Initialize the inference tree with some initial training data"""
    global inference_tree
    
    logger.info("Initializing inference tree...")
    inference_tree = HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba',
    )
    
    # Initial training
    dataset = synth.Agrawal(classification_function=0, seed=42)
    for i, (x, y) in enumerate(dataset.take(10)):
        inference_tree.learn_one(x, y)
    
    logger.info("Inference tree initialized with 10 samples")

def consume_kafka_events():
    """Background thread to consume Kafka messages and update the tree"""
    global inference_tree
    
    consumer = Consumer(kafka_conf)
    consumer.subscribe(['my_confluent_topic'])
    
    logger.info("Kafka consumer started, listening for events...")
    
    try:
        while True:
            msg = consumer.poll(timeout=0.1)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    logger.info(f"End of partition reached {msg.topic()}/{msg.partition()}")
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                continue
            
            # Deserialize the message
            try:
                data = pickle.loads(msg.value())
                timestamp = data.get('timestamp', time.time())
                update_type = data.get('update_type', 'unknown')
                
                logger.info(f"Received event: {update_type} at {timestamp}")
                
                # Apply the update to the tree (thread-safe)
                with tree_lock:
                    try:
                        if update_type == 'split':
                            inference_tree.apply_split_event(data)
                            logger.info(f"  ✅ Applied Split Node ID: {data['split_node']['node_id']}")
                            logger.info(f"  New Leaves: {[leaf['node_id'] for leaf in data['new_leaves']]}")
                        else:  # leaf_update or complete_node
                            inference_tree.apply_distributed_update(data)
                            node_id = data.get('node_id', 'unknown')
                            total_weight = data.get('data', {}).get('leaf_stats', {}).get('total_weight', 'N/A')
                            logger.info(f"  ✅ Updated Leaf ID: {node_id}, Total Weight: {total_weight}")
                    except Exception as update_error:
                        logger.warning(f"  ⚠️  Could not apply update: {update_error}")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        logger.info("Kafka consumer interrupted")
    finally:
        consumer.close()
        logger.info("Kafka consumer closed")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tree_initialized': inference_tree is not None
    })

@app.route('/inference', methods=['POST'])
def inference():
    """
    Inference endpoint that receives Agrawal dataset features
    Expected input: JSON object with Agrawal features
    Example: {
        "salary": 50000,
        "commission": 0.1,
        "age": 35,
        "elevel": 2,
        "car": 1,
        "zipcode": 5,
        "hvalue": 200000,
        "hyears": 10,
        "loan": 0
    }
    """
    global inference_tree
    
    if inference_tree is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make prediction (thread-safe)
        with tree_lock:
            prediction = inference_tree.predict_one(data)
            
            # Also get prediction probabilities if available
            try:
                proba = inference_tree.predict_proba_one(data)
            except:
                proba = None
        
        response = {
            'prediction': prediction,
            'input': data
        }
        
        if proba is not None:
            response['probabilities'] = proba
        
        logger.info(f"Inference made: {prediction}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({'error': str(e)}), 500

'''
when hit /test, then 
trigger this
results = []
dataset = synth.Agrawal(classification_function=0, seed=44)
for i, (x, y) in enumerate(dataset.take(100)):
    results.append(update_tree.predict_proba_one(x))
'''
@app.route('/test', methods=['GET'])
def test_inference():
    """Test endpoint to run inference on 100 samples from Agrawal dataset"""
    global inference_tree
    
    if inference_tree is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        results = []
        dataset = synth.Agrawal(classification_function=0, seed=44)
        for i, (x, y) in enumerate(dataset.take(100)):
            results.append(inference_tree.predict_proba_one(x))
        
        # save the result in json_data_lab/test_results.json
        with open('json_data_lab/test_results.json', 'w') as f:
            import json
            json.dump(results, f)
        
        return "oke"
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tree-stats', methods=['GET'])
def tree_stats():
    """Get statistics about the current tree"""
    global inference_tree
    
    if inference_tree is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        with tree_lock:
            stats = {
                'n_nodes': inference_tree.n_nodes,
                'n_branches': inference_tree.n_branches,
                'n_leaves': inference_tree.n_leaves,
                'height': inference_tree.height,
                'model_summary': str(inference_tree)[:500]  # Truncated summary
            }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting tree stats: {e}")
        return jsonify({'error': str(e)}), 500

def start_kafka_consumer():
    """Start the Kafka consumer in a background thread"""
    consumer_thread = threading.Thread(target=consume_kafka_events, daemon=True)
    consumer_thread.start()
    logger.info("Kafka consumer thread started")

if __name__ == '__main__':
    # Initialize the tree
    initialize_tree()
    
    # Start Kafka consumer in background
    start_kafka_consumer()
    
    # Start Flask app
    logger.info("Starting Flask app on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
