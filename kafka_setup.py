#!/usr/bin/env python3
"""
🔧 KAFKA SETUP HELPER
====================
Checks Kafka connectivity and creates required topics.
"""

import sys
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError, KafkaError
    import time
except ImportError:
    print("❌ kafka-python not installed. Run: pip install kafka-python")
    sys.exit(1)

def check_kafka_connectivity(bootstrap_servers="kafka_service:9092"):
    """Check if Kafka is accessible."""
    print(f"🔗 CHECKING KAFKA CONNECTIVITY")
    print(f"   Server: {bootstrap_servers}")

    try:
        # Try to create a producer to test connectivity
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=5000,
            retries=1
        )
        
        # Get topic metadata to verify connection
        partitions = producer.partitions_for('hoeffding_tree_updates')
        producer.close()
        
        if partitions is not None:
            print(f"✅ Kafka is accessible!")
            print(f"   Topic 'hoeffding_tree_updates' partitions: {partitions}")
            return True
        else:
            print(f"❌ Could not fetch metadata for topic 'hoeffding_tree_updates'")
            return False
        
    except Exception as e:
        print(f"❌ Kafka connection failed: {e}")
        print(f"💡 Make sure Kafka is running:")
        print(f"   Docker: docker-compose up kafka_service")
        print(f"   Check ports: 9092 (PLAINTEXT), 9093 (CONTROLLER)")
        return False

def create_topic(bootstrap_servers="kafka_service:9092", topic_name="hoeffding_tree_updates"):
    """Create Kafka topic if it doesn't exist."""
    print(f"\n📝 CREATING KAFKA TOPIC")
    print(f"   Topic: {topic_name}")
    
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='topic_creator',
            request_timeout_ms=10000
        )
        
        # Create topic configuration
        topic = NewTopic(
            name=topic_name,
            num_partitions=3,
            replication_factor=1
        )
        
        # Create the topic
        admin_client.create_topics([topic])
        print(f"✅ Topic '{topic_name}' created successfully")
        
        admin_client.close()
        return True
        
    except TopicAlreadyExistsError:
        print(f"ℹ️  Topic '{topic_name}' already exists")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create topic: {e}")
        return False

def list_topics(bootstrap_servers="kafka_service:9092"):
    """List all available topics."""
    print(f"\n📋 LISTING KAFKA TOPICS")
    
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='topic_lister',
            request_timeout_ms=5000
        )
        
        topics_metadata = admin_client.describe_topics()
        topics = list(admin_client.list_topics())
        
        print(f"   Available topics: {len(topics)}")
        for topic in sorted(topics):
            print(f"   - {topic}")
        
        admin_client.close()
        return topics
        
    except Exception as e:
        print(f"❌ Failed to list topics: {e}")
        return []

def main():
    """Main setup function."""
    print(f"🚀 KAFKA SETUP FOR DISTRIBUTED HOEFFDING TREE")
    print("=" * 50)
    
    # Get Kafka server from command line or use default
    kafka_servers = sys.argv[1] if len(sys.argv) > 1 else "kafka_service:9092"
    
    # Check connectivity
    if not check_kafka_connectivity(kafka_servers):
        print(f"\n💡 TROUBLESHOOTING:")
        print(f"   1. Check if Kafka container is running:")
        print(f"      docker ps | grep kafka")
        print(f"   2. Check if ports are exposed:")
        print(f"      docker port <kafka_container>")
        print(f"   3. Try different bootstrap server:")
        print(f"      python kafka_setup.py <server:port>")
        return False
    
    # Create topic
    if not create_topic(kafka_servers):
        return False
    
    # List all topics
    list_topics(kafka_servers)
    
    print(f"\n✅ KAFKA SETUP COMPLETE!")
    print(f"   🎯 Ready to run distributed test:")
    print(f"      python kafka_distributed_test.py")
    
    return True

if __name__ == "__main__":
    main()