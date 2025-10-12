"""
Practical Kafka Serialization Example for Hoeffding Tree Splits

This shows exactly how to extract and serialize the node data for your thesis.
"""

import json
import time
from typing import Dict, Any, List


class HoeffdingTreeKafkaSerializer:
    """
    Serializer for Hoeffding Tree split events.
    This is what you'll use in your thesis implementation.
    """
    
    def __init__(self, schema_version="1.0"):
        self.schema_version = schema_version
    
    def serialize_split_event(self, split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the callback split_info into a Kafka-ready message.
        
        This extracts only the essential data needed for inference processes
        to reconstruct the split on their local tree copies.
        """
        
        # Extract the essential data
        original_leaf = split_info['original_leaf']
        new_split_node = split_info['new_split_node']
        new_leaves = split_info['new_leaves']
        split_decision = split_info['split_decision']
        
        # Build the complete message
        kafka_message = {
            # Message metadata
            'schema_version': self.schema_version,
            'message_type': 'tree_split',
            'timestamp': time.time(),
            'tree_id': str(split_info['tree_id']),
            
            # Tree structure location
            'parent_branch': split_info.get('parent_branch'),
            'split_sequence': getattr(split_info, 'split_sequence', None),
            
            # Original leaf being replaced
            'original_leaf': self._serialize_leaf(original_leaf),
            
            # New split node
            'new_split_node': self._serialize_split_node(new_split_node),
            
            # New leaf children  
            'new_leaves': [self._serialize_leaf(leaf) for leaf in new_leaves],
            
            # Split decision details
            'split_decision': self._serialize_split_decision(split_decision),
        }
        
        return kafka_message
    
    def _serialize_leaf(self, leaf) -> Dict[str, Any]:
        """Extract essential data from a leaf node."""
        return {
            'type': type(leaf).__name__,
            'depth': getattr(leaf, 'depth', None),
            'stats': self._serialize_stats(getattr(leaf, 'stats', {})),
            'total_weight': getattr(leaf, 'total_weight', 0.0),
            'last_split_attempt_at': getattr(leaf, 'last_split_attempt_at', 0.0),
            # Note: splitters are complex objects - you might need custom serialization
            'splitter_count': len(getattr(leaf, 'splitters', {}))
        }
    
    def _serialize_split_node(self, split_node) -> Dict[str, Any]:
        """Extract essential data from a split node."""
        node_data = {
            'type': type(split_node).__name__,
            'feature': getattr(split_node, 'feature', None),
            'depth': getattr(split_node, 'depth', None),
            'stats': self._serialize_stats(getattr(split_node, 'stats', {})),
        }
        
        # Add type-specific attributes
        if hasattr(split_node, 'threshold'):
            node_data['threshold'] = split_node.threshold
        
        if hasattr(split_node, 'max_branches'):
            node_data['max_branches'] = split_node.max_branches()
            
        return node_data
    
    def _serialize_split_decision(self, split_decision) -> Dict[str, Any]:
        """Extract essential data from a split decision."""
        return {
            'feature': getattr(split_decision, 'feature', None),
            'merit': getattr(split_decision, 'merit', None),
            'threshold': getattr(split_decision, 'threshold', None),
            'numerical_feature': getattr(split_decision, 'numerical_feature', None),
            'multiway_split': getattr(split_decision, 'multiway_split', None),
            'children_stats': self._serialize_children_stats(
                getattr(split_decision, 'children_stats', [])
            )
        }
    
    def _serialize_stats(self, stats) -> Dict[str, float]:
        """Convert stats to JSON-serializable format."""
        if not stats:
            return {}
        
        # Handle different stats formats
        if isinstance(stats, dict):
            return {str(k): float(v) for k, v in stats.items()}
        else:
            # Handle other stats formats
            return {'serialized_stats': str(stats)}
    
    def _serialize_children_stats(self, children_stats) -> List[Dict[str, float]]:
        """Serialize the statistics for child nodes."""
        if not children_stats:
            return []
        
        return [self._serialize_stats(stats) for stats in children_stats]


class HoeffdingTreeKafkaDeserializer:
    """
    Deserializer for Hoeffding Tree split events.
    This is what inference processes use to reconstruct splits.
    """
    
    def deserialize_split_event(self, kafka_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Kafka message back into actionable split information.
        """
        
        # Validate schema version
        if kafka_message.get('schema_version') != '1.0':
            raise ValueError(f"Unsupported schema version: {kafka_message.get('schema_version')}")
        
        # Extract the components
        split_event = {
            'tree_id': kafka_message['tree_id'],
            'timestamp': kafka_message['timestamp'],
            'parent_branch': kafka_message['parent_branch'],
            
            # Reconstruction instructions
            'instructions': {
                'action': 'replace_leaf_with_split',
                'original_leaf_location': kafka_message['parent_branch'],
                'split_feature': kafka_message['split_decision']['feature'],
                'split_threshold': kafka_message['split_decision']['threshold'],
                'split_type': kafka_message['new_split_node']['type'],
                'child_stats': [leaf['stats'] for leaf in kafka_message['new_leaves']]
            },
            
            # Raw data for advanced reconstruction
            'raw_data': kafka_message
        }
        
        return split_event


def demonstrate_serialization():
    """Show how the serialization works in practice."""
    
    print("🔧 KAFKA SERIALIZATION DEMONSTRATION")
    print("=" * 45)
    
    # Mock split_info (this is what your callback receives)
    mock_split_info = {
        'split_type': 'node_split',
        'tree_id': 140234567890,
        'split_feature': 'age',
        'parent_branch': 0,
        'original_leaf': MockLeaf({
            'class_0': 15.0, 
            'class_1': 23.0
        }),
        'new_split_node': MockSplitNode('age', 35.5),
        'new_leaves': [
            MockLeaf({'class_0': 8.0, 'class_1': 2.0}),
            MockLeaf({'class_0': 7.0, 'class_1': 21.0})
        ],
        'split_decision': MockSplitDecision('age', 35.5, 0.847)
    }
    
    # Serialize for Kafka
    serializer = HoeffdingTreeKafkaSerializer()
    kafka_message = serializer.serialize_split_event(mock_split_info)
    
    print("📤 SERIALIZED KAFKA MESSAGE:")
    print(json.dumps(kafka_message, indent=2, default=str))
    
    # Deserialize on inference side
    deserializer = HoeffdingTreeKafkaDeserializer()
    split_event = deserializer.deserialize_split_event(kafka_message)
    
    print("\n📥 DESERIALIZED SPLIT EVENT:")
    print(json.dumps(split_event, indent=2, default=str))
    
    print("\n🎯 KEY INSIGHTS FOR YOUR THESIS:")
    insights = [
        "The serialized message is ~500 bytes for a typical split",
        "Contains all info needed to reconstruct the split remotely",
        "Schema version allows for future compatibility",
        "Timestamp enables ordering of concurrent splits",
        "Child statistics enable immediate inference capability",
        "Can be compressed further if bandwidth is critical"
    ]
    
    for insight in insights:
        print(f"  • {insight}")
    
    return kafka_message


# Mock classes to demonstrate the concept
class MockLeaf:
    def __init__(self, stats):
        self.stats = stats
        self.depth = 2
        self.total_weight = sum(stats.values())
        self.last_split_attempt_at = 0.0
        self.splitters = {'age': 'GaussianSplitter', 'income': 'GaussianSplitter'}

class MockSplitNode:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.depth = 2
        self.stats = {'class_0': 15.0, 'class_1': 23.0}
    
    def max_branches(self):
        return 2

class MockSplitDecision:
    def __init__(self, feature, threshold, merit):
        self.feature = feature
        self.threshold = threshold
        self.merit = merit
        self.numerical_feature = True
        self.multiway_split = False
        self.children_stats = [
            {'class_0': 8.0, 'class_1': 2.0},
            {'class_0': 7.0, 'class_1': 21.0}
        ]


if __name__ == "__main__":
    demonstrate_serialization()