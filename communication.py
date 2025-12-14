"""
Communication system between robots
Includes messaging, information sharing, and coordination
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum


class MessageType(Enum):
    """Message types"""
    RESOURCE_FOUND = "resource_found"
    RESOURCE_DEPLETED = "resource_depleted"
    OBSTACLE_WARNING = "obstacle_warning"
    HELP_REQUEST = "help_request"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"


class Message:
    """Message class for robot communication"""
    
    def __init__(
        self,
        sender_id: int,
        message_type: MessageType,
        content: Dict,
        position: Tuple[int, int, int],
        timestamp: int,
        range_limit: float = 10.0
    ):
        self.sender_id = sender_id
        self.message_type = message_type
        self.content = content
        self.position = position
        self.timestamp = timestamp
        self.range_limit = range_limit
        self.received_by: Set[int] = set()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sender_id': self.sender_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'position': self.position,
            'timestamp': self.timestamp,
            'range_limit': self.range_limit
        }
    
    def is_in_range(self, receiver_pos: Tuple[int, int, int]) -> bool:
        """Check if message is within receiver range"""
        distance = np.linalg.norm(np.array(self.position) - np.array(receiver_pos))
        return distance <= self.range_limit


class CommunicationNetwork:
    """Communication network for robots"""
    
    def __init__(self, max_message_history: int = 1000, message_ttl: int = 50):
        self.messages: List[Message] = []
        self.max_message_history = max_message_history
        self.message_ttl = message_ttl
        self.current_timestamp = 0
        self.message_stats = {
            'sent': 0,
            'received': 0,
            'expired': 0
        }
    
    def broadcast_message(
        self,
        sender_id: int,
        message_type: MessageType,
        content: Dict,
        position: Tuple[int, int, int],
        range_limit: float = 10.0
    ) -> Message:
        """Broadcast message to network"""
        message = Message(
            sender_id=sender_id,
            message_type=message_type,
            content=content,
            position=position,
            timestamp=self.current_timestamp,
            range_limit=range_limit
        )
        
        self.messages.append(message)
        self.message_stats['sent'] += 1
        
        if len(self.messages) > self.max_message_history:
            self.messages.pop(0)
        
        return message
    
    def get_messages_for_agent(
        self,
        agent_id: int,
        agent_position: Tuple[int, int, int],
        message_types: Optional[List[MessageType]] = None
    ) -> List[Message]:
        """Get messages relevant to an agent"""
        relevant_messages = []
        
        for message in self.messages:
            if self.current_timestamp - message.timestamp > self.message_ttl:
                continue
            
            if message_types and message.message_type not in message_types:
                continue
            
            if not message.is_in_range(agent_position):
                continue
            
            if agent_id in message.received_by:
                continue
            
            if message.sender_id == agent_id:
                continue
            
            relevant_messages.append(message)
            message.received_by.add(agent_id)
            self.message_stats['received'] += 1
        
        return relevant_messages
    
    def update_timestamp(self, new_timestamp: int):
        """Update timestamp"""
        self.current_timestamp = new_timestamp
        
        self.messages = [
            msg for msg in self.messages
            if new_timestamp - msg.timestamp <= self.message_ttl
        ]
    
    def get_stats(self) -> Dict:
        """Get communication statistics"""
        return self.message_stats.copy()


class CommunicationProtocol:
    """Communication protocol for robots"""
    
    def __init__(self, agent_id: int, network: CommunicationNetwork):
        self.agent_id = agent_id
        self.network = network
        self.message_queue: List[Message] = []
        self.known_resources: Dict[Tuple[int, int, int], int] = {}
        self.known_obstacles: Set[Tuple[int, int, int]] = set()
        self.cooperation_partners: Set[int] = set()
    
    def send_resource_found(self, position: Tuple[int, int, int], energy_value: float):
        """Send resource found message"""
        message = self.network.broadcast_message(
            sender_id=self.agent_id,
            message_type=MessageType.RESOURCE_FOUND,
            content={
                'position': position,
                'energy_value': energy_value,
                'confidence': 1.0
            },
            position=position,
            range_limit=15.0
        )
        self.known_resources[position] = self.network.current_timestamp
    
    def send_resource_depleted(self, position: Tuple[int, int, int]):
        """Send resource depleted message"""
        self.network.broadcast_message(
            sender_id=self.agent_id,
            message_type=MessageType.RESOURCE_DEPLETED,
            content={'position': position},
            position=position,
            range_limit=12.0
        )
        if position in self.known_resources:
            del self.known_resources[position]
    
    def send_obstacle_warning(self, position: Tuple[int, int, int]):
        """Send obstacle warning"""
        self.network.broadcast_message(
            sender_id=self.agent_id,
            message_type=MessageType.OBSTACLE_WARNING,
            content={'position': position, 'severity': 'high'},
            position=position,
            range_limit=8.0
        )
        self.known_obstacles.add(position)
    
    def send_help_request(self, position: Tuple[int, int, int], reason: str):
        """Send help request"""
        self.network.broadcast_message(
            sender_id=self.agent_id,
            message_type=MessageType.HELP_REQUEST,
            content={'position': position, 'reason': reason, 'urgency': 'medium'},
            position=position,
            range_limit=20.0
        )
    
    def send_coordination(self, target_position: Tuple[int, int, int], action: str):
        """Send coordination message"""
        self.network.broadcast_message(
            sender_id=self.agent_id,
            message_type=MessageType.COORDINATION,
            content={'target': target_position, 'action': action},
            position=target_position,
            range_limit=10.0
        )
    
    def process_messages(self, current_position: Tuple[int, int, int]) -> Dict:
        """Process received messages"""
        messages = self.network.get_messages_for_agent(
            self.agent_id,
            current_position,
            message_types=None
        )
        
        processed_info = {
            'new_resources': [],
            'depleted_resources': [],
            'new_obstacles': [],
            'coordination_requests': [],
            'help_requests': []
        }
        
        for message in messages:
            if message.message_type == MessageType.RESOURCE_FOUND:
                resource_pos = tuple(message.content['position'])
                if resource_pos not in self.known_resources:
                    processed_info['new_resources'].append({
                        'position': resource_pos,
                        'energy_value': message.content.get('energy_value', 25),
                        'sender': message.sender_id
                    })
                    self.known_resources[resource_pos] = message.timestamp
            
            elif message.message_type == MessageType.RESOURCE_DEPLETED:
                resource_pos = tuple(message.content['position'])
                if resource_pos in self.known_resources:
                    processed_info['depleted_resources'].append(resource_pos)
                    del self.known_resources[resource_pos]
            
            elif message.message_type == MessageType.OBSTACLE_WARNING:
                obstacle_pos = tuple(message.content['position'])
                if obstacle_pos not in self.known_obstacles:
                    processed_info['new_obstacles'].append(obstacle_pos)
                    self.known_obstacles.add(obstacle_pos)
            
            elif message.message_type == MessageType.COORDINATION:
                processed_info['coordination_requests'].append({
                    'sender': message.sender_id,
                    'target': tuple(message.content['target']),
                    'action': message.content['action']
                })
                self.cooperation_partners.add(message.sender_id)
            
            elif message.message_type == MessageType.HELP_REQUEST:
                processed_info['help_requests'].append({
                    'sender': message.sender_id,
                    'position': tuple(message.content['position']),
                    'reason': message.content.get('reason', 'unknown')
                })
        
        return processed_info
    
    def get_nearest_known_resource(
        self,
        current_position: Tuple[int, int, int],
        max_age: int = 30
    ) -> Optional[Tuple[int, int, int]]:
        """Find nearest known resource"""
        valid_resources = [
            pos for pos, timestamp in self.known_resources.items()
            if self.network.current_timestamp - timestamp <= max_age
        ]
        
        if not valid_resources:
            return None
        
        distances = [
            (pos, np.linalg.norm(np.array(current_position) - np.array(pos)))
            for pos in valid_resources
        ]
        distances.sort(key=lambda x: x[1])
        
        return distances[0][0] if distances else None
    
    def should_cooperate(self, other_agent_id: int) -> bool:
        """Decide whether to cooperate"""
        return other_agent_id in self.cooperation_partners
    
    def get_communication_features(self, current_position: Tuple[int, int, int]) -> np.ndarray:
        """Extract communication features for observation"""
        processed = self.process_messages(current_position)
        
        num_new_resources = len(processed['new_resources'])
        
        nearest_resource = self.get_nearest_known_resource(current_position)
        if nearest_resource:
            dist_to_resource = np.linalg.norm(
                np.array(current_position) - np.array(nearest_resource)
            )
        else:
            dist_to_resource = 100.0
        
        nearby_obstacles = sum(
            1 for obs in self.known_obstacles
            if np.linalg.norm(np.array(current_position) - np.array(obs)) < 5.0
        )
        
        nearby_partners = len(self.cooperation_partners)
        
        return np.array([
            num_new_resources / 10.0,
            dist_to_resource / 50.0,
            nearby_obstacles / 5.0,
            nearby_partners / 5.0
        ], dtype=np.float32)
