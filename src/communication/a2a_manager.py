#!/usr/bin/env python3

from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
import json
import uuid
from datetime import datetime
from src.utils.logger import setup_logger

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"

class Message:
    
    def __init__(self, 
                 msg_id: str = None,
                 sender_id: str = None,
                 receiver_id: str = None,
                 msg_type: MessageType = MessageType.REQUEST,
                 content: Dict[str, Any] = None,
                 timestamp: datetime = None,
                 correlation_id: str = None):
        
        self.id = msg_id or str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.type = msg_type
        self.content = content or {}
        self.timestamp = timestamp or datetime.now()
        self.correlation_id = correlation_id or str(uuid.uuid4())  # For tracking related messages
        self.status = "created"  # created, sent, delivered, processed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "status": self.status
        }
    
    def __str__(self) -> str:
        return f"Message(id={self.id}, sender={self.sender_id}, receiver={self.receiver_id}, type={self.type.value})"
    
    def __repr__(self) -> str:
        return self.__str__()

class A2ACommunicationManager:
    
    def __init__(self):
        self.logger = setup_logger("A2ACommunicationManager")
        self.message_queues = {}  # agent_id -> asyncio.Queue
        self.active_connections = {}  # agent_id -> connection_info
        self.message_history = []  # Store recent messages for debugging
        self.max_history_size = 1000
        self.handlers = {}  # agent_id -> handler function
    
    async def register_agent(self, agent_id: str, handler_func=None):
        self.active_connections[agent_id] = {
            "connected_at": datetime.now(),
            "status": "active",
            "last_heartbeat": datetime.now()
        }
        
        # Create a message queue for this agent
        self.message_queues[agent_id] = asyncio.Queue()
        
        if handler_func:
            self.handlers[agent_id] = handler_func
        
        self.logger.info(f"Agent {agent_id} registered with communication manager")
    
    async def unregister_agent(self, agent_id: str):
        if agent_id in self.active_connections:
            del self.active_connections[agent_id]
        
        if agent_id in self.message_queues:
            del self.message_queues[agent_id]
        
        if agent_id in self.handlers:
            del self.handlers[agent_id]
        
        self.logger.info(f"Agent {agent_id} unregistered from communication manager")
    
    async def send_message(self, message: Message) -> bool:
        if not message.receiver_id:
            self.logger.error("Cannot send message without receiver_id")
            return False
        
        if message.receiver_id not in self.active_connections:
            self.logger.error(f"Cannot send message to inactive agent: {message.receiver_id}")
            return False
        
        message.status = "sent"
        self._add_to_history(message)
        
        # Put the message in the receiver's queue
        await self.message_queues[message.receiver_id].put(message)
        message.status = "delivered"
        
        return True
    
    async def broadcast_message(self, message: Message, exclude_sender: bool = True) -> int:
        sent_count = 0
        message.type = MessageType.BROADCAST
        
        for agent_id in list(self.active_connections.keys()):
            if exclude_sender and agent_id == message.sender_id:
                continue
            
            # Create a copy of the message for each recipient
            recipient_message = Message(
                sender_id=message.sender_id,
                receiver_id=agent_id,
                msg_type=message.type,
                content=message.content.copy(),
                correlation_id=message.correlation_id
            )
            
            if await self.send_message(recipient_message):
                sent_count += 1
        
        self.logger.info(f"Broadcast message sent to {sent_count} agents")
        return sent_count
    
    async def get_messages_for_agent(self, agent_id: str, limit: int = 10) -> List[Message]:
        if agent_id not in self.message_queues:
            return []
        
        messages = []
        queue = self.message_queues[agent_id]
        
        # Get messages from the agent's queue
        for _ in range(min(limit, queue.qsize())):
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.1)
                messages.append(message)
            except asyncio.TimeoutError:
                break
        
        return messages
    
    def _add_to_history(self, message: Message):
        self.message_history.append(message.to_dict())
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)  # Remove oldest message
    
    async def send_request_response(self, sender_id: str, receiver_id: str, 
                                  request_content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        correlation_id = str(uuid.uuid4())
        
        # Send request
        request_msg = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            msg_type=MessageType.REQUEST,
            content=request_content,
            correlation_id=correlation_id
        )
        
        if not await self.send_message(request_msg):
            return None
        
        # Wait for response
        timeout = 10  # seconds
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            messages = await self.get_messages_for_agent(sender_id, 10)
            
            for msg in messages:
                if (msg.correlation_id == correlation_id and 
                    msg.type == MessageType.RESPONSE and 
                    msg.sender_id == receiver_id):
                    return msg.content
            
            await asyncio.sleep(0.1)  # Brief pause before checking again
        
        self.logger.warning(f"Timeout waiting for response to request {correlation_id}")
        return None
    
    def get_communication_stats(self) -> Dict[str, Any]:
        return {
            "active_agents": len(self.active_connections),
            "total_messages_processed": len(self.message_history),
            "max_history_size": self.max_history_size,
            "agent_queues": {agent_id: queue.qsize() for agent_id, queue in self.message_queues.items()}
        }
    
    async def heartbeat(self, agent_id: str) -> bool:
        if agent_id in self.active_connections:
            self.active_connections[agent_id]["last_heartbeat"] = datetime.now()
            self.active_connections[agent_id]["status"] = "active"
            return True
        return False
    
    def get_active_agents(self) -> List[str]:
        return list(self.active_connections.keys())
    
    async def process_agent_messages(self, agent_id: str, message_handler_func):
        """
        Process messages for a specific agent using the provided handler function
        """
        if agent_id not in self.message_queues:
            return
        
        messages = await self.get_messages_for_agent(agent_id, 100)  # Get up to 100 messages
        
        for message in messages:
            try:
                # Process the message with the agent's handler
                await message_handler_func(message)
                message.status = "processed"
            except Exception as e:
                self.logger.error(f"Error processing message for agent {agent_id}: {e}")
                message.status = "failed"
    
    def get_message_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent message history
        """
        return self.message_history[-limit:]