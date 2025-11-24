"""
Modern A2A Communication Manager with Redis/RabbitMQ Support

Provides enterprise-grade agent-to-agent communication with:
- Redis for fast message queuing and pub/sub
- RabbitMQ for reliable message delivery and routing
- Backward compatibility with legacy A2A manager
- Real-time message broadcasting
- Message persistence and replay
- Connection pooling and automatic reconnection
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        aioredis = None

try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    aio_pika = None

try:
    from ..utils.logger import setup_logger
    from .a2a_manager import Message, MessageType
except ImportError:
    setup_logger = lambda x: None
    Message = None
    MessageType = None


class CommunicationBackend(Enum):
    """Available communication backends"""
    MEMORY = "memory"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"


@dataclass
class A2AConfig:
    """Configuration for modern A2A communication"""
    backend: CommunicationBackend = CommunicationBackend.MEMORY
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_username: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_vhost: str = "/"
    rabbitmq_exchange: str = "qweenone_agents"
    
    message_ttl: int = 3600
    max_queue_size: int = 10000
    enable_persistence: bool = True
    enable_message_history: bool = True
    max_history_size: int = 10000


class ModernA2ACommunicationManager:
    """
    Modern agent-to-agent communication manager with enterprise features
    
    Supports multiple backends:
    - Memory: Fast in-process communication (default)
    - Redis: Distributed communication with pub/sub
    - RabbitMQ: Enterprise message queue with guaranteed delivery
    """
    
    def __init__(self, config: A2AConfig = None):
        """Initialize modern A2A communication manager"""
        
        self.config = config or A2AConfig()
        self.logger = setup_logger("ModernA2ACommunicationManager")
        
        self.backend = None
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Callable] = {}
        
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        self.redis_client = None
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        self.rabbitmq_exchange = None
        
        if self.logger:
            self.logger.info(f"Initializing A2A with backend: {self.config.backend.value}")
    
    async def initialize(self):
        """Initialize the selected communication backend"""
        
        if self.config.backend == CommunicationBackend.REDIS:
            await self._initialize_redis()
        elif self.config.backend == CommunicationBackend.RABBITMQ:
            await self._initialize_rabbitmq()
        else:
            await self._initialize_memory()
        
        if self.logger:
            self.logger.info(f"A2A backend initialized: {self.config.backend.value}")
    
    async def _initialize_memory(self):
        """Initialize in-memory communication"""
        self.backend = "memory"
        if self.logger:
            self.logger.info("Using in-memory communication backend")
    
    async def _initialize_redis(self):
        """Initialize Redis-based communication"""
        
        if not REDIS_AVAILABLE:
            if self.logger:
                self.logger.warning("Redis not available, falling back to memory")
            await self._initialize_memory()
            return
        
        try:
            self.redis_client = aioredis.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                password=self.config.redis_password,
                encoding="utf-8",
                decode_responses=True
            )
            
            await self.redis_client.ping()
            
            self.backend = "redis"
            
            if self.logger:
                self.logger.info(f"Redis connection established: {self.config.redis_host}:{self.config.redis_port}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Redis initialization failed: {e}, falling back to memory")
            await self._initialize_memory()
    
    async def _initialize_rabbitmq(self):
        """Initialize RabbitMQ-based communication"""
        
        if not RABBITMQ_AVAILABLE:
            if self.logger:
                self.logger.warning("RabbitMQ not available, falling back to memory")
            await self._initialize_memory()
            return
        
        try:
            connection_url = f"amqp://{self.config.rabbitmq_username}:{self.config.rabbitmq_password}@{self.config.rabbitmq_host}:{self.config.rabbitmq_port}/{self.config.rabbitmq_vhost}"
            
            self.rabbitmq_connection = await aio_pika.connect_robust(connection_url)
            self.rabbitmq_channel = await self.rabbitmq_connection.channel()
            
            self.rabbitmq_exchange = await self.rabbitmq_channel.declare_exchange(
                self.config.rabbitmq_exchange,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            self.backend = "rabbitmq"
            
            if self.logger:
                self.logger.info(f"RabbitMQ connection established: {self.config.rabbitmq_host}:{self.config.rabbitmq_port}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"RabbitMQ initialization failed: {e}, falling back to memory")
            await self._initialize_memory()
    
    async def register_agent(self, agent_id: str, handler_func: Optional[Callable] = None):
        """Register agent with communication manager"""
        
        self.active_agents[agent_id] = {
            "connected_at": datetime.now(),
            "status": "active",
            "last_heartbeat": datetime.now(),
            "message_count": 0
        }
        
        if handler_func:
            self.handlers[agent_id] = handler_func
        
        if self.backend == "memory":
            self.message_queues[agent_id] = asyncio.Queue(maxsize=self.config.max_queue_size)
            
        elif self.backend == "redis" and self.redis_client:
            await self.redis_client.hset(f"agent:{agent_id}", mapping={
                "connected_at": datetime.now().isoformat(),
                "status": "active"
            })
            
        elif self.backend == "rabbitmq" and self.rabbitmq_channel:
            queue = await self.rabbitmq_channel.declare_queue(
                f"agent_{agent_id}",
                durable=True,
                arguments={"x-message-ttl": self.config.message_ttl * 1000}
            )
            
            await queue.bind(
                self.rabbitmq_exchange,
                routing_key=f"agent.{agent_id}"
            )
        
        if self.logger:
            self.logger.info(f"Agent registered: {agent_id} (backend: {self.backend})")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister agent from communication manager"""
        
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        
        if agent_id in self.handlers:
            del self.handlers[agent_id]
        
        if self.backend == "memory" and agent_id in self.message_queues:
            del self.message_queues[agent_id]
            
        elif self.backend == "redis" and self.redis_client:
            await self.redis_client.delete(f"agent:{agent_id}")
            await self.redis_client.delete(f"queue:{agent_id}")
            
        elif self.backend == "rabbitmq" and self.rabbitmq_channel:
            try:
                await self.rabbitmq_channel.queue_delete(f"agent_{agent_id}")
            except:
                pass
        
        if self.logger:
            self.logger.info(f"Agent unregistered: {agent_id}")
    
    async def send_message(self, message: 'Message') -> bool:
        """Send message to another agent"""
        
        if not message.receiver_id:
            if self.logger:
                self.logger.error("Cannot send message without receiver_id")
            return False
        
        if message.receiver_id not in self.active_agents:
            if self.logger:
                self.logger.warning(f"Receiver {message.receiver_id} not active")
            return False
        
        message.status = "sent"
        
        if self.config.enable_message_history:
            self._add_to_history(message)
        
        try:
            if self.backend == "memory":
                await self._send_message_memory(message)
                
            elif self.backend == "redis":
                await self._send_message_redis(message)
                
            elif self.backend == "rabbitmq":
                await self._send_message_rabbitmq(message)
            
            message.status = "delivered"
            
            self.active_agents[message.receiver_id]["message_count"] += 1
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to send message: {e}")
            message.status = "failed"
            return False
    
    async def _send_message_memory(self, message: 'Message'):
        """Send message using in-memory queue"""
        if message.receiver_id in self.message_queues:
            await self.message_queues[message.receiver_id].put(message)
    
    async def _send_message_redis(self, message: 'Message'):
        """Send message using Redis"""
        message_data = json.dumps(message.to_dict())
        await self.redis_client.rpush(f"queue:{message.receiver_id}", message_data)
        
        await self.redis_client.publish(
            f"agent_channel:{message.receiver_id}",
            message_data
        )
    
    async def _send_message_rabbitmq(self, message: 'Message'):
        """Send message using RabbitMQ"""
        message_body = json.dumps(message.to_dict()).encode()
        
        await self.rabbitmq_exchange.publish(
            aio_pika.Message(
                body=message_body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
                message_id=message.id,
                timestamp=datetime.now()
            ),
            routing_key=f"agent.{message.receiver_id}"
        )
    
    async def broadcast_message(self, message: 'Message', exclude_sender: bool = True) -> int:
        """Broadcast message to all active agents"""
        
        sent_count = 0
        message.type = MessageType.BROADCAST if MessageType else "broadcast"
        
        for agent_id in list(self.active_agents.keys()):
            if exclude_sender and agent_id == message.sender_id:
                continue
            
            recipient_message = self._create_message_copy(message, agent_id)
            
            if await self.send_message(recipient_message):
                sent_count += 1
        
        if self.logger:
            self.logger.info(f"Broadcast message sent to {sent_count} agents")
        
        return sent_count
    
    async def get_messages_for_agent(self, agent_id: str, limit: int = 10) -> List['Message']:
        """Get messages for specific agent"""
        
        if agent_id not in self.active_agents:
            return []
        
        messages = []
        
        try:
            if self.backend == "memory":
                messages = await self._get_messages_memory(agent_id, limit)
                
            elif self.backend == "redis":
                messages = await self._get_messages_redis(agent_id, limit)
                
            elif self.backend == "rabbitmq":
                messages = await self._get_messages_rabbitmq(agent_id, limit)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get messages for {agent_id}: {e}")
        
        return messages
    
    async def _get_messages_memory(self, agent_id: str, limit: int) -> List['Message']:
        """Get messages from in-memory queue"""
        messages = []
        queue = self.message_queues.get(agent_id)
        
        if queue:
            for _ in range(min(limit, queue.qsize())):
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                    messages.append(message)
                except asyncio.TimeoutError:
                    break
        
        return messages
    
    async def _get_messages_redis(self, agent_id: str, limit: int) -> List['Message']:
        """Get messages from Redis"""
        messages = []
        
        for _ in range(limit):
            message_data = await self.redis_client.lpop(f"queue:{agent_id}")
            if not message_data:
                break
            
            message_dict = json.loads(message_data)
            message = self._dict_to_message(message_dict)
            messages.append(message)
        
        return messages
    
    async def _get_messages_rabbitmq(self, agent_id: str, limit: int) -> List['Message']:
        """Get messages from RabbitMQ"""
        messages = []
        
        try:
            queue = await self.rabbitmq_channel.get_queue(f"agent_{agent_id}")
            
            for _ in range(limit):
                try:
                    rabbitmq_message = await queue.get(timeout=0.1)
                    if rabbitmq_message:
                        message_dict = json.loads(rabbitmq_message.body.decode())
                        message = self._dict_to_message(message_dict)
                        messages.append(message)
                        await rabbitmq_message.ack()
                    else:
                        break
                except asyncio.TimeoutError:
                    break
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"RabbitMQ message retrieval failed: {e}")
        
        return messages
    
    async def heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat"""
        
        if agent_id in self.active_agents:
            self.active_agents[agent_id]["last_heartbeat"] = datetime.now()
            self.active_agents[agent_id]["status"] = "active"
            
            if self.backend == "redis" and self.redis_client:
                await self.redis_client.hset(
                    f"agent:{agent_id}",
                    "last_heartbeat",
                    datetime.now().isoformat()
                )
            
            return True
        
        return False
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs"""
        return list(self.active_agents.keys())
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics"""
        
        total_messages = sum(
            agent["message_count"] for agent in self.active_agents.values()
        )
        
        return {
            "backend": self.backend,
            "active_agents": len(self.active_agents),
            "total_messages_sent": total_messages,
            "message_history_size": len(self.message_history),
            "max_history_size": self.config.max_history_size,
            "redis_available": REDIS_AVAILABLE,
            "rabbitmq_available": RABBITMQ_AVAILABLE,
            "persistence_enabled": self.config.enable_persistence,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_message_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent message history"""
        return self.message_history[-limit:]
    
    def _add_to_history(self, message: 'Message'):
        """Add message to history"""
        self.message_history.append(message.to_dict())
        
        if len(self.message_history) > self.config.max_history_size:
            self.message_history.pop(0)
    
    def _create_message_copy(self, message: 'Message', receiver_id: str) -> 'Message':
        """Create a copy of message for new receiver"""
        if Message:
            return Message(
                sender_id=message.sender_id,
                receiver_id=receiver_id,
                msg_type=message.type,
                content=message.content.copy() if isinstance(message.content, dict) else message.content,
                correlation_id=message.correlation_id
            )
        else:
            return message
    
    def _dict_to_message(self, message_dict: Dict[str, Any]) -> 'Message':
        """Convert dictionary back to Message object"""
        if Message and MessageType:
            return Message(
                msg_id=message_dict["id"],
                sender_id=message_dict["sender_id"],
                receiver_id=message_dict["receiver_id"],
                msg_type=MessageType(message_dict["type"]) if isinstance(message_dict["type"], str) else message_dict["type"],
                content=message_dict["content"],
                correlation_id=message_dict["correlation_id"]
            )
        else:
            return message_dict
    
    async def close(self):
        """Close all connections and clean up"""
        
        if self.redis_client:
            await self.redis_client.close()
            
        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()
        
        if self.logger:
            self.logger.info("A2A communication manager closed")


if __name__ == "__main__":
    """Demo modern A2A communication"""
    
    async def demo_modern_a2a():
        """Demonstrate modern A2A capabilities"""
        
        print("🔗 === MODERN A2A COMMUNICATION DEMO ===")
        
        configs = [
            (A2AConfig(backend=CommunicationBackend.MEMORY), "Memory"),
            (A2AConfig(backend=CommunicationBackend.REDIS), "Redis"),
            (A2AConfig(backend=CommunicationBackend.RABBITMQ), "RabbitMQ"),
        ]
        
        for config, name in configs:
            print(f"\n📡 Testing {name} backend:")
            
            manager = ModernA2ACommunicationManager(config)
            await manager.initialize()
            
            await manager.register_agent("agent_1")
            await manager.register_agent("agent_2")
            
            stats = manager.get_communication_stats()
            print(f"✅ Backend: {stats['backend']}")
            print(f"   Active agents: {stats['active_agents']}")
            print(f"   Persistence: {stats['persistence_enabled']}")
            
            await manager.close()
    
    asyncio.run(demo_modern_a2a())
