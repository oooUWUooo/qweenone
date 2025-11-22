"""
OpenRouter Integration Demo for Agentic Architecture System
Demonstrates the full capabilities of the OpenRouter integration
"""
import asyncio
import os
import json
from typing import Dict, Any, List
from datetime import datetime

from src.api_integrations import APIManager, APIConfig
from src.api_integrations.llm_provider import ProviderType
from src.api_router import APIRouter, ProviderRouter, RouterConfig
from src.api_router.router_config import LoadBalancingStrategy, FailoverStrategy
from src.agents.ai_agent import AIAgent, AIAgentConfig, SpecializedAIAgent, MultiModelAIAgent
from src.utils.logger import setup_logger


async def demonstrate_api_manager():
    """Demonstrate the API Manager capabilities"""
    print("=== API Manager Demonstration ===")
    logger = setup_logger("APIManagerDemo")
    
    # Create API manager
    api_manager = APIManager()
    
    # Configure OpenRouter
    openrouter_config = APIConfig(
        provider=ProviderType.OPENROUTER,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        model="openchat/openchat-7b",
        temperature=0.7,
        max_tokens=500
    )
    
    # Register the provider
    success = api_manager.register_provider(openrouter_config)
    if not success:
        print("Warning: Could not register OpenRouter provider (API key may be missing)")
        # Use a mock provider for demonstration
        openrouter_config.api_key = "sk-mock-key-for-demo"
        success = api_manager.register_provider(openrouter_config)
    
    if success:
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = await api_manager.chat_completion(messages)
        print(f"API Response Success: {response.success}")
        if response.success:
            content = response.data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
            print(f"Response: {content}")
            print(f"Tokens used: {response.tokens_used}")
            print(f"Cost: ${response.cost:.6f}")
    
    # Show statistics
    stats = api_manager.get_statistics()
    print(f"API Manager Stats: {json.dumps(stats, indent=2)}")
    
    # Clean up
    await api_manager.close_all()
    print()


async def demonstrate_router():
    """Demonstrate the API Router capabilities"""
    print("=== API Router Demonstration ===")
    
    # Create router
    router = APIRouter()
    
    # Configure providers
    openrouter_config = APIConfig(
        provider=ProviderType.OPENROUTER,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        model="openchat/openchat-7b",
        temperature=0.7,
        max_tokens=500
    )
    
    openai_config = APIConfig(
        provider=ProviderType.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY", "your-openai-key-here"),
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    
    # Register providers
    router.register_provider(openrouter_config)
    router.register_provider(openai_config)
    
    # Add a custom routing rule
    from src.api_router.api_router import RouteRule
    rule = RouteRule(
        name="coding_tasks",
        condition="task_type",
        condition_value="code",
        provider=ProviderType.OPENROUTER,
        priority=1,
        description="Route coding tasks to OpenRouter"
    )
    router.add_route_rule(rule)
    
    # Test routing
    messages = [
        {"role": "user", "content": "Write a Python function to calculate factorial."}
    ]
    
    response = await router.route_request(messages, model="openchat/openchat-7b")
    print(f"Routing Response Success: {response.success}")
    if response.success:
        content = response.data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
        routing_info = response.data.get("routing_info", {})
        print(f"Response: {content[:200]}...")  # First 200 chars
        print(f"Routed to: {routing_info.get('provider_used', 'Unknown')}")
    
    # Show routing statistics
    stats = router.get_routing_statistics()
    print(f"Routing Stats: {json.dumps(stats, indent=2)}")
    
    # Get recommendations
    recommendations = router.get_recommendations()
    print(f"Recommendations: {recommendations}")
    
    # Clean up
    await router.close()
    print()


async def demonstrate_provider_router():
    """Demonstrate the Provider Router capabilities"""
    print("=== Provider Router Demonstration ===")
    
    # Create provider router
    provider_router = ProviderRouter()
    
    # Configure providers with preferences
    openrouter_config = APIConfig(
        provider=ProviderType.OPENROUTER,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        model="openchat/openchat-7b",
        temperature=0.7,
        max_tokens=500
    )
    
    anthropic_config = APIConfig(
        provider=ProviderType.ANTHROPIC,
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key-here"),
        model="claude-3-haiku",
        temperature=0.7,
        max_tokens=500
    )
    
    # Register providers
    provider_router.register_provider(openrouter_config)
    provider_router.register_provider(anthropic_config)
    
    # Set provider preferences
    from src.api_router.provider_router import ProviderPreference
    openrouter_pref = ProviderPreference(
        provider=ProviderType.OPENROUTER,
        priority=1,
        max_concurrent_requests=5,
        weight=1.0
    )
    anthropic_pref = ProviderPreference(
        provider=ProviderType.ANTHROPIC,
        priority=2,
        max_concurrent_requests=3,
        weight=0.8
    )
    
    provider_router.set_provider_preference(openrouter_pref)
    provider_router.set_provider_preference(anthropic_pref)
    
    # Add fallback relationship
    provider_router.add_fallback_provider(ProviderType.OPENROUTER, ProviderType.ANTHROPIC)
    
    # Test routing with provider
    messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    
    response = await provider_router.route_request(messages)
    print(f"Provider Routing Response Success: {response.success}")
    if response.success:
        content = response.data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
        routing_info = response.data.get("routing_info", {})
        print(f"Response: {content[:200]}...")  # First 200 chars
        print(f"Provider used: {routing_info.get('provider_used', 'Unknown')}")
    
    # Show provider statistics
    stats = provider_router.get_provider_statistics()
    print(f"Provider Stats: {json.dumps(stats, indent=2)}")
    
    # Run health checks
    health_results = await provider_router.run_health_check_all_providers()
    print(f"Health Check Results: {health_results}")
    
    # Clean up
    await provider_router.close()
    print()


async def demonstrate_configurable_router():
    """Demonstrate the Configurable Router capabilities"""
    print("=== Configurable Router Demonstration ===")
    
    # Create router config
    config = RouterConfig()
    config.default_strategy = LoadBalancingStrategy.COST_OPTIMIZED
    config.failover_strategy = FailoverStrategy.IMMEDIATE_FAILOVER
    config.cost_optimization_enabled = True
    
    # Create configurable router
    configurable_router = ConfigurableAPIRouter(config)
    await configurable_router.initialize()
    
    # Get current config
    current_config = configurable_router.get_config()
    print(f"Default strategy: {current_config.default_strategy.value}")
    print(f"Cost optimization enabled: {current_config.cost_optimization_enabled}")
    
    # Get security and performance config
    security_config = configurable_router.get_security_config()
    performance_config = configurable_router.get_performance_config()
    
    print(f"Security config: {json.dumps(security_config, indent=2)}")
    print(f"Performance config: {json.dumps(performance_config, indent=2)}")
    
    # Shutdown router
    await configurable_router.shutdown()
    print()


async def demonstrate_ai_agents():
    """Demonstrate AI agents with OpenRouter integration"""
    print("=== AI Agent Demonstration ===")
    
    # Create a basic AI agent using OpenRouter
    agent_config = AIAgentConfig(
        name="OpenRouterAssistant",
        description="An AI assistant using OpenRouter",
        provider_type=ProviderType.OPENROUTER,
        model="openchat/openchat-7b",
        temperature=0.7,
        max_tokens=500,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        system_prompt="You are a helpful AI assistant powered by OpenRouter."
    )
    
    agent = AIAgent(agent_config)
    
    # Execute a task
    task = {
        "id": "task_1",
        "title": "Explain machine learning",
        "description": "Provide a simple explanation of what machine learning is."
    }
    
    result = await agent.execute_task(task)
    print(f"Agent execution success: {result.get('status') == 'completed'}")
    if result.get("status") == "completed":
        print(f"Result: {result.get('result', '')[:200]}...")  # First 200 chars
        print(f"Tokens used: {result.get('tokens_used', 0)}")
        print(f"Cost: ${result.get('cost', 0):.6f}")
        print(f"Model used: {result.get('model_used', 'Unknown')}")
    
    # Show conversation summary
    summary = agent.get_conversation_summary()
    print(f"Conversation summary: {summary}")
    
    # Get API stats
    api_stats = agent.get_api_stats()
    print(f"API Stats: {json.dumps(api_stats, indent=2)}")
    
    # Clean up
    await agent.close()
    print()


async def demonstrate_specialized_agents():
    """Demonstrate specialized AI agents"""
    print("=== Specialized Agent Demonstration ===")
    
    # Create specialized agents
    coding_agent_config = AIAgentConfig(
        name="CodingAssistant",
        description="A coding specialist using OpenRouter",
        provider_type=ProviderType.OPENROUTER,
        model="openchat/openchat-7b",
        temperature=0.5,  # Lower temperature for more deterministic code
        max_tokens=1000,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"),
        system_prompt="You are an expert programmer. Write clean, efficient, and well-documented code."
    )
    
    coding_agent = SpecializedAIAgent(coding_agent_config, specialization="code")
    
    # Execute a coding task
    coding_task = {
        "id": "coding_task_1",
        "title": "Python function",
        "description": "Write a Python function to reverse a string",
        "params": {"language": "python", "requirements": ["efficient", "well-documented"]}
    }
    
    coding_result = await coding_agent.execute_task(coding_task)
    print(f"Coding agent success: {coding_result.get('status') == 'completed'}")
    if coding_result.get("status") == "completed":
        print(f"Code result: {coding_result.get('result', '')[:300]}...")  # First 300 chars
    
    # Get specialization insights
    insights = coding_agent.get_specialization_insights()
    print(f"Specialization insights: {json.dumps(insights, indent=2)}")
    
    # Clean up
    await coding_agent.close()
    print()


async def demonstrate_multi_model_agent():
    """Demonstrate multi-model agent that can switch between providers"""
    print("=== Multi-Model Agent Demonstration ===")
    
    # Create configurations for different providers
    openrouter_config = AIAgentConfig(
        name="MultiModelAgent",
        description="Agent that can use multiple providers",
        provider_type=ProviderType.OPENROUTER,
        model="openchat/openchat-7b",
        temperature=0.7,
        max_tokens=500,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here")
    )
    
    openai_config = AIAgentConfig(
        name="MultiModelAgent",
        description="Agent that can use multiple providers",
        provider_type=ProviderType.OPENAI,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        api_key=os.getenv("OPENAI_API_KEY", "your-openai-key-here")
    )
    
    # Create multi-model agent
    multi_agent = MultiModelAIAgent([openrouter_config, openai_config])
    
    # Execute task with default provider
    task = {
        "id": "multi_task_1",
        "title": "General question",
        "description": "What are the benefits of renewable energy?"
    }
    
    result = await multi_agent.execute_task(task)
    print(f"Multi-model agent success: {result.get('status') == 'completed'}")
    if result.get("status") == "completed":
        print(f"Result: {result.get('result', '')[:200]}...")  # First 200 chars
        print(f"Provider used: {result.get('provider_used', 'Unknown')}")
    
    # Execute task with specific provider
    openai_result = await multi_agent.execute_task_with_provider(task, ProviderType.OPENAI)
    print(f"OpenAI-specific result provider: {openai_result.get('provider_used', 'Unknown')}")
    
    # Switch provider temporarily
    switch_success = await multi_agent.switch_provider(ProviderType.OPENROUTER)
    print(f"Provider switch success: {switch_success}")
    
    # Clean up
    await multi_agent.close()
    print()


async def demonstrate_complete_integration():
    """Demonstrate complete integration of all components"""
    print("=== Complete Integration Demonstration ===")
    
    # Create API manager
    api_manager = APIManager()
    
    # Register multiple providers
    providers = [
        APIConfig(
            provider=ProviderType.OPENROUTER,
            api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here"),
            model="openchat/openchat-7b",
            temperature=0.7,
            max_tokens=500
        ),
        APIConfig(
            provider=ProviderType.ANTHROPIC,
            api_key=os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key-here"),
            model="claude-3-haiku",
            temperature=0.7,
            max_tokens=500
        )
    ]
    
    for provider_config in providers:
        api_manager.register_provider(provider_config)
    
    # Create router with API manager
    router = APIRouter()
    # Manually set the API manager to use the same providers
    for provider_config in providers:
        router.register_provider(provider_config)
    
    # Create an AI agent that uses the router
    agent_config = AIAgentConfig(
        name="IntegratedAgent",
        description="Agent using full integration stack",
        provider_type=ProviderType.OPENROUTER,
        model="openchat/openchat-7b",
        temperature=0.7,
        max_tokens=500,
        api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here")
    )
    
    agent = AIAgent(agent_config)
    # Replace the agent's internal API manager with our shared one
    agent.api_manager = api_manager
    
    # Execute a complex task
    complex_task = {
        "id": "complex_task_1",
        "title": "Research and summarize",
        "description": "Research the impact of AI on healthcare and provide a summary",
        "params": {
            "length": "concise",
            "focus": ["diagnostics", "treatment", "efficiency"],
            "requirements": ["factual", "up-to-date"]
        }
    }
    
    result = await agent.execute_task(complex_task)
    print(f"Integrated agent success: {result.get('status') == 'completed'}")
    if result.get("status") == "completed":
        print(f"Result: {result.get('result', '')[:300]}...")  # First 300 chars
        print(f"Total tokens: {result.get('tokens_used', 0)}")
        print(f"Total cost: ${result.get('cost', 0):.6f}")
    
    # Show overall statistics
    api_stats = api_manager.get_statistics()
    print(f"Overall API Stats: {json.dumps(api_stats, indent=2)}")
    
    # Clean up
    await agent.close()
    await router.close()
    await api_manager.close_all()
    print()


async def main():
    """Main function to run all demonstrations"""
    print("OpenRouter Integration Demo")
    print("=" * 50)
    
    await demonstrate_api_manager()
    await demonstrate_router()
    await demonstrate_provider_router()
    await demonstrate_configurable_router()
    await demonstrate_ai_agents()
    await demonstrate_specialized_agents()
    await demonstrate_multi_model_agent()
    await demonstrate_complete_integration()
    
    print("Demo completed successfully!")
    print("The Agentic Architecture System now supports:")
    print("- Multiple LLM providers including OpenRouter")
    print("- Intelligent API routing with various strategies")
    print("- Configurable provider management")
    print("- Specialized AI agents with memory")
    print("- Cost optimization and performance monitoring")
    print("- Fallback and redundancy mechanisms")


if __name__ == "__main__":
    asyncio.run(main())