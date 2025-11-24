#!/usr/bin/env python3
"""
Modernization Integration Demo

Демонстрирует интеграцию новых современных инструментов с существующей системой qweenone.
Показывает, как Prefect заменяет AdvancedTaskManager с обратной совместимостью.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Modern imports (with fallbacks)
try:
    from src.workflow_engine.prefect_manager import (
        PrefectWorkflowManager, 
        ModernTask,
        PrefectTaskManagerAdapter
    )
    PREFECT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Prefect integration not available: {e}")
    print("📦 Please install: pip install prefect>=3.0.0")
    PREFECT_AVAILABLE = False

# Legacy imports for comparison
try:
    from src.task_manager.advanced_task_manager import AdvancedTaskManager
    from src.task_manager.task_decomposer import TaskDecomposer
    from src.builders.agent_builder import AgentBuilder
    LEGACY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Legacy system not available: {e}")
    LEGACY_AVAILABLE = False

import time
from datetime import datetime
from typing import List, Dict, Any


class ModernizationDemo:
    """
    Демонстрирует переход от legacy AdvancedTaskManager к современной Prefect-based системе
    """
    
    def __init__(self):
        self.prefect_manager = None
        self.legacy_manager = None
        
        if PREFECT_AVAILABLE:
            self.prefect_manager = PrefectWorkflowManager(concurrent_limit=3)
            print("✅ Prefect Workflow Manager initialized")
        
        if LEGACY_AVAILABLE:
            self.legacy_manager = AdvancedTaskManager()
            print("✅ Legacy Task Manager initialized")
    
    async def demonstrate_modern_workflow(self):
        """Демонстрация современного Prefect workflow"""
        print("\n🚀 === MODERN PREFECT WORKFLOW DEMO ===")
        
        if not self.prefect_manager:
            print("❌ Prefect not available, skipping demo")
            return
        
        # Создаем современные задачи
        modern_tasks = [
            ModernTask(
                id="modern_1",
                title="Analyze Requirements",
                description="Analyze project requirements using AI agent",
                agent_type="analysis",
                priority=5,  # High priority
                tags=["analysis", "ai", "modern"]
            ),
            ModernTask(
                id="modern_2", 
                title="Generate Code Structure",
                description="Generate basic code structure based on requirements",
                agent_type="code_writer",
                priority=4,
                dependencies=["modern_1"],  # Depends on analysis
                tags=["coding", "generation", "modern"]
            ),
            ModernTask(
                id="modern_3",
                title="Create Tests",
                description="Create comprehensive tests for generated code",
                agent_type="tester",
                priority=3,
                dependencies=["modern_2"],  # Depends on code generation
                tags=["testing", "validation", "modern"]
            )
        ]
        
        print(f"📋 Created {len(modern_tasks)} modern tasks with dependencies")
        
        # Выполняем workflow
        start_time = time.time()
        
        result = await self.prefect_manager.execute_workflow(
            tasks=modern_tasks,
            workflow_name="modernization_demo",
            parallel_execution=True  # Prefect автоматически учтет зависимости
        )
        
        execution_time = time.time() - start_time
        
        print(f"⏱️ Execution completed in {execution_time:.2f}s")
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📊 Tasks: {result.get('successful_tasks', 0)}/{result.get('total_tasks', 0)} successful")
        
        # Показываем детали выполнения
        print("\n📈 Task Results:")
        for i, task_result in enumerate(result.get('task_results', []), 1):
            status = "✅" if task_result.get('success', False) else "❌"
            exec_time = task_result.get('execution_time', 0)
            print(f"  {status} Task {i}: {exec_time:.2f}s")
        
        return result
    
    async def demonstrate_task_decomposition(self):
        """Демонстрация современной декомпозиции задач"""
        print("\n🧠 === TASK DECOMPOSITION DEMO ===")
        
        if not self.prefect_manager:
            print("❌ Prefect not available, skipping decomposition demo")
            return
        
        complex_task = "Create a web scraper for Instagram posts with authentication"
        
        print(f"🎯 Complex Task: {complex_task}")
        print("🔄 Decomposing into manageable subtasks...")
        
        try:
            result = await self.prefect_manager.decompose_and_execute(
                task_description=complex_task,
                iterations=3,
                auto_execute=False  # Just decompose, don't execute yet
            )
            
            decomposition = result.get('decomposition', {})
            modern_tasks = result.get('modern_tasks', [])
            
            print(f"📋 Original task decomposed into {decomposition.get('total_subtasks', 0)} subtasks")
            print(f"🔄 Across {decomposition.get('total_iterations', 0)} iterations")
            
            # Показываем структуру декомпозиции
            for iteration in decomposition.get('iterations', []):
                print(f"\n🔸 Iteration {iteration['iteration_number']}: {iteration['focus']}")
                for subtask in iteration['subtasks']:
                    print(f"  - {subtask['title']}")
            
            # Опционально выполняем декомпозированные задачи
            execute = input("\n❓ Execute decomposed tasks? (y/n): ").strip().lower()
            if execute == 'y':
                print("🚀 Executing decomposed workflow...")
                
                # Конвертируем в ModernTask и выполняем
                tasks_to_execute = []
                for task_data in modern_tasks:
                    modern_task = ModernTask(
                        id=task_data['id'],
                        title=task_data['title'],
                        description=task_data['description'],
                        agent_type=task_data.get('agent_type', 'default'),
                        tags=['decomposed', 'auto_generated']
                    )
                    tasks_to_execute.append(modern_task)
                
                execution_result = await self.prefect_manager.execute_workflow(
                    tasks=tasks_to_execute,
                    workflow_name="decomposed_scraper_workflow"
                )
                
                print(f"✅ Decomposed workflow completed: {execution_result.get('success', False)}")
                
        except Exception as e:
            print(f"❌ Decomposition failed: {e}")
    
    async def compare_legacy_vs_modern(self):
        """Сравнение legacy и современной системы"""
        print("\n⚡ === LEGACY VS MODERN COMPARISON ===")
        
        # Определяем тестовые задачи для сравнения
        test_tasks = [
            {
                "title": "Process Data File",
                "description": "Process and analyze uploaded data file",
                "agent_type": "data_processing"
            },
            {
                "title": "Generate Report", 
                "description": "Generate comprehensive analysis report",
                "agent_type": "report_generator"
            },
            {
                "title": "Send Notifications",
                "description": "Send completion notifications to stakeholders", 
                "agent_type": "communication"
            }
        ]
        
        results = {}
        
        # Тест legacy системы
        if self.legacy_manager and LEGACY_AVAILABLE:
            print("\n📊 Testing Legacy AdvancedTaskManager...")
            
            legacy_start = time.time()
            try:
                # Создаем задачи в legacy формате
                legacy_tasks = []
                for i, task_data in enumerate(test_tasks):
                    legacy_task = self.legacy_manager.create_task(
                        title=task_data["title"],
                        description=task_data["description"],
                        task_type=task_data.get("agent_type", "default"),
                        priority=3
                    )
                    legacy_tasks.append(legacy_task)
                
                # Выполняем через legacy систему
                legacy_success_count = 0
                for task in legacy_tasks:
                    success = self.legacy_manager.submit_task(task)
                    if success:
                        legacy_success_count += 1
                
                legacy_time = time.time() - legacy_start
                results['legacy'] = {
                    'execution_time': legacy_time,
                    'success_count': legacy_success_count,
                    'total_tasks': len(legacy_tasks),
                    'success_rate': legacy_success_count / len(legacy_tasks) * 100
                }
                
                print(f"⏱️ Legacy execution: {legacy_time:.2f}s")
                print(f"✅ Success rate: {results['legacy']['success_rate']:.1f}%")
                
            except Exception as e:
                print(f"❌ Legacy execution failed: {e}")
                results['legacy'] = {'error': str(e)}
        
        # Тест современной системы
        if self.prefect_manager:
            print("\n🚀 Testing Modern Prefect System...")
            
            modern_start = time.time()
            try:
                # Создаем modern задачи
                modern_tasks = []
                for i, task_data in enumerate(test_tasks):
                    modern_task = ModernTask(
                        id=f"comparison_{i+1}",
                        title=task_data["title"],
                        description=task_data["description"],
                        agent_type=task_data.get("agent_type", "default"),
                        tags=['comparison', 'modern']
                    )
                    modern_tasks.append(modern_task)
                
                # Выполняем через современную систему
                modern_result = await self.prefect_manager.execute_workflow(
                    tasks=modern_tasks,
                    workflow_name="comparison_modern"
                )
                
                modern_time = time.time() - modern_start
                results['modern'] = {
                    'execution_time': modern_time,
                    'success_count': modern_result.get('successful_tasks', 0),
                    'total_tasks': modern_result.get('total_tasks', 0),
                    'success_rate': (modern_result.get('successful_tasks', 0) / 
                                   modern_result.get('total_tasks', 1) * 100),
                    'orchestration_overhead': modern_result.get('total_execution_time', 0) - modern_time
                }
                
                print(f"⏱️ Modern execution: {modern_time:.2f}s")  
                print(f"✅ Success rate: {results['modern']['success_rate']:.1f}%")
                
            except Exception as e:
                print(f"❌ Modern execution failed: {e}")
                results['modern'] = {'error': str(e)}
        
        # Сравнительный анализ
        print("\n📊 === COMPARISON RESULTS ===")
        
        if 'legacy' in results and 'error' not in results['legacy']:
            if 'modern' in results and 'error' not in results['modern']:
                legacy_time = results['legacy']['execution_time']
                modern_time = results['modern']['execution_time']
                
                speedup = legacy_time / modern_time if modern_time > 0 else 1
                
                print(f"⚡ Speed improvement: {speedup:.2f}x")
                print(f"📈 Legacy success rate: {results['legacy']['success_rate']:.1f}%")
                print(f"📈 Modern success rate: {results['modern']['success_rate']:.1f}%")
                
                if results['modern']['success_rate'] > results['legacy']['success_rate']:
                    print("✅ Modern system shows better reliability")
                
                print("\n🎯 Key Advantages of Modern System:")
                print("  • Built-in retry and failure handling")
                print("  • Real-time progress tracking") 
                print("  • Automatic dependency resolution")
                print("  • Horizontal scaling capabilities")
                print("  • Rich observability and monitoring")
                print("  • Python-native workflow definition")
                
            else:
                print("❌ Modern system test failed, cannot compare")
        else:
            print("❌ Legacy system test failed, cannot compare")
        
        return results
    
    def demonstrate_backward_compatibility(self):
        """Демонстрация обратной совместимости"""
        print("\n🔄 === BACKWARD COMPATIBILITY DEMO ===")
        
        if not PREFECT_AVAILABLE:
            print("❌ Prefect not available, skipping compatibility demo")
            return
        
        # Используем адаптер для совместимости
        adapter = PrefectTaskManagerAdapter()
        
        print("🔌 Using PrefectTaskManagerAdapter for legacy compatibility")
        
        # Создаем задачу через legacy-style API
        task = asyncio.run(adapter.create_task(
            title="Legacy Style Task",
            description="Task created using legacy-style API but executed with Prefect",
            agent_type="default",
            priority=3
        ))
        
        print(f"✅ Created task: {task.title}")
        
        # Выполняем через адаптер
        success = asyncio.run(adapter.submit_task(task))
        
        print(f"🎯 Task execution success: {success}")
        
        # Получаем статус системы в legacy формате
        status = adapter.get_system_status()
        print(f"📊 System status: {status}")
        
        print("\n💡 This demonstrates how existing code can use new Prefect backend")
        print("   without changing the API interface!")


async def main():
    """Главная функция демонстрации"""
    print("🔄 === QWEENONE MODERNIZATION INTEGRATION DEMO ===")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Проверяем доступность компонентов
    print(f"\n🔍 System Check:")
    print(f"  • Prefect available: {'✅' if PREFECT_AVAILABLE else '❌'}")
    print(f"  • Legacy system available: {'✅' if LEGACY_AVAILABLE else '❌'}")
    
    if not PREFECT_AVAILABLE and not LEGACY_AVAILABLE:
        print("❌ No systems available for demo. Please install dependencies.")
        return
    
    demo = ModernizationDemo()
    
    # Запускаем демонстрации
    try:
        # 1. Современный workflow
        await demo.demonstrate_modern_workflow()
        
        # 2. Декомпозиция задач
        await demo.demonstrate_task_decomposition()
        
        # 3. Сравнение систем
        await demo.compare_legacy_vs_modern()
        
        # 4. Обратная совместимость
        demo.demonstrate_backward_compatibility()
        
        print(f"\n🎉 === DEMO COMPLETED SUCCESSFULLY ===")
        print(f"🕒 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🚀 Next Steps for Full Modernization:")
        print(f"  1. Install modern dependencies: pip install -r requirements_modern.txt")
        print(f"  2. Run integration tests: python -m pytest tests/")
        print(f"  3. Migrate production workloads gradually")
        print(f"  4. Monitor performance and adjust configuration")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запускаем демо
    asyncio.run(main())