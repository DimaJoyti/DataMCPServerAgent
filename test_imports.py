#!/usr/bin/env python3
"""
Simple test to verify that all orchestration modules can be imported correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test importing all orchestration modules."""
    
    print("🔍 Testing imports for orchestration system...")
    
    try:
        print("   📦 Testing advanced reasoning import...")
        from src.agents.advanced_reasoning import AdvancedReasoningEngine, ReasoningStepType, ReasoningStep, ReasoningChain
        print("   ✅ Advanced reasoning imported successfully")
        
        print("   📦 Testing meta reasoning import...")
        from src.agents.meta_reasoning import MetaReasoningEngine, MetaReasoningStrategy, CognitiveState
        print("   ✅ Meta reasoning imported successfully")
        
        print("   📦 Testing advanced planning import...")
        from src.agents.advanced_planning import AdvancedPlanningEngine, Plan, Action, Condition
        print("   ✅ Advanced planning imported successfully")
        
        print("   📦 Testing reflection systems import...")
        from src.agents.reflection_systems import AdvancedReflectionEngine, ReflectionType, ReflectionInsight
        print("   ✅ Reflection systems imported successfully")
        
        print("   📦 Testing orchestration coordinator import...")
        from src.core.orchestration_main import OrchestrationCoordinator
        print("   ✅ Orchestration coordinator imported successfully")
        
        print("   📦 Testing memory persistence import...")
        from src.memory.memory_persistence import MemoryDatabase
        print("   ✅ Memory persistence imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of imported modules."""
    
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test enum values
        from src.agents.advanced_reasoning import ReasoningStepType
        print(f"   ✅ ReasoningStepType has {len(ReasoningStepType)} values")
        
        from src.agents.meta_reasoning import MetaReasoningStrategy
        print(f"   ✅ MetaReasoningStrategy has {len(MetaReasoningStrategy)} values")
        
        from src.agents.reflection_systems import ReflectionType
        print(f"   ✅ ReflectionType has {len(ReflectionType)} values")
        
        # Test class instantiation (without dependencies)
        from src.memory.memory_persistence import MemoryDatabase
        
        # Create a temporary database to test
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db = MemoryDatabase(tmp_db.name)
            print("   ✅ MemoryDatabase created successfully")
            
            # Clean up
            os.unlink(tmp_db.name)
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Import Tests for Orchestration System")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    if success:
        print("\n✨ All tests passed! The orchestration system is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
