#!/usr/bin/env python3
"""
Quick test to verify the document processing pipeline is working.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")
    
    try:
        from data_pipeline.document_processing.parsers.text_parser import TextParser
        print("✅ TextParser imported successfully")
        
        parser = TextParser()
        print("✅ TextParser created successfully")
        
        # Test parsing a simple text
        test_text = "This is a test document.\nIt has multiple lines.\nAnd some content."
        result = parser.parse_text(test_text)
        print(f"✅ Text parsed successfully: {len(result.text)} characters")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_document_processor():
    """Test document processor."""
    print("\n🧪 Testing document processor...")
    
    try:
        from data_pipeline.document_processing.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        processor = DocumentProcessor()
        print("✅ DocumentProcessor created successfully")
        
        return True
    except Exception as e:
        print(f"❌ DocumentProcessor test failed: {e}")
        return False

def test_vector_stores():
    """Test vector stores."""
    print("\n🧪 Testing vector stores...")
    
    try:
        from data_pipeline.vector_stores.backends.memory_store import MemoryVectorStore
        print("✅ MemoryVectorStore imported successfully")
        
        from data_pipeline.vector_stores.schemas import VectorStoreConfig, VectorStoreType
        print("✅ Vector store schemas imported successfully")
        
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="test",
            embedding_dimension=384
        )
        store = MemoryVectorStore(config)
        print("✅ MemoryVectorStore created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Quick Test - Document Processing Pipeline")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_document_processor,
        test_vector_stores
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print("📊 Test Results")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
