#!/usr/bin/env python3
"""
Simple test for Brand Agent Phase 1 implementation.
Tests only the domain models without external dependencies.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_brand_agent_models():
    """Test Brand Agent domain models."""
    print("🧪 Testing Brand Agent Models...")

    try:
        from app.domain.models.brand_agent import (
            BrandAgent,
            BrandAgentConfiguration,
            BrandAgentType,
            BrandKnowledge,
            BrandPersonality,
            ConversationChannel,
            ConversationMessage,
            ConversationSession,
            KnowledgeType,
            PersonalityTrait,
        )

        # Test BrandPersonality
        personality = BrandPersonality(
            traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL],
            tone="professional",
            communication_style="helpful",
            response_length="medium",
            formality_level="semi-formal",
            emoji_usage=False,
            custom_phrases=["How can I help you today?"]
        )
        print(f"✅ Created BrandPersonality with traits: {personality.traits}")

        # Test BrandAgentConfiguration
        configuration = BrandAgentConfiguration(
            max_response_length=500,
            response_timeout_seconds=30,
            supported_channels=[ConversationChannel.WEBSITE_CHAT, ConversationChannel.EMAIL],
            escalation_triggers=["human agent", "manager"],
            business_hours={"monday": "9-17", "tuesday": "9-17"},
            auto_responses={"greeting": "Hello! How can I assist you?"}
        )
        print(f"✅ Created BrandAgentConfiguration with {len(configuration.supported_channels)} channels")

        # Test BrandAgent
        agent = BrandAgent(
            name="Customer Support Agent",
            brand_id="test-brand-123",
            agent_type=BrandAgentType.CUSTOMER_SUPPORT,
            owner_id="user-123",
            description="Helps customers with their questions and issues",
            personality=personality,
            configuration=configuration,
        )

        print(f"✅ Created Brand Agent: {agent.name}")
        print(f"   - ID: {agent.id}")
        print(f"   - Type: {agent.agent_type}")
        print(f"   - Active: {agent.is_active}")
        print(f"   - Deployed: {agent.is_deployed}")

        # Test agent methods
        agent.activate()
        print(f"   - Activated: {agent.is_active}")

        agent.deploy_to_channel(ConversationChannel.WEBSITE_CHAT)
        print(f"   - Deployed to: {agent.deployment_channels}")

        agent.add_knowledge_item("knowledge-123")
        print(f"   - Knowledge items: {agent.knowledge_items}")

        print(f"   - Success rate: {agent.success_rate}%")
        print(f"   - Performance: {agent.is_performing_well}")

        # Test BrandKnowledge
        knowledge = BrandKnowledge(
            title="Product Return Policy",
            content="Our return policy allows customers to return items within 30 days...",
            knowledge_type=KnowledgeType.POLICIES,
            tags=["returns", "policy", "customer-service"],
            priority=8,
            source_url="https://example.com/returns"
        )

        print(f"✅ Created Knowledge Item: {knowledge.title}")
        print(f"   - ID: {knowledge.id}")
        print(f"   - Type: {knowledge.knowledge_type}")
        print(f"   - Priority: {knowledge.priority}")

        # Test ConversationSession
        session = ConversationSession(
            brand_agent_id=agent.id,
            session_token="session-123",
            channel=ConversationChannel.WEBSITE_CHAT,
        )

        print(f"✅ Created Conversation Session: {session.id}")
        print(f"   - Agent ID: {session.brand_agent_id}")
        print(f"   - Channel: {session.channel}")
        print(f"   - Status: {session.status}")

        # Test ConversationMessage
        message = ConversationMessage(
            session_id=session.id,
            sender_type="user",
            content="Hello, I need help with my order",
        )

        print(f"✅ Created Conversation Message: {message.id}")
        print(f"   - Session ID: {message.session_id}")
        print(f"   - Sender: {message.sender_type}")
        print(f"   - Content: {message.content[:30]}...")

        return True

    except Exception as e:
        print(f"❌ Error testing models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enums_and_types():
    """Test all enums and types."""
    print("\n🧪 Testing Enums and Types...")

    try:
        from app.domain.models.brand_agent import (
            BrandAgentType,
            ConversationChannel,
            KnowledgeType,
            PersonalityTrait,
        )

        print("✅ BrandAgentType values:")
        for agent_type in BrandAgentType:
            print(f"   - {agent_type.value}")

        print("✅ PersonalityTrait values:")
        for trait in PersonalityTrait:
            print(f"   - {trait.value}")

        print("✅ ConversationChannel values:")
        for channel in ConversationChannel:
            print(f"   - {channel.value}")

        print("✅ KnowledgeType values:")
        for knowledge_type in KnowledgeType:
            print(f"   - {knowledge_type.value}")

        return True

    except Exception as e:
        print(f"❌ Error testing enums: {e}")
        return False


def test_validation():
    """Test model validation."""
    print("\n🧪 Testing Model Validation...")

    try:
        from app.domain.models.brand_agent import (
            BrandPersonality,
            PersonalityTrait,
        )

        # Test personality trait limit
        try:
            personality = BrandPersonality(
                traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL,
                       PersonalityTrait.PROFESSIONAL, PersonalityTrait.KNOWLEDGEABLE,
                       PersonalityTrait.EMPATHETIC, PersonalityTrait.CONFIDENT]  # 6 traits (max is 5)
            )
            print("❌ Should have failed with too many traits")
            return False
        except ValueError as e:
            print(f"✅ Correctly validated trait limit: {e}")

        # Test valid personality
        personality = BrandPersonality(
            traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL]
        )
        print(f"✅ Valid personality with {len(personality.traits)} traits")

        return True

    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        return False


def test_business_logic():
    """Test business logic methods."""
    print("\n🧪 Testing Business Logic...")

    try:
        from app.domain.models.brand_agent import (
            BrandAgent,
            BrandAgentConfiguration,
            BrandAgentType,
            ConversationChannel,
        )

        # Create agent with proper configuration
        configuration = BrandAgentConfiguration(
            supported_channels=[ConversationChannel.WEBSITE_CHAT]
        )

        agent = BrandAgent(
            name="Test Agent",
            brand_id="test-brand",
            agent_type=BrandAgentType.CUSTOMER_SUPPORT,
            owner_id="user-123",
            configuration=configuration,
        )

        # Test activation/deactivation
        assert agent.is_active == True, "Agent should be active by default"

        agent.deactivate()
        assert agent.is_active == False, "Agent should be inactive after deactivation"
        assert agent.is_deployed == False, "Agent should not be deployed when deactivated"

        agent.activate()
        assert agent.is_active == True, "Agent should be active after activation"

        # Test deployment
        agent.deploy_to_channel(ConversationChannel.WEBSITE_CHAT)
        assert agent.is_deployed == True, "Agent should be deployed"
        assert ConversationChannel.WEBSITE_CHAT in agent.deployment_channels, "Channel should be in deployment list"

        # Test knowledge management
        agent.add_knowledge_item("knowledge-1")
        assert "knowledge-1" in agent.knowledge_items, "Knowledge item should be added"

        agent.remove_knowledge_item("knowledge-1")
        assert "knowledge-1" not in agent.knowledge_items, "Knowledge item should be removed"

        print("✅ All business logic tests passed")
        return True

    except Exception as e:
        print(f"❌ Error testing business logic: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Simple Brand Agent Phase 1 Tests")
    print("=" * 60)

    tests = [
        test_brand_agent_models,
        test_enums_and_types,
        test_validation,
        test_business_logic,
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
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Phase 1 models are working correctly.")
        print("\n📋 Phase 1 Implementation Status:")
        print("✅ Brand Agent domain models")
        print("✅ Knowledge management models")
        print("✅ Conversation models")
        print("✅ Model validation")
        print("✅ Business logic methods")
        print("✅ Enum definitions")
        print("✅ Type safety")

        print("\n🎯 Ready for:")
        print("- Service layer implementation")
        print("- API endpoint implementation")
        print("- Frontend integration")
        print("- Database persistence")

        return True
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
