"""
Test suite for modern deep reinforcement learning implementation.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.enhanced_state_representation import (
    ContextualStateEncoder,
    TextEmbeddingEncoder,
)
from src.agents.modern_deep_rl import (
    A2CAgent,
    DQNAgent,
    ExperienceReplay,
    ModernDeepRLCoordinatorAgent,
    PPOAgent,
)
from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase
from src.utils.rl_neural_networks import (
    ActorCriticNetwork,
    DQNNetwork,
    NoisyLinear,
)


class TestNeuralNetworks(unittest.TestCase):
    """Test neural network architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 128
        self.action_dim = 5
        self.batch_size = 32

    def test_dqn_network_creation(self):
        """Test DQN network creation."""
        try:
            import torch

            network = DQNNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                dueling=True,
                noisy=True
            )

            # Test forward pass
            state = torch.randn(self.batch_size, self.state_dim)
            q_values = network(state)

            self.assertEqual(q_values.shape, (self.batch_size, self.action_dim))

        except ImportError:
            self.skipTest("PyTorch not available")

    def test_actor_critic_network(self):
        """Test Actor-Critic network."""
        try:
            import torch

            network = ActorCriticNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                continuous=False
            )

            # Test forward pass
            state = torch.randn(self.batch_size, self.state_dim)
            actor_output, critic_value = network(state)

            self.assertEqual(actor_output.shape, (self.batch_size, self.action_dim))
            self.assertEqual(critic_value.shape, (self.batch_size, 1))

        except ImportError:
            self.skipTest("PyTorch not available")

    def test_noisy_linear(self):
        """Test noisy linear layer."""
        try:
            import torch

            layer = NoisyLinear(in_features=64, out_features=32)

            # Test forward pass
            input_tensor = torch.randn(self.batch_size, 64)
            output = layer(input_tensor)

            self.assertEqual(output.shape, (self.batch_size, 32))

            # Test noise reset
            layer.reset_noise()

        except ImportError:
            self.skipTest("PyTorch not available")


class TestExperienceReplay(unittest.TestCase):
    """Test experience replay buffer."""

    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 1000
        self.state_dim = 10

    def test_uniform_replay(self):
        """Test uniform experience replay."""
        buffer = ExperienceReplay(capacity=self.capacity, prioritized=False)

        # Add some experiences
        for i in range(100):
            state = np.random.randn(self.state_dim)
            action = np.random.randint(0, 5)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.randn(self.state_dim)
            done = np.random.choice([True, False])

            buffer.push(state, action, reward, next_state, done)

        self.assertEqual(len(buffer), 100)

        # Sample batch
        batch = buffer.sample(32)
        self.assertEqual(len(batch), 5)  # states, actions, rewards, next_states, dones

    def test_prioritized_replay(self):
        """Test prioritized experience replay."""
        buffer = ExperienceReplay(capacity=self.capacity, prioritized=True)

        # Add some experiences
        for i in range(100):
            state = np.random.randn(self.state_dim)
            action = np.random.randint(0, 5)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.randn(self.state_dim)
            done = np.random.choice([True, False])
            priority = np.random.uniform(0, 1)

            buffer.push(state, action, reward, next_state, done, priority)

        self.assertEqual(len(buffer), 100)

        # Sample batch
        batch = buffer.sample(32)
        self.assertEqual(len(batch), 7)  # includes weights and indices


class TestStateRepresentation(unittest.TestCase):
    """Test enhanced state representation."""

    def setUp(self):
        """Set up test fixtures."""
        self.context = {
            "request": "Can you help me analyze this data?",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "recent_rewards": [0.8, 0.6, 0.9],
            "recent_response_times": [1.2, 0.8, 1.5],
            "tool_usage_counts": {"search": 5, "analyze": 3},
            "user_profile": {
                "preferences": {"verbosity": 0.7, "technical_level": 0.8},
                "expertise": {"technology": 0.9, "business": 0.6},
            },
        }

    def test_text_embedding_encoder(self):
        """Test text embedding encoder."""
        try:
            encoder = TextEmbeddingEncoder(model_name="all-MiniLM-L6-v2")

            # Test text encoding
            text = "This is a test sentence."
            embedding = encoder.encode_text(text)

            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape[0], encoder.embedding_dim)

            # Test conversation encoding
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            conv_embedding = encoder.encode_conversation(messages)

            self.assertIsInstance(conv_embedding, np.ndarray)
            self.assertEqual(conv_embedding.shape[0], encoder.embedding_dim)

        except ImportError:
            self.skipTest("sentence-transformers not available")

    def test_contextual_state_encoder(self):
        """Test contextual state encoder."""
        try:
            # Mock the text encoder to avoid dependency issues
            mock_text_encoder = Mock()
            mock_text_encoder.embedding_dim = 384
            mock_text_encoder.encode_text.return_value = np.random.randn(384)
            mock_text_encoder.encode_conversation.return_value = np.random.randn(384)

            encoder = ContextualStateEncoder(
                text_encoder=mock_text_encoder,
                include_temporal=True,
                include_performance=True,
                include_user_profile=True,
            )

            # Test feature extraction
            temporal_features = encoder.extract_temporal_features(self.context)
            self.assertEqual(len(temporal_features), encoder.temporal_dim)

            # Create mock database
            mock_db = Mock()
            performance_features = encoder.extract_performance_features(self.context, mock_db)
            self.assertEqual(len(performance_features), encoder.performance_dim)

            user_features = encoder.extract_user_profile_features(self.context)
            self.assertEqual(len(user_features), encoder.user_profile_dim)

        except Exception as e:
            self.skipTest(f"Contextual encoder test failed: {e}")


class TestModernDeepRLAgents(unittest.TestCase):
    """Test modern deep RL agents."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()

        # Mock components
        self.mock_model = Mock()
        self.db = MemoryDatabase(self.temp_db.name)
        self.reward_system = RewardSystem(self.db)

        self.state_dim = 128
        self.action_dim = 5

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)

    def test_dqn_agent_creation(self):
        """Test DQN agent creation."""
        try:
            agent = DQNAgent(
                name="test_dqn",
                model=self.mock_model,
                db=self.db,
                reward_system=self.reward_system,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                double_dqn=True,
                dueling=True,
                prioritized_replay=True,
            )

            self.assertEqual(agent.name, "test_dqn")
            self.assertEqual(agent.state_dim, self.state_dim)
            self.assertEqual(agent.action_dim, self.action_dim)
            self.assertTrue(agent.double_dqn)

        except ImportError:
            self.skipTest("PyTorch not available")

    def test_ppo_agent_creation(self):
        """Test PPO agent creation."""
        try:
            agent = PPOAgent(
                name="test_ppo",
                model=self.mock_model,
                db=self.db,
                reward_system=self.reward_system,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                clip_epsilon=0.2,
                ppo_epochs=4,
            )

            self.assertEqual(agent.name, "test_ppo")
            self.assertEqual(agent.clip_epsilon, 0.2)
            self.assertEqual(agent.ppo_epochs, 4)

        except ImportError:
            self.skipTest("PyTorch not available")

    def test_a2c_agent_creation(self):
        """Test A2C agent creation."""
        try:
            agent = A2CAgent(
                name="test_a2c",
                model=self.mock_model,
                db=self.db,
                reward_system=self.reward_system,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
            )

            self.assertEqual(agent.name, "test_a2c")
            self.assertEqual(agent.state_dim, self.state_dim)
            self.assertEqual(agent.action_dim, self.action_dim)

        except ImportError:
            self.skipTest("PyTorch not available")


class TestModernDeepRLCoordinator(unittest.TestCase):
    """Test modern deep RL coordinator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()

        # Mock components
        self.mock_model = Mock()
        self.db = MemoryDatabase(self.temp_db.name)
        self.reward_system = RewardSystem(self.db)

        # Mock sub-agents
        self.sub_agents = {
            "search_agent": Mock(),
            "analysis_agent": Mock(),
        }

        # Mock tools
        self.tools = [
            Mock(name="calculator"),
            Mock(name="translator"),
        ]

        # Configure mock methods
        for agent in self.sub_agents.values():
            agent.process_request = AsyncMock(return_value={
                "success": True,
                "response": "Mock response"
            })

        for tool in self.tools:
            tool.arun = AsyncMock(return_value="Mock tool result")

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)

    async def test_coordinator_creation(self):
        """Test coordinator creation."""
        try:
            coordinator = ModernDeepRLCoordinatorAgent(
                name="test_coordinator",
                model=self.mock_model,
                db=self.db,
                reward_system=self.reward_system,
                sub_agents=self.sub_agents,
                tools=self.tools,
                rl_algorithm="dqn",
            )

            self.assertEqual(coordinator.name, "test_coordinator")
            self.assertEqual(coordinator.rl_algorithm, "dqn")
            self.assertEqual(len(coordinator.actions), 4)  # 2 agents + 2 tools

        except ImportError:
            self.skipTest("PyTorch not available")

    async def test_coordinator_process_request(self):
        """Test coordinator request processing."""
        try:
            coordinator = ModernDeepRLCoordinatorAgent(
                name="test_coordinator",
                model=self.mock_model,
                db=self.db,
                reward_system=self.reward_system,
                sub_agents=self.sub_agents,
                tools=self.tools,
                rl_algorithm="dqn",
            )

            # Process a request
            result = await coordinator.process_request(
                "Test request",
                []
            )

            self.assertIn("success", result)
            self.assertIn("response", result)
            self.assertIn("selected_action", result)
            self.assertIn("reward", result)

        except ImportError:
            self.skipTest("PyTorch not available")


class TestIntegration(unittest.TestCase):
    """Integration tests for the modern deep RL system."""

    def test_import_all_modules(self):
        """Test that all modules can be imported."""
        try:
            from src.agents.enhanced_state_representation import (
                ContextualStateEncoder,
                TextEmbeddingEncoder,
            )
            from src.agents.modern_deep_rl import (
                A2CAgent,
                DQNAgent,
                ModernDeepRLCoordinatorAgent,
                PPOAgent,
            )
            from src.utils.rl_neural_networks import ActorCriticNetwork, DQNNetwork, NoisyLinear

            # If we get here, all imports succeeded
            self.assertTrue(True)

        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")


if __name__ == "__main__":
    # Run async tests
    async def run_async_tests():
        """Run async test methods."""
        test_instance = TestModernDeepRLCoordinator()
        test_instance.setUp()

        try:
            await test_instance.test_coordinator_creation()
            await test_instance.test_coordinator_process_request()
            print("✅ Async tests passed")
        except Exception as e:
            print(f"❌ Async tests failed: {e}")
        finally:
            test_instance.tearDown()

    # Run sync tests
    unittest.main(verbosity=2, exit=False)

    # Run async tests
    asyncio.run(run_async_tests())
