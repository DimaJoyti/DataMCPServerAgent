"""
Curriculum learning module for reinforcement learning in DataMCPServerAgent.
This module implements automatic curriculum generation and progressive task difficulty.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_anthropic import ChatAnthropic

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase


class Task:
    """Represents a learning task with difficulty and requirements."""

    def __init__(
        self,
        task_id: str,
        description: str,
        difficulty: float,
        prerequisites: List[str] = None,
        success_threshold: float = 0.8,
        max_attempts: int = 10,
    ):
        """Initialize task.
        
        Args:
            task_id: Unique task identifier
            description: Task description
            difficulty: Task difficulty (0.0 to 1.0)
            prerequisites: List of prerequisite task IDs
            success_threshold: Success rate threshold to pass task
            max_attempts: Maximum attempts before moving on
        """
        self.task_id = task_id
        self.description = description
        self.difficulty = difficulty
        self.prerequisites = prerequisites or []
        self.success_threshold = success_threshold
        self.max_attempts = max_attempts

        # Performance tracking
        self.attempts = 0
        self.successes = 0
        self.total_reward = 0.0
        self.completion_times = []
        self.is_mastered = False
        self.last_attempt_time = None

    def add_attempt(self, success: bool, reward: float, completion_time: float):
        """Add attempt result.
        
        Args:
            success: Whether attempt was successful
            reward: Reward received
            completion_time: Time taken to complete
        """
        self.attempts += 1
        if success:
            self.successes += 1
        self.total_reward += reward
        self.completion_times.append(completion_time)
        self.last_attempt_time = time.time()

        # Check if task is mastered
        if self.attempts >= 3:  # Need at least 3 attempts
            success_rate = self.successes / self.attempts
            if success_rate >= self.success_threshold:
                self.is_mastered = True

    def get_success_rate(self) -> float:
        """Get current success rate."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def get_average_reward(self) -> float:
        """Get average reward."""
        if self.attempts == 0:
            return 0.0
        return self.total_reward / self.attempts

    def get_average_completion_time(self) -> float:
        """Get average completion time."""
        if not self.completion_times:
            return 0.0
        return np.mean(self.completion_times)

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return not self.is_mastered and self.attempts < self.max_attempts


class CurriculumGenerator:
    """Generates curriculum based on agent performance."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        difficulty_increment: float = 0.1,
        mastery_threshold: float = 0.8,
    ):
        """Initialize curriculum generator.
        
        Args:
            model: Language model
            db: Memory database
            difficulty_increment: How much to increase difficulty
            mastery_threshold: Threshold for task mastery
        """
        self.model = model
        self.db = db
        self.difficulty_increment = difficulty_increment
        self.mastery_threshold = mastery_threshold

        # Task templates for different categories
        self.task_templates = {
            "search": [
                "Search for basic information about {topic}",
                "Find specific details about {topic} with constraints",
                "Perform complex multi-step search for {topic}",
                "Search and synthesize information from multiple sources about {topic}",
            ],
            "analysis": [
                "Analyze simple data about {topic}",
                "Perform statistical analysis on {topic}",
                "Conduct comparative analysis of {topic}",
                "Perform advanced predictive analysis on {topic}",
            ],
            "creation": [
                "Create a simple summary of {topic}",
                "Generate detailed report on {topic}",
                "Create interactive content about {topic}",
                "Develop comprehensive multimedia presentation on {topic}",
            ],
            "problem_solving": [
                "Solve a basic problem related to {topic}",
                "Solve multi-step problem involving {topic}",
                "Solve complex optimization problem for {topic}",
                "Solve novel problem requiring creative thinking about {topic}",
            ],
        }

        # Topics for task generation
        self.topics = [
            "technology", "science", "business", "health", "education",
            "environment", "finance", "marketing", "data analysis", "AI"
        ]

    async def generate_initial_curriculum(self) -> List[Task]:
        """Generate initial curriculum with basic tasks.
        
        Returns:
            List of initial tasks
        """
        tasks = []
        task_id = 0

        # Generate basic tasks for each category
        for category, templates in self.task_templates.items():
            for i, template in enumerate(templates):
                topic = np.random.choice(self.topics)
                description = template.format(topic=topic)
                difficulty = (i + 1) * 0.2  # Increasing difficulty

                task = Task(
                    task_id=f"{category}_{task_id}",
                    description=description,
                    difficulty=difficulty,
                    prerequisites=[f"{category}_{task_id-1}"] if i > 0 else [],
                )
                tasks.append(task)
                task_id += 1

        return tasks

    async def generate_adaptive_task(
        self,
        agent_performance: Dict[str, float],
        completed_tasks: List[Task],
        current_difficulty: float,
    ) -> Task:
        """Generate adaptive task based on agent performance.
        
        Args:
            agent_performance: Agent performance metrics
            completed_tasks: List of completed tasks
            current_difficulty: Current difficulty level
            
        Returns:
            Generated adaptive task
        """
        # Analyze performance to determine next task
        avg_success_rate = agent_performance.get("success_rate", 0.5)
        avg_completion_time = agent_performance.get("avg_completion_time", 1.0)

        # Adjust difficulty based on performance
        if avg_success_rate > 0.9:
            # Too easy, increase difficulty
            new_difficulty = min(1.0, current_difficulty + self.difficulty_increment)
        elif avg_success_rate < 0.5:
            # Too hard, decrease difficulty
            new_difficulty = max(0.1, current_difficulty - self.difficulty_increment)
        else:
            # Just right, slight increase
            new_difficulty = min(1.0, current_difficulty + self.difficulty_increment * 0.5)

        # Determine task category based on weakest area
        category_performance = {}
        for task in completed_tasks:
            category = task.task_id.split("_")[0]
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(task.get_success_rate())

        # Find weakest category
        weakest_category = "search"  # Default
        lowest_performance = 1.0

        for category, performances in category_performance.items():
            avg_perf = np.mean(performances) if performances else 0.0
            if avg_perf < lowest_performance:
                lowest_performance = avg_perf
                weakest_category = category

        # Generate task for weakest category
        templates = self.task_templates.get(weakest_category, self.task_templates["search"])
        difficulty_level = int(new_difficulty * (len(templates) - 1))
        template = templates[min(difficulty_level, len(templates) - 1)]

        topic = np.random.choice(self.topics)
        description = template.format(topic=topic)

        # Find prerequisites
        prerequisites = []
        for task in completed_tasks:
            if (task.task_id.startswith(weakest_category) and
                task.difficulty < new_difficulty and
                task.is_mastered):
                prerequisites.append(task.task_id)

        task_id = f"{weakest_category}_adaptive_{int(time.time())}"

        return Task(
            task_id=task_id,
            description=description,
            difficulty=new_difficulty,
            prerequisites=prerequisites[-2:] if len(prerequisites) > 2 else prerequisites,
        )

    async def generate_challenge_task(
        self,
        mastered_tasks: List[Task],
        agent_strengths: List[str],
    ) -> Task:
        """Generate challenging task that combines multiple skills.
        
        Args:
            mastered_tasks: List of mastered tasks
            agent_strengths: List of agent's strongest areas
            
        Returns:
            Generated challenge task
        """
        # Combine multiple categories for challenge
        categories = list(set(task.task_id.split("_")[0] for task in mastered_tasks))
        selected_categories = np.random.choice(
            categories,
            size=min(3, len(categories)),
            replace=False
        ).tolist()

        # Create complex task description
        topic = np.random.choice(self.topics)

        task_description = f"Complete a comprehensive project on {topic} that involves: "
        task_parts = []

        for category in selected_categories:
            if category == "search":
                task_parts.append("researching and gathering information")
            elif category == "analysis":
                task_parts.append("analyzing and interpreting data")
            elif category == "creation":
                task_parts.append("creating deliverables and presentations")
            elif category == "problem_solving":
                task_parts.append("solving complex problems")

        task_description += ", ".join(task_parts) + "."

        # High difficulty for challenge tasks
        difficulty = 0.9

        # Prerequisites are the mastered tasks from selected categories
        prerequisites = [
            task.task_id for task in mastered_tasks
            if task.task_id.split("_")[0] in selected_categories and task.is_mastered
        ]

        task_id = f"challenge_{int(time.time())}"

        return Task(
            task_id=task_id,
            description=task_description,
            difficulty=difficulty,
            prerequisites=prerequisites,
            success_threshold=0.9,  # Higher threshold for challenges
            max_attempts=15,  # More attempts allowed
        )


class CurriculumLearningAgent:
    """Agent that learns through curriculum learning."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        base_agent: Any,
        curriculum_generator: CurriculumGenerator,
    ):
        """Initialize curriculum learning agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            base_agent: Base RL agent to train
            curriculum_generator: Curriculum generator
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.base_agent = base_agent
        self.curriculum_generator = curriculum_generator

        # Curriculum state
        self.current_tasks = []
        self.completed_tasks = []
        self.current_task_index = 0
        self.curriculum_stage = "initial"  # initial, adaptive, challenge

        # Performance tracking
        self.performance_history = []
        self.learning_progress = {}

    async def initialize_curriculum(self):
        """Initialize the curriculum with basic tasks."""
        self.current_tasks = await self.curriculum_generator.generate_initial_curriculum()
        self.current_task_index = 0
        self.curriculum_stage = "initial"

        print(f"📚 Initialized curriculum with {len(self.current_tasks)} tasks")

    def get_current_task(self) -> Optional[Task]:
        """Get the current task to work on.
        
        Returns:
            Current task or None if curriculum is complete
        """
        if self.current_task_index >= len(self.current_tasks):
            return None

        task = self.current_tasks[self.current_task_index]

        # Check prerequisites
        for prereq_id in task.prerequisites:
            prereq_task = next(
                (t for t in self.completed_tasks if t.task_id == prereq_id),
                None
            )
            if not prereq_task or not prereq_task.is_mastered:
                # Find next task with satisfied prerequisites
                for i in range(self.current_task_index + 1, len(self.current_tasks)):
                    candidate = self.current_tasks[i]
                    if all(
                        any(t.task_id == prereq and t.is_mastered for t in self.completed_tasks)
                        for prereq in candidate.prerequisites
                    ):
                        self.current_task_index = i
                        return candidate
                return None

        return task

    async def process_task_request(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a task request.
        
        Args:
            task: Task to process
            context: Additional context
            
        Returns:
            Task processing result
        """
        start_time = time.time()

        # Use base agent to process the task
        result = await self.base_agent.process_request(
            task.description,
            context.get("history", [])
        )

        end_time = time.time()
        completion_time = end_time - start_time

        # Evaluate task performance
        success = result.get("success", False)
        reward = result.get("reward", 0.0)

        # Add attempt to task
        task.add_attempt(success, reward, completion_time)

        # Update performance tracking
        self.performance_history.append({
            "task_id": task.task_id,
            "success": success,
            "reward": reward,
            "completion_time": completion_time,
            "difficulty": task.difficulty,
            "timestamp": time.time(),
        })

        # Check if task is completed
        if task.is_mastered or not task.should_retry():
            self.completed_tasks.append(task)
            self.current_task_index += 1

            if task.is_mastered:
                print(f"✅ Mastered task: {task.task_id} (Success rate: {task.get_success_rate():.2f})")
            else:
                print(f"⏭️ Moving on from task: {task.task_id} (Max attempts reached)")

        return {
            "success": success,
            "response": result.get("response", ""),
            "task_id": task.task_id,
            "task_mastered": task.is_mastered,
            "attempts": task.attempts,
            "success_rate": task.get_success_rate(),
            "reward": reward,
            "completion_time": completion_time,
        }

    async def advance_curriculum(self):
        """Advance to next stage of curriculum."""
        if self.curriculum_stage == "initial" and self.current_task_index >= len(self.current_tasks):
            # Move to adaptive stage
            self.curriculum_stage = "adaptive"
            await self._generate_adaptive_tasks()

        elif self.curriculum_stage == "adaptive":
            # Check if ready for challenges
            mastered_count = sum(1 for task in self.completed_tasks if task.is_mastered)
            if mastered_count >= 10:  # Threshold for challenge stage
                self.curriculum_stage = "challenge"
                await self._generate_challenge_tasks()

    async def _generate_adaptive_tasks(self):
        """Generate adaptive tasks based on performance."""
        # Calculate performance metrics
        recent_performance = self.performance_history[-20:] if len(self.performance_history) >= 20 else self.performance_history

        if not recent_performance:
            return

        agent_performance = {
            "success_rate": np.mean([p["success"] for p in recent_performance]),
            "avg_reward": np.mean([p["reward"] for p in recent_performance]),
            "avg_completion_time": np.mean([p["completion_time"] for p in recent_performance]),
        }

        current_difficulty = np.mean([p["difficulty"] for p in recent_performance])

        # Generate new adaptive tasks
        new_tasks = []
        for _ in range(5):  # Generate 5 adaptive tasks
            task = await self.curriculum_generator.generate_adaptive_task(
                agent_performance, self.completed_tasks, current_difficulty
            )
            new_tasks.append(task)

        self.current_tasks.extend(new_tasks)
        print(f"📈 Generated {len(new_tasks)} adaptive tasks")

    async def _generate_challenge_tasks(self):
        """Generate challenge tasks."""
        mastered_tasks = [task for task in self.completed_tasks if task.is_mastered]

        # Identify agent strengths
        category_performance = {}
        for task in mastered_tasks:
            category = task.task_id.split("_")[0]
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(task.get_success_rate())

        agent_strengths = [
            category for category, performances in category_performance.items()
            if np.mean(performances) > 0.8
        ]

        # Generate challenge tasks
        new_tasks = []
        for _ in range(3):  # Generate 3 challenge tasks
            task = await self.curriculum_generator.generate_challenge_task(
                mastered_tasks, agent_strengths
            )
            new_tasks.append(task)

        self.current_tasks.extend(new_tasks)
        print(f"🏆 Generated {len(new_tasks)} challenge tasks")

    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics.
        
        Returns:
            Learning progress information
        """
        total_tasks = len(self.completed_tasks) + (len(self.current_tasks) - self.current_task_index)
        completed_count = len(self.completed_tasks)
        mastered_count = sum(1 for task in self.completed_tasks if task.is_mastered)

        # Calculate category-wise progress
        category_progress = {}
        for task in self.completed_tasks:
            category = task.task_id.split("_")[0]
            if category not in category_progress:
                category_progress[category] = {"total": 0, "mastered": 0}
            category_progress[category]["total"] += 1
            if task.is_mastered:
                category_progress[category]["mastered"] += 1

        # Calculate learning velocity
        if len(self.performance_history) >= 10:
            recent_success_rate = np.mean([p["success"] for p in self.performance_history[-10:]])
            early_success_rate = np.mean([p["success"] for p in self.performance_history[:10]])
            learning_velocity = recent_success_rate - early_success_rate
        else:
            learning_velocity = 0.0

        return {
            "curriculum_stage": self.curriculum_stage,
            "total_tasks": total_tasks,
            "completed_tasks": completed_count,
            "mastered_tasks": mastered_count,
            "completion_rate": completed_count / total_tasks if total_tasks > 0 else 0.0,
            "mastery_rate": mastered_count / completed_count if completed_count > 0 else 0.0,
            "category_progress": category_progress,
            "learning_velocity": learning_velocity,
            "current_task": self.get_current_task().task_id if self.get_current_task() else None,
        }


# Factory function to create curriculum learning agent
async def create_curriculum_learning_agent(
    model: ChatAnthropic,
    db: MemoryDatabase,
    base_agent: Any,
    difficulty_increment: float = 0.1,
) -> CurriculumLearningAgent:
    """Create a curriculum learning agent.
    
    Args:
        model: Language model to use
        db: Memory database for persistence
        base_agent: Base RL agent to train
        difficulty_increment: Difficulty increment for curriculum
        
    Returns:
        Curriculum learning agent
    """
    # Create reward system and curriculum generator
    reward_system = RewardSystem(db)
    curriculum_generator = CurriculumGenerator(
        model=model,
        db=db,
        difficulty_increment=difficulty_increment,
    )

    # Create curriculum learning agent
    curriculum_agent = CurriculumLearningAgent(
        name="curriculum_learning_agent",
        model=model,
        db=db,
        reward_system=reward_system,
        base_agent=base_agent,
        curriculum_generator=curriculum_generator,
    )

    # Initialize curriculum
    await curriculum_agent.initialize_curriculum()

    return curriculum_agent
