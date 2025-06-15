"""
Explainable reinforcement learning module for DataMCPServerAgent.
This module provides interpretability and explanation capabilities for RL decisions.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase


class ActionExplanation:
    """Represents an explanation for an RL agent's action."""

    def __init__(
        self,
        action: int,
        confidence: float,
        reasoning: str,
        contributing_factors: Dict[str, float],
        alternative_actions: List[Dict[str, Any]],
        risk_assessment: Dict[str, float],
    ):
        """Initialize action explanation.
        
        Args:
            action: Selected action
            confidence: Confidence in action selection (0.0 to 1.0)
            reasoning: Natural language reasoning
            contributing_factors: Factors that influenced the decision
            alternative_actions: Alternative actions considered
            risk_assessment: Risk assessment for the action
        """
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.contributing_factors = contributing_factors
        self.alternative_actions = alternative_actions
        self.risk_assessment = risk_assessment
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "contributing_factors": self.contributing_factors,
            "alternative_actions": self.alternative_actions,
            "risk_assessment": self.risk_assessment,
            "timestamp": self.timestamp,
        }

    def get_summary(self) -> str:
        """Get a summary of the explanation.
        
        Returns:
            Summary string
        """
        top_factors = sorted(
            self.contributing_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        factor_text = ", ".join([f"{name} ({value:.2f})" for name, value in top_factors])

        return (
            f"Action {self.action} selected with {self.confidence:.1%} confidence. "
            f"Key factors: {factor_text}. {self.reasoning}"
        )


class FeatureImportanceAnalyzer:
    """Analyzes feature importance for RL decisions."""

    def __init__(self, feature_names: List[str]):
        """Initialize feature importance analyzer.
        
        Args:
            feature_names: Names of input features
        """
        self.feature_names = feature_names
        self.importance_history = []

    def compute_feature_importance(
        self,
        model: nn.Module,
        state: torch.Tensor,
        action: int,
        method: str = "gradient"
    ) -> Dict[str, float]:
        """Compute feature importance for a decision.
        
        Args:
            model: RL model
            state: Input state
            action: Selected action
            method: Importance computation method
            
        Returns:
            Feature importance scores
        """
        if method == "gradient":
            return self._gradient_based_importance(model, state, action)
        elif method == "permutation":
            return self._permutation_importance(model, state, action)
        elif method == "integrated_gradients":
            return self._integrated_gradients_importance(model, state, action)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _gradient_based_importance(
        self,
        model: nn.Module,
        state: torch.Tensor,
        action: int
    ) -> Dict[str, float]:
        """Compute gradient-based feature importance.
        
        Args:
            model: RL model
            state: Input state
            action: Selected action
            
        Returns:
            Feature importance scores
        """
        state.requires_grad_(True)

        # Forward pass
        if hasattr(model, 'get_q_values'):
            q_values = model.get_q_values(state.unsqueeze(0))
        else:
            q_values = model(state.unsqueeze(0))

        # Get Q-value for selected action
        action_value = q_values[0, action]

        # Backward pass
        action_value.backward()

        # Get gradients
        gradients = state.grad.abs().detach().numpy()

        # Normalize gradients
        if gradients.sum() > 0:
            gradients = gradients / gradients.sum()

        # Map to feature names
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            if i < len(gradients):
                importance_dict[name] = float(gradients[i])
            else:
                importance_dict[name] = 0.0

        return importance_dict

    def _permutation_importance(
        self,
        model: nn.Module,
        state: torch.Tensor,
        action: int
    ) -> Dict[str, float]:
        """Compute permutation-based feature importance.
        
        Args:
            model: RL model
            state: Input state
            action: Selected action
            
        Returns:
            Feature importance scores
        """
        with torch.no_grad():
            # Get baseline prediction
            if hasattr(model, 'get_q_values'):
                baseline_q = model.get_q_values(state.unsqueeze(0))
            else:
                baseline_q = model(state.unsqueeze(0))

            baseline_value = baseline_q[0, action].item()

            importance_scores = {}

            for i, feature_name in enumerate(self.feature_names):
                if i >= len(state):
                    importance_scores[feature_name] = 0.0
                    continue

                # Permute feature
                perturbed_state = state.clone()
                perturbed_state[i] = torch.randn_like(perturbed_state[i])

                # Get prediction with perturbed feature
                if hasattr(model, 'get_q_values'):
                    perturbed_q = model.get_q_values(perturbed_state.unsqueeze(0))
                else:
                    perturbed_q = model(perturbed_state.unsqueeze(0))

                perturbed_value = perturbed_q[0, action].item()

                # Importance is the change in prediction
                importance = abs(baseline_value - perturbed_value)
                importance_scores[feature_name] = importance

        # Normalize scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                name: score / total_importance
                for name, score in importance_scores.items()
            }

        return importance_scores

    def _integrated_gradients_importance(
        self,
        model: nn.Module,
        state: torch.Tensor,
        action: int,
        steps: int = 50
    ) -> Dict[str, float]:
        """Compute integrated gradients importance.
        
        Args:
            model: RL model
            state: Input state
            action: Selected action
            steps: Number of integration steps
            
        Returns:
            Feature importance scores
        """
        # Baseline (zero state)
        baseline = torch.zeros_like(state)

        # Compute integrated gradients
        integrated_grads = torch.zeros_like(state)

        for step in range(steps):
            # Interpolate between baseline and input
            alpha = step / steps
            interpolated = baseline + alpha * (state - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            if hasattr(model, 'get_q_values'):
                q_values = model.get_q_values(interpolated.unsqueeze(0))
            else:
                q_values = model(interpolated.unsqueeze(0))

            action_value = q_values[0, action]

            # Backward pass
            action_value.backward()

            # Accumulate gradients
            integrated_grads += interpolated.grad

            # Clear gradients
            interpolated.grad.zero_()

        # Average gradients and multiply by input difference
        integrated_grads = integrated_grads / steps
        attributions = integrated_grads * (state - baseline)

        # Convert to importance scores
        attributions = attributions.abs().detach().numpy()

        # Normalize
        if attributions.sum() > 0:
            attributions = attributions / attributions.sum()

        # Map to feature names
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            if i < len(attributions):
                importance_dict[name] = float(attributions[i])
            else:
                importance_dict[name] = 0.0

        return importance_dict


class DecisionTreeExplainer:
    """Explains RL decisions using decision tree approximation."""

    def __init__(self, max_depth: int = 5):
        """Initialize decision tree explainer.
        
        Args:
            max_depth: Maximum depth of explanation tree
        """
        self.max_depth = max_depth
        self.explanation_trees = {}

    def build_explanation_tree(
        self,
        model: nn.Module,
        state_samples: List[torch.Tensor],
        action_samples: List[int],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Build decision tree explanation for model behavior.
        
        Args:
            model: RL model
            state_samples: Sample states
            action_samples: Corresponding actions
            feature_names: Names of features
            
        Returns:
            Decision tree explanation
        """
        # This is a simplified implementation
        # In practice, you'd use sklearn's DecisionTreeClassifier

        if not state_samples or not action_samples:
            return {"error": "No samples provided"}

        # Convert to numpy arrays
        X = torch.stack(state_samples).detach().numpy()
        y = np.array(action_samples)

        # Build simple decision tree explanation
        tree_explanation = self._build_simple_tree(X, y, feature_names, depth=0)

        return tree_explanation

    def _build_simple_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        depth: int
    ) -> Dict[str, Any]:
        """Build simple decision tree recursively.
        
        Args:
            X: Feature matrix
            y: Target actions
            feature_names: Feature names
            depth: Current depth
            
        Returns:
            Tree node
        """
        # Base cases
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) < 2:
            most_common_action = np.bincount(y).argmax()
            return {
                "type": "leaf",
                "action": int(most_common_action),
                "samples": len(X),
                "confidence": np.mean(y == most_common_action),
            }

        # Find best split
        best_feature = 0
        best_threshold = 0.0
        best_score = 0.0

        for feature_idx in range(min(len(feature_names), X.shape[1])):
            feature_values = X[:, feature_idx]
            thresholds = np.percentile(feature_values, [25, 50, 75])

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain (simplified)
                left_purity = self._calculate_purity(y[left_mask])
                right_purity = self._calculate_purity(y[right_mask])

                weighted_purity = (
                    np.sum(left_mask) / len(y) * left_purity +
                    np.sum(right_mask) / len(y) * right_purity
                )

                if weighted_purity > best_score:
                    best_score = weighted_purity
                    best_feature = feature_idx
                    best_threshold = threshold

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Build child nodes
        left_child = self._build_simple_tree(
            X[left_mask], y[left_mask], feature_names, depth + 1
        )
        right_child = self._build_simple_tree(
            X[right_mask], y[right_mask], feature_names, depth + 1
        )

        return {
            "type": "split",
            "feature": feature_names[best_feature] if best_feature < len(feature_names) else f"feature_{best_feature}",
            "threshold": float(best_threshold),
            "left": left_child,
            "right": right_child,
            "samples": len(X),
        }

    def _calculate_purity(self, y: np.ndarray) -> float:
        """Calculate purity of a set of labels.
        
        Args:
            y: Labels
            
        Returns:
            Purity score
        """
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # Gini impurity
        gini = 1.0 - np.sum(probabilities ** 2)
        return 1.0 - gini  # Convert to purity


class ExplainableRLAgent:
    """RL agent with explainability capabilities."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: MemoryDatabase,
        reward_system: RewardSystem,
        base_agent: Any,
        feature_names: Optional[List[str]] = None,
        explanation_methods: List[str] = ["gradient", "permutation"],
    ):
        """Initialize explainable RL agent.
        
        Args:
            name: Agent name
            model: Language model
            db: Memory database
            reward_system: Reward system
            base_agent: Base RL agent
            feature_names: Names of input features
            explanation_methods: Methods for generating explanations
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.base_agent = base_agent
        self.explanation_methods = explanation_methods

        # Feature names
        if feature_names is None:
            state_dim = getattr(base_agent, 'state_dim', 128)
            self.feature_names = [f"feature_{i}" for i in range(state_dim)]
        else:
            self.feature_names = feature_names

        # Explanation components
        self.importance_analyzer = FeatureImportanceAnalyzer(self.feature_names)
        self.tree_explainer = DecisionTreeExplainer()

        # Explanation history
        self.explanation_history = []
        self.decision_samples = []

    async def select_action_with_explanation(
        self,
        state: np.ndarray,
        context: Dict[str, Any],
        training: bool = True
    ) -> Tuple[int, ActionExplanation]:
        """Select action and generate explanation.
        
        Args:
            state: Current state
            context: Additional context
            training: Whether in training mode
            
        Returns:
            Tuple of (action, explanation)
        """
        # Get action from base agent
        if hasattr(self.base_agent, 'select_action'):
            action = self.base_agent.select_action(state, training)
        else:
            action = np.random.randint(0, 5)  # Fallback

        # Generate explanation
        explanation = await self._generate_explanation(state, action, context)

        # Store for future analysis
        self.explanation_history.append(explanation)
        self.decision_samples.append({
            "state": torch.FloatTensor(state),
            "action": action,
            "context": context,
        })

        # Keep history bounded
        if len(self.explanation_history) > 1000:
            self.explanation_history.pop(0)
            self.decision_samples.pop(0)

        return action, explanation

    async def _generate_explanation(
        self,
        state: np.ndarray,
        action: int,
        context: Dict[str, Any]
    ) -> ActionExplanation:
        """Generate explanation for an action.
        
        Args:
            state: Current state
            action: Selected action
            context: Additional context
            
        Returns:
            Action explanation
        """
        state_tensor = torch.FloatTensor(state)

        # Compute feature importance
        contributing_factors = {}

        if hasattr(self.base_agent, 'q_network'):
            model = self.base_agent.q_network

            for method in self.explanation_methods:
                try:
                    importance = self.importance_analyzer.compute_feature_importance(
                        model, state_tensor, action, method
                    )

                    # Merge importance scores
                    for feature, score in importance.items():
                        if feature not in contributing_factors:
                            contributing_factors[feature] = 0.0
                        contributing_factors[feature] += score
                except Exception as e:
                    print(f"Warning: Failed to compute {method} importance: {e}")

        # Normalize contributing factors
        total_importance = sum(abs(score) for score in contributing_factors.values())
        if total_importance > 0:
            contributing_factors = {
                name: score / total_importance
                for name, score in contributing_factors.items()
            }

        # Generate natural language reasoning
        reasoning = await self._generate_natural_language_explanation(
            state, action, contributing_factors, context
        )

        # Assess alternative actions
        alternative_actions = await self._assess_alternative_actions(state, action)

        # Risk assessment
        risk_assessment = self._assess_action_risk(state, action, context)

        # Compute confidence
        confidence = self._compute_action_confidence(state, action, contributing_factors)

        return ActionExplanation(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors=contributing_factors,
            alternative_actions=alternative_actions,
            risk_assessment=risk_assessment,
        )

    async def _generate_natural_language_explanation(
        self,
        state: np.ndarray,
        action: int,
        contributing_factors: Dict[str, float],
        context: Dict[str, Any]
    ) -> str:
        """Generate natural language explanation.
        
        Args:
            state: Current state
            action: Selected action
            contributing_factors: Feature importance scores
            context: Additional context
            
        Returns:
            Natural language explanation
        """
        # Get top contributing factors
        top_factors = sorted(
            contributing_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        # Create explanation prompt
        factor_descriptions = []
        for factor_name, importance in top_factors:
            if importance > 0.1:  # Only include significant factors
                factor_descriptions.append(f"{factor_name} (importance: {importance:.2f})")

        action_names = {
            0: "search for information",
            1: "analyze data",
            2: "create content",
            3: "communicate with user",
            4: "wait and observe"
        }

        action_description = action_names.get(action, f"action {action}")

        prompt = f"""
        Explain why the AI agent chose to {action_description}.
        
        Key contributing factors: {', '.join(factor_descriptions)}
        Context: {context.get('request', 'No specific request')}
        
        Provide a brief, clear explanation in 1-2 sentences.
        """

        try:
            response = await self.model.ainvoke([
                SystemMessage(content="You are an AI explainer. Provide clear, concise explanations for AI decisions."),
                HumanMessage(content=prompt)
            ])

            return response.content.strip()
        except Exception:
            # Fallback explanation
            return f"Selected {action_description} based on current state analysis and context."

    async def _assess_alternative_actions(
        self,
        state: np.ndarray,
        selected_action: int
    ) -> List[Dict[str, Any]]:
        """Assess alternative actions that could have been taken.
        
        Args:
            state: Current state
            selected_action: Action that was selected
            
        Returns:
            List of alternative actions with their assessments
        """
        alternatives = []

        if hasattr(self.base_agent, 'q_network'):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                if hasattr(self.base_agent.q_network, 'get_q_values'):
                    q_values = self.base_agent.q_network.get_q_values(state_tensor)
                else:
                    q_values = self.base_agent.q_network(state_tensor)

                q_values = q_values.squeeze().numpy()

                # Get top 3 alternative actions
                action_indices = np.argsort(q_values)[::-1]

                for i, action_idx in enumerate(action_indices[:4]):  # Top 4 including selected
                    if action_idx == selected_action:
                        continue

                    alternatives.append({
                        "action": int(action_idx),
                        "q_value": float(q_values[action_idx]),
                        "rank": i + 1,
                        "probability": float(np.exp(q_values[action_idx]) / np.sum(np.exp(q_values))),
                    })

                    if len(alternatives) >= 3:
                        break

        return alternatives

    def _assess_action_risk(
        self,
        state: np.ndarray,
        action: int,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess risk associated with the action.
        
        Args:
            state: Current state
            action: Selected action
            context: Additional context
            
        Returns:
            Risk assessment
        """
        # Simple risk assessment based on action type and context
        base_risks = {
            0: 0.1,  # Search - low risk
            1: 0.3,  # Analyze - medium risk
            2: 0.5,  # Create - higher risk
            3: 0.2,  # Communicate - low-medium risk
            4: 0.0,  # Wait - no risk
        }

        base_risk = base_risks.get(action, 0.5)

        # Adjust risk based on context
        risk_factors = {
            "uncertainty": 0.0,
            "complexity": 0.0,
            "time_pressure": 0.0,
            "resource_usage": 0.0,
        }

        # Estimate uncertainty from state variance
        state_variance = np.var(state)
        risk_factors["uncertainty"] = min(1.0, state_variance / 10.0)

        # Estimate complexity from state magnitude
        state_magnitude = np.linalg.norm(state)
        risk_factors["complexity"] = min(1.0, state_magnitude / 100.0)

        # Time pressure from context
        if context.get("urgent", False):
            risk_factors["time_pressure"] = 0.8

        # Resource usage risk
        if action in [1, 2]:  # Analyze, Create
            risk_factors["resource_usage"] = 0.6

        # Overall risk
        overall_risk = base_risk + 0.3 * np.mean(list(risk_factors.values()))
        overall_risk = min(1.0, overall_risk)

        risk_factors["overall"] = overall_risk

        return risk_factors

    def _compute_action_confidence(
        self,
        state: np.ndarray,
        action: int,
        contributing_factors: Dict[str, float]
    ) -> float:
        """Compute confidence in action selection.
        
        Args:
            state: Current state
            action: Selected action
            contributing_factors: Feature importance scores
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from Q-values if available
        base_confidence = 0.5

        if hasattr(self.base_agent, 'q_network'):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                if hasattr(self.base_agent.q_network, 'get_q_values'):
                    q_values = self.base_agent.q_network.get_q_values(state_tensor)
                else:
                    q_values = self.base_agent.q_network(state_tensor)

                q_values = q_values.squeeze().numpy()

                # Softmax to get probabilities
                probs = np.exp(q_values) / np.sum(np.exp(q_values))
                base_confidence = probs[action]

        # Adjust confidence based on feature importance concentration
        importance_values = list(contributing_factors.values())
        if importance_values:
            # Higher concentration of importance = higher confidence
            importance_concentration = np.max(importance_values)
            confidence_adjustment = importance_concentration * 0.3
        else:
            confidence_adjustment = 0.0

        final_confidence = min(1.0, base_confidence + confidence_adjustment)

        return final_confidence

    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get statistics about explanations generated.
        
        Returns:
            Explanation statistics
        """
        if not self.explanation_history:
            return {"error": "No explanations generated yet"}

        # Average confidence
        avg_confidence = np.mean([exp.confidence for exp in self.explanation_history])

        # Most important features
        all_factors = {}
        for exp in self.explanation_history:
            for factor, importance in exp.contributing_factors.items():
                if factor not in all_factors:
                    all_factors[factor] = []
                all_factors[factor].append(abs(importance))

        avg_importance = {
            factor: np.mean(importances)
            for factor, importances in all_factors.items()
        }

        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        # Risk distribution
        risk_levels = [exp.risk_assessment.get("overall", 0.5) for exp in self.explanation_history]
        avg_risk = np.mean(risk_levels)

        return {
            "total_explanations": len(self.explanation_history),
            "avg_confidence": avg_confidence,
            "avg_risk": avg_risk,
            "top_important_features": top_features,
            "explanation_methods": self.explanation_methods,
        }


# Factory function to create explainable RL agent
async def create_explainable_rl_agent(
    model: ChatAnthropic,
    db: MemoryDatabase,
    base_agent: Any,
    feature_names: Optional[List[str]] = None,
    explanation_methods: List[str] = ["gradient", "permutation"],
) -> ExplainableRLAgent:
    """Create explainable RL agent.
    
    Args:
        model: Language model
        db: Memory database
        base_agent: Base RL agent to make explainable
        feature_names: Names of input features
        explanation_methods: Methods for generating explanations
        
    Returns:
        Explainable RL agent
    """
    # Create reward system
    reward_system = RewardSystem(db)

    # Create explainable RL agent
    explainable_agent = ExplainableRLAgent(
        name="explainable_rl_agent",
        model=model,
        db=db,
        reward_system=reward_system,
        base_agent=base_agent,
        feature_names=feature_names,
        explanation_methods=explanation_methods,
    )

    return explainable_agent
