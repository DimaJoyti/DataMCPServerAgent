"""
Advanced error analysis module for DataMCPServerAgent.
This module provides sophisticated error analysis techniques including:
- Error clustering and pattern recognition
- Root cause analysis
- Predictive error detection
- Enhanced error classification with NLP
- Error correlation analysis
"""

import json
import logging
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import classify_error
from src.utils.error_recovery import ErrorRecoverySystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ErrorCluster:
    """Represents a cluster of similar errors."""

    def __init__(self, cluster_id: int, error_type: str, representative_error: str):
        """Initialize the error cluster.

        Args:
            cluster_id: Unique identifier for the cluster
            error_type: Type of errors in this cluster
            representative_error: Representative error message for this cluster
        """
        self.cluster_id = cluster_id
        self.error_type = error_type
        self.representative_error = representative_error
        self.errors = []
        self.tools = set()
        self.timestamps = []
        self.frequency = 0
        self.last_occurrence = 0

    def add_error(self, error: Dict[str, Any]):
        """Add an error to the cluster.

        Args:
            error: Error information
        """
        self.errors.append(error)
        if "tool_name" in error:
            self.tools.add(error["tool_name"])
        if "timestamp" in error:
            self.timestamps.append(error["timestamp"])
            self.last_occurrence = max(self.timestamps)
        self.frequency = len(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the cluster to a dictionary.

        Returns:
            Dictionary representation of the cluster
        """
        return {
            "cluster_id": self.cluster_id,
            "error_type": self.error_type,
            "representative_error": self.representative_error,
            "tools": list(self.tools),
            "frequency": self.frequency,
            "last_occurrence": self.last_occurrence,
            "first_occurrence": min(self.timestamps) if self.timestamps else 0,
            "error_count": len(self.errors),
        }


class AdvancedErrorAnalysis:
    """Advanced error analysis system."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        tools: List[BaseTool],
        error_recovery: ErrorRecoverySystem,
    ):
        """Initialize the advanced error analysis system.

        Args:
            model: Language model to use
            db: Memory database for persistence
            tools: List of available tools
            error_recovery: Error recovery system
        """
        self.model = model
        self.db = db
        self.tools = tools
        self.error_recovery = error_recovery
        self.tool_map = {tool.name: tool for tool in tools}
        self.error_clusters = []
        self.error_correlations = {}
        self.root_causes = {}
        self.predictive_patterns = []

        # Create the root cause analysis prompt
        self.root_cause_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an advanced error analysis system responsible for identifying root causes of errors.
Your job is to analyze error information and determine the underlying root causes.

For the error cluster provided, you should:
1. Identify the most likely root cause of the errors
2. Determine if there are any common factors across the errors
3. Analyze the error chain to find the original trigger
4. Suggest ways to address the root cause rather than just the symptoms

Respond with a JSON object containing:
- "root_cause": The identified root cause of the errors
- "common_factors": Array of common factors across the errors
- "error_chain": Array describing the chain of events leading to the errors
- "prevention_strategies": Array of strategies to prevent the root cause
"""
                ),
                HumanMessage(
                    content="""
Error cluster:
{error_cluster}

Tool information:
{tool_information}

Context:
{context}

Analyze this error cluster and identify the root cause.
"""
                ),
            ]
        )

        # Create the error correlation prompt
        self.correlation_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an advanced error analysis system responsible for identifying correlations between errors.
Your job is to analyze error patterns and determine if there are correlations between different types of errors.

For the error patterns provided, you should:
1. Identify correlations between different error types
2. Determine if certain errors tend to precede or follow other errors
3. Analyze if errors in one tool correlate with errors in another tool
4. Identify any cascading error patterns

Respond with a JSON object containing:
- "correlations": Array of identified correlations between error types
- "sequential_patterns": Array of error sequences that occur frequently
- "tool_correlations": Object mapping tools to correlated tools based on errors
- "cascading_patterns": Array of identified cascading error patterns
"""
                ),
                HumanMessage(
                    content="""
Error patterns:
{error_patterns}

Tool usage patterns:
{tool_usage_patterns}

Analyze these patterns and identify correlations between errors.
"""
                ),
            ]
        )

        # Create the predictive error detection prompt
        self.predictive_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an advanced error prediction system responsible for identifying potential future errors.
Your job is to analyze error patterns and predict potential future errors before they occur.

For the error history and current state provided, you should:
1. Identify patterns that might indicate upcoming errors
2. Predict potential error types that might occur soon
3. Suggest preemptive actions to prevent predicted errors
4. Assign confidence levels to your predictions

Respond with a JSON object containing:
- "predicted_errors": Array of predicted error types with confidence levels
- "warning_signs": Array of warning signs that indicate potential errors
- "preemptive_actions": Array of actions to take to prevent predicted errors
- "high_risk_tools": Array of tools that are at high risk of errors
"""
                ),
                HumanMessage(
                    content="""
Error history:
{error_history}

Current system state:
{current_state}

Recent tool usage:
{recent_tool_usage}

Predict potential future errors based on this information.
"""
                ),
            ]
        )

    async def cluster_errors(self) -> List[ErrorCluster]:
        """Cluster similar errors based on patterns.

        Returns:
            List of error clusters
        """
        # Get all error executions from the database
        executions = self.db.get_entities_by_prefix("error_recovery", "execution_")

        # Filter failed executions
        failed_executions = [
            e for e in executions if not e.get("success", True) and "error_message" in e
        ]

        if not failed_executions:
            return []

        # Extract error messages
        error_messages = [e["error_message"] for e in failed_executions]

        # Create TF-IDF vectors for error messages
        vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )

        try:
            # Transform error messages to TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform(error_messages)

            # Cluster error messages using DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
            clusters = dbscan.fit_predict(tfidf_matrix)

            # Create error clusters
            error_clusters = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise points
                    continue

                # Get indices of errors in this cluster
                indices = [i for i, c in enumerate(clusters) if c == cluster_id]

                # Get error type (most common in the cluster)
                error_types = [
                    classify_error(
                        Exception(failed_executions[i]["error_message"])
                    ).error_type
                    for i in indices
                ]
                error_type = Counter(error_types).most_common(1)[0][0]

                # Get representative error (closest to cluster centroid)
                centroid = tfidf_matrix[indices].mean(axis=0)
                distances = [
                    np.linalg.norm(tfidf_matrix[i].toarray() - centroid)
                    for i in indices
                ]
                representative_idx = indices[distances.index(min(distances))]
                representative_error = error_messages[representative_idx]

                # Create error cluster
                cluster = ErrorCluster(
                    cluster_id=int(cluster_id),
                    error_type=error_type,
                    representative_error=representative_error,
                )

                # Add errors to cluster
                for i in indices:
                    cluster.add_error(failed_executions[i])

                error_clusters.append(cluster)

            # Save clusters to database
            self.db.save_entity(
                "advanced_error_analysis",
                f"clusters_{int(time.time())}",
                {
                    "clusters": [cluster.to_dict() for cluster in error_clusters],
                    "timestamp": time.time(),
                },
            )

            self.error_clusters = error_clusters
            return error_clusters
        except Exception as e:
            logger.error(f"Error clustering errors: {str(e)}")
            return []

    async def analyze_root_causes(self) -> Dict[int, Dict[str, Any]]:
        """Analyze root causes of error clusters.

        Returns:
            Dictionary mapping cluster IDs to root cause analysis results
        """
        if not self.error_clusters:
            await self.cluster_errors()

        if not self.error_clusters:
            return {}

        root_causes = {}

        for cluster in self.error_clusters:
            # Format cluster information
            cluster_info = json.dumps(cluster.to_dict())

            # Format tool information
            tool_info = []
            for tool_name in cluster.tools:
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    tool_info.append(f"- {tool.name}: {tool.description}")
            tool_information = "\n".join(tool_info)

            # Format context information
            context = f"Error frequency: {cluster.frequency}\n"
            context += f"Affected tools: {', '.join(cluster.tools)}\n"
            context += f"First occurrence: {datetime.fromtimestamp(cluster.to_dict()['first_occurrence']).strftime('%Y-%m-%d %H:%M:%S')}\n"
            context += f"Last occurrence: {datetime.fromtimestamp(cluster.last_occurrence).strftime('%Y-%m-%d %H:%M:%S')}\n"

            # Prepare the input for the prompt
            input_values = {
                "error_cluster": cluster_info,
                "tool_information": tool_information,
                "context": context,
            }

            try:
                # Get the root cause analysis from the model
                messages = self.root_cause_prompt.format_messages(**input_values)
                response = await self.model.ainvoke(messages)

                # Parse the response
                content = response.content
                json_str = (
                    content.split("```json")[1].split("```")[0]
                    if "```json" in content
                    else content
                )
                json_str = json_str.strip()

                # Handle cases where the JSON might be embedded in text
                if not json_str.startswith("{"):
                    start_idx = json_str.find("{")
                    end_idx = json_str.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = json_str[start_idx:end_idx]

                analysis = json.loads(json_str)

                # Add cluster information to the analysis
                analysis["cluster_id"] = cluster.cluster_id
                analysis["error_type"] = cluster.error_type
                analysis["representative_error"] = cluster.representative_error
                analysis["timestamp"] = time.time()

                # Save the analysis to the database
                self.db.save_entity(
                    "advanced_error_analysis",
                    f"root_cause_{cluster.cluster_id}_{int(time.time())}",
                    analysis,
                )

                root_causes[cluster.cluster_id] = analysis
            except Exception as e:
                logger.error(
                    f"Error analyzing root cause for cluster {cluster.cluster_id}: {str(e)}"
                )

                # Create a default analysis
                default_analysis = {
                    "cluster_id": cluster.cluster_id,
                    "error_type": cluster.error_type,
                    "representative_error": cluster.representative_error,
                    "root_cause": "Unknown",
                    "common_factors": ["Error analyzing root cause"],
                    "error_chain": ["Unknown"],
                    "prevention_strategies": ["Implement more robust error handling"],
                    "timestamp": time.time(),
                    "parsing_error": str(e),
                }

                # Save the default analysis to the database
                self.db.save_entity(
                    "advanced_error_analysis",
                    f"root_cause_{cluster.cluster_id}_{int(time.time())}",
                    default_analysis,
                )

                root_causes[cluster.cluster_id] = default_analysis

        self.root_causes = root_causes
        return root_causes

    async def analyze_error_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different errors.

        Returns:
            Correlation analysis results
        """
        # Get all error executions from the database
        executions = self.db.get_entities_by_prefix("error_recovery", "execution_")

        # Filter failed executions
        failed_executions = [
            e for e in executions if not e.get("success", True) and "error_message" in e
        ]

        if not failed_executions:
            return {
                "correlations": [],
                "sequential_patterns": [],
                "tool_correlations": {},
                "cascading_patterns": [],
            }

        # Sort executions by timestamp
        failed_executions.sort(key=lambda e: e.get("timestamp", 0))

        # Format error patterns
        error_patterns = []
        for execution in failed_executions:
            error = classify_error(Exception(execution.get("error_message", "")))
            error_patterns.append(
                {
                    "tool": execution.get("tool_name", "unknown"),
                    "error_type": error.error_type,
                    "timestamp": execution.get("timestamp", 0),
                    "error_message": execution.get("error_message", ""),
                }
            )

        # Format tool usage patterns
        tool_usage = {}
        for execution in executions:
            tool_name = execution.get("tool_name", "unknown")
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "error_types": Counter(),
                }

            tool_usage[tool_name]["total"] += 1

            if execution.get("success", True):
                tool_usage[tool_name]["success"] += 1
            else:
                tool_usage[tool_name]["failure"] += 1
                if "error_message" in execution:
                    error = classify_error(Exception(execution["error_message"]))
                    tool_usage[tool_name]["error_types"][error.error_type] += 1

        # Prepare the input for the prompt
        input_values = {
            "error_patterns": json.dumps(error_patterns),
            "tool_usage_patterns": json.dumps(tool_usage),
        }

        try:
            # Get the correlation analysis from the model
            messages = self.correlation_prompt.format_messages(**input_values)
            response = await self.model.ainvoke(messages)

            # Parse the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0]
                if "```json" in content
                else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            correlations = json.loads(json_str)

            # Add timestamp to the analysis
            correlations["timestamp"] = time.time()

            # Save the analysis to the database
            self.db.save_entity(
                "advanced_error_analysis",
                f"correlations_{int(time.time())}",
                correlations,
            )

            self.error_correlations = correlations
            return correlations
        except Exception as e:
            logger.error(f"Error analyzing error correlations: {str(e)}")

            # Create a default analysis
            default_correlations = {
                "correlations": [],
                "sequential_patterns": [],
                "tool_correlations": {},
                "cascading_patterns": [],
                "timestamp": time.time(),
                "parsing_error": str(e),
            }

            # Save the default analysis to the database
            self.db.save_entity(
                "advanced_error_analysis",
                f"correlations_{int(time.time())}",
                default_correlations,
            )

            self.error_correlations = default_correlations
            return default_correlations

    async def predict_potential_errors(self) -> Dict[str, Any]:
        """Predict potential future errors based on historical patterns.

        Returns:
            Prediction results with potential errors and preemptive actions
        """
        # Get all error executions from the database
        executions = self.db.get_entities_by_prefix("error_recovery", "execution_")

        if not executions:
            return {
                "predicted_errors": [],
                "warning_signs": [],
                "preemptive_actions": [],
                "high_risk_tools": [],
            }

        # Sort executions by timestamp
        executions.sort(key=lambda e: e.get("timestamp", 0))

        # Get recent executions (last 50 or all if less than 50)
        recent_executions = executions[-50:] if len(executions) > 50 else executions

        # Format error history
        error_history = []
        for execution in executions:
            if not execution.get("success", True) and "error_message" in execution:
                error = classify_error(Exception(execution.get("error_message", "")))
                error_history.append(
                    {
                        "tool": execution.get("tool_name", "unknown"),
                        "error_type": error.error_type,
                        "timestamp": execution.get("timestamp", 0),
                        "error_message": execution.get("error_message", ""),
                    }
                )

        # Format current system state
        current_state = {
            "total_executions": len(executions),
            "recent_executions": len(recent_executions),
            "recent_success_rate": sum(
                1 for e in recent_executions if e.get("success", True)
            )
            / len(recent_executions)
            if recent_executions
            else 0,
            "total_success_rate": sum(1 for e in executions if e.get("success", True))
            / len(executions)
            if executions
            else 0,
            "unique_tools_used": len(
                set(e.get("tool_name", "unknown") for e in executions)
            ),
            "error_clusters": len(self.error_clusters) if self.error_clusters else 0,
        }

        # Format recent tool usage
        recent_tool_usage = {}
        for execution in recent_executions:
            tool_name = execution.get("tool_name", "unknown")
            if tool_name not in recent_tool_usage:
                recent_tool_usage[tool_name] = {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "error_types": Counter(),
                }

            recent_tool_usage[tool_name]["total"] += 1

            if execution.get("success", True):
                recent_tool_usage[tool_name]["success"] += 1
            else:
                recent_tool_usage[tool_name]["failure"] += 1
                if "error_message" in execution:
                    error = classify_error(Exception(execution["error_message"]))
                    recent_tool_usage[tool_name]["error_types"][error.error_type] += 1

        # Prepare the input for the prompt
        input_values = {
            "error_history": json.dumps(error_history),
            "current_state": json.dumps(current_state),
            "recent_tool_usage": json.dumps(recent_tool_usage),
        }

        try:
            # Get the predictive analysis from the model
            messages = self.predictive_prompt.format_messages(**input_values)
            response = await self.model.ainvoke(messages)

            # Parse the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0]
                if "```json" in content
                else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            predictions = json.loads(json_str)

            # Add timestamp to the predictions
            predictions["timestamp"] = time.time()

            # Save the predictions to the database
            self.db.save_entity(
                "advanced_error_analysis",
                f"predictions_{int(time.time())}",
                predictions,
            )

            self.predictive_patterns = predictions
            return predictions
        except Exception as e:
            logger.error(f"Error predicting potential errors: {str(e)}")

            # Create default predictions
            default_predictions = {
                "predicted_errors": [],
                "warning_signs": [],
                "preemptive_actions": [],
                "high_risk_tools": [],
                "timestamp": time.time(),
                "parsing_error": str(e),
            }

            # Save the default predictions to the database
            self.db.save_entity(
                "advanced_error_analysis",
                f"predictions_{int(time.time())}",
                default_predictions,
            )

            self.predictive_patterns = default_predictions
            return default_predictions

    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run a comprehensive error analysis including clustering, root cause analysis,
        correlation analysis, and predictive analysis.

        Returns:
            Comprehensive analysis results
        """
        # Run all analyses
        clusters = await self.cluster_errors()
        root_causes = await self.analyze_root_causes()
        correlations = await self.analyze_error_correlations()
        predictions = await self.predict_potential_errors()

        # Combine results
        comprehensive_results = {
            "clusters": [cluster.to_dict() for cluster in clusters],
            "root_causes": root_causes,
            "correlations": correlations,
            "predictions": predictions,
            "timestamp": time.time(),
        }

        # Save comprehensive results to database
        self.db.save_entity(
            "advanced_error_analysis",
            f"comprehensive_{int(time.time())}",
            comprehensive_results,
        )

        return comprehensive_results
