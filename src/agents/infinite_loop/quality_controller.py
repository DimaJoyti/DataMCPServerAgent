"""
Quality Controller

Validates and ensures quality of generated iterations according to
specifications and quality standards.
"""

import logging
import re
from typing import Any, Dict, List, Optional


class QualityController:
    """
    Controls and validates quality of generated iterations.
    
    Features:
    - Content quality validation
    - Specification compliance checking
    - Uniqueness verification
    - Format validation
    - Quality scoring and metrics
    """
    
    def __init__(self, config: Any):
        """Initialize the quality controller."""
        self.config = config
        self.logger = logging.getLogger("quality_controller")
        
        # Quality thresholds
        self.quality_threshold = config.quality_threshold
        self.uniqueness_threshold = config.uniqueness_threshold
        
        # Validation rules
        self.validation_rules = {
            "min_length": 50,
            "max_length": 100000,
            "required_sections": [],
            "forbidden_patterns": [],
        }
    
    async def validate_iteration(
        self,
        content: str,
        spec_analysis: Dict[str, Any],
        existing_iterations: List[str],
        iteration_number: int,
    ) -> Dict[str, Any]:
        """
        Validate a generated iteration for quality and compliance.
        
        Args:
            content: Generated content to validate
            spec_analysis: Specification analysis
            existing_iterations: List of existing iteration content
            iteration_number: Current iteration number
            
        Returns:
            Validation result with quality score and issues
        """
        validation_result = {
            "valid": True,
            "quality_score": 0.0,
            "uniqueness_score": 0.0,
            "compliance_score": 0.0,
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }
        
        try:
            # Basic content validation
            basic_validation = await self._validate_basic_content(content)
            validation_result.update(basic_validation)
            
            # Specification compliance
            compliance_validation = await self._validate_specification_compliance(
                content, spec_analysis
            )
            validation_result["compliance_score"] = compliance_validation["score"]
            validation_result["issues"].extend(compliance_validation["issues"])
            
            # Uniqueness validation
            uniqueness_validation = await self._validate_uniqueness(
                content, existing_iterations
            )
            validation_result["uniqueness_score"] = uniqueness_validation["score"]
            validation_result["issues"].extend(uniqueness_validation["issues"])
            
            # Format validation
            format_validation = await self._validate_format(content, spec_analysis)
            validation_result["issues"].extend(format_validation["issues"])
            
            # Calculate overall quality score
            validation_result["quality_score"] = self._calculate_quality_score(
                validation_result["compliance_score"],
                validation_result["uniqueness_score"],
                len(validation_result["issues"]),
            )
            
            # Determine if valid
            validation_result["valid"] = (
                validation_result["quality_score"] >= self.quality_threshold and
                validation_result["uniqueness_score"] >= self.uniqueness_threshold and
                len([issue for issue in validation_result["issues"] if issue["severity"] == "error"]) == 0
            )
            
            # Generate recommendations
            validation_result["recommendations"] = self._generate_recommendations(
                validation_result
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            validation_result.update({
                "valid": False,
                "quality_score": 0.0,
                "issues": [{"severity": "error", "message": f"Validation error: {str(e)}"}],
            })
        
        return validation_result
    
    async def _validate_basic_content(self, content: str) -> Dict[str, Any]:
        """Validate basic content properties."""
        issues = []
        
        # Length validation
        if len(content) < self.validation_rules["min_length"]:
            issues.append({
                "severity": "error",
                "message": f"Content too short: {len(content)} < {self.validation_rules['min_length']}",
            })
        
        if len(content) > self.validation_rules["max_length"]:
            issues.append({
                "severity": "error",
                "message": f"Content too long: {len(content)} > {self.validation_rules['max_length']}",
            })
        
        # Empty content check
        if not content.strip():
            issues.append({
                "severity": "error",
                "message": "Content is empty or whitespace only",
            })
        
        # Basic quality indicators
        word_count = len(content.split())
        if word_count < 10:
            issues.append({
                "severity": "warning",
                "message": f"Very low word count: {word_count}",
            })
        
        return {"issues": issues}
    
    async def _validate_specification_compliance(
        self, content: str, spec_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate compliance with specification requirements."""
        issues = []
        score = 1.0
        
        # Check requirements
        requirements = spec_analysis.get("requirements", [])
        for requirement in requirements:
            if not self._check_requirement_compliance(content, requirement):
                issues.append({
                    "severity": "error",
                    "message": f"Requirement not met: {requirement}",
                })
                score -= 0.2
        
        # Check constraints
        constraints = spec_analysis.get("constraints", [])
        for constraint in constraints:
            if not self._check_constraint_compliance(content, constraint):
                issues.append({
                    "severity": "error",
                    "message": f"Constraint violated: {constraint}",
                })
                score -= 0.3
        
        return {"score": max(0.0, score), "issues": issues}
    
    async def _validate_uniqueness(
        self, content: str, existing_iterations: List[str]
    ) -> Dict[str, Any]:
        """Validate uniqueness against existing iterations."""
        issues = []
        
        if not existing_iterations:
            return {"score": 1.0, "issues": []}
        
        # Calculate similarity scores
        similarity_scores = []
        for existing_content in existing_iterations:
            similarity = self._calculate_similarity(content, existing_content)
            similarity_scores.append(similarity)
        
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        uniqueness_score = 1.0 - max_similarity
        
        if max_similarity > 0.8:
            issues.append({
                "severity": "error",
                "message": f"Content too similar to existing iteration: {max_similarity:.1%} similarity",
            })
        elif max_similarity > 0.6:
            issues.append({
                "severity": "warning",
                "message": f"Content somewhat similar to existing iteration: {max_similarity:.1%} similarity",
            })
        
        return {"score": uniqueness_score, "issues": issues}
    
    async def _validate_format(self, content: str, spec_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content format."""
        issues = []
        format_type = spec_analysis.get("format", "text")
        
        if format_type == "json":
            try:
                import json
                json.loads(content)
            except json.JSONDecodeError as e:
                issues.append({
                    "severity": "error",
                    "message": f"Invalid JSON format: {str(e)}",
                })
        
        elif format_type == "yaml":
            try:
                import yaml
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                issues.append({
                    "severity": "error",
                    "message": f"Invalid YAML format: {str(e)}",
                })
        
        elif format_type == "python":
            try:
                compile(content, '<string>', 'exec')
            except SyntaxError as e:
                issues.append({
                    "severity": "error",
                    "message": f"Invalid Python syntax: {str(e)}",
                })
        
        return {"issues": issues}
    
    def _check_requirement_compliance(self, content: str, requirement: str) -> bool:
        """Check if content meets a specific requirement."""
        # Simple keyword-based checking
        requirement_lower = requirement.lower()
        content_lower = content.lower()
        
        # Extract key terms from requirement
        key_terms = re.findall(r'\b\w+\b', requirement_lower)
        
        # Check if most key terms are present
        present_terms = sum(1 for term in key_terms if term in content_lower)
        compliance_ratio = present_terms / len(key_terms) if key_terms else 1.0
        
        return compliance_ratio >= 0.7  # 70% of terms should be present
    
    def _check_constraint_compliance(self, content: str, constraint: str) -> bool:
        """Check if content violates a constraint."""
        constraint_lower = constraint.lower()
        content_lower = content.lower()
        
        # Check for forbidden patterns
        forbidden_patterns = [
            "must not", "cannot", "forbidden", "prohibited", "not allowed"
        ]
        
        for pattern in forbidden_patterns:
            if pattern in constraint_lower:
                # Extract what should not be present
                parts = constraint_lower.split(pattern)
                if len(parts) > 1:
                    forbidden_content = parts[1].strip()
                    if forbidden_content in content_lower:
                        return False
        
        return True
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_quality_score(
        self, compliance_score: float, uniqueness_score: float, issue_count: int
    ) -> float:
        """Calculate overall quality score."""
        base_score = (compliance_score + uniqueness_score) / 2.0
        
        # Penalize for issues
        issue_penalty = min(issue_count * 0.1, 0.5)  # Max 50% penalty
        
        return max(0.0, base_score - issue_penalty)
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if validation_result["quality_score"] < self.quality_threshold:
            recommendations.append("Improve overall content quality")
        
        if validation_result["uniqueness_score"] < self.uniqueness_threshold:
            recommendations.append("Increase uniqueness compared to existing iterations")
        
        if validation_result["compliance_score"] < 0.8:
            recommendations.append("Better align content with specification requirements")
        
        error_issues = [issue for issue in validation_result["issues"] if issue["severity"] == "error"]
        if error_issues:
            recommendations.append("Fix critical errors before proceeding")
        
        return recommendations
