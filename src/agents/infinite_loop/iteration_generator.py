"""
Iteration Generator

Generates unique content iterations based on specifications, existing content analysis,
and assigned innovation dimensions. Core component for creating novel iterations.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool


class IterationGenerator:
    """
    Generates unique content iterations based on specifications and innovation dimensions.
    
    Features:
    - Specification-driven content generation
    - Innovation dimension focus for uniqueness
    - Existing content analysis and differentiation
    - Quality control and validation
    - Multiple output formats support
    - Error handling and recovery
    """
    
    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        agent_id: str,
    ):
        """Initialize the iteration generator."""
        self.model = model
        self.tools = tools
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"iteration_generator_{agent_id}")
        
        # Generation settings
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Quality thresholds
        self.min_content_length = 100
        self.max_content_length = 50000
        
    async def generate_iteration(
        self,
        iteration_number: int,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        innovation_dimension: str,
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Generate a unique iteration based on the provided parameters.
        
        Args:
            iteration_number: The iteration number to generate
            spec_analysis: Parsed specification analysis
            directory_state: Current directory state analysis
            innovation_dimension: Assigned innovation dimension
            output_dir: Output directory path
            
        Returns:
            Generation result with content and metadata
        """
        start_time = time.time()
        
        self.logger.info(f"Generating iteration {iteration_number} with dimension: {innovation_dimension}")
        
        try:
            # Prepare generation context
            context = await self._prepare_generation_context(
                iteration_number, spec_analysis, directory_state, innovation_dimension
            )
            
            # Generate content with retries
            content = await self._generate_content_with_retries(context)
            
            # Validate content
            validation_result = await self._validate_content(content, spec_analysis)
            if not validation_result["valid"]:
                raise ValueError(f"Content validation failed: {validation_result['reason']}")
            
            # Determine output filename
            filename = await self._determine_filename(
                iteration_number, spec_analysis, output_dir
            )
            
            # Save content to file
            file_path = await self._save_content(content, filename, output_dir)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare result
            result = {
                "success": True,
                "iteration_number": iteration_number,
                "innovation_dimension": innovation_dimension,
                "content": content,
                "file_path": str(file_path),
                "filename": filename,
                "content_length": len(content),
                "generation_time": generation_time,
                "agent_id": self.agent_id,
                "validation": validation_result,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "spec_type": spec_analysis.get("content_type", "unknown"),
                    "format": spec_analysis.get("format", "unknown"),
                    "evolution_pattern": spec_analysis.get("evolution_pattern", "incremental"),
                },
            }
            
            self.logger.info(f"Successfully generated iteration {iteration_number} in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_message = str(e)
            
            self.logger.error(f"Failed to generate iteration {iteration_number}: {error_message}")
            
            return {
                "success": False,
                "iteration_number": iteration_number,
                "innovation_dimension": innovation_dimension,
                "error": error_message,
                "generation_time": generation_time,
                "agent_id": self.agent_id,
            }
    
    async def _prepare_generation_context(
        self,
        iteration_number: int,
        spec_analysis: Dict[str, Any],
        directory_state: Dict[str, Any],
        innovation_dimension: str,
    ) -> Dict[str, Any]:
        """Prepare the context for content generation."""
        # Extract key information
        content_type = spec_analysis.get("content_type", "unknown")
        format_type = spec_analysis.get("format", "unknown")
        evolution_pattern = spec_analysis.get("evolution_pattern", "incremental")
        requirements = spec_analysis.get("requirements", [])
        constraints = spec_analysis.get("constraints", [])
        
        # Analyze existing iterations
        existing_iterations = directory_state.get("iteration_files", [])
        existing_summary = await self._summarize_existing_iterations(existing_iterations)
        
        # Prepare innovation focus
        innovation_focus = await self._prepare_innovation_focus(
            innovation_dimension, content_type, iteration_number
        )
        
        context = {
            "iteration_number": iteration_number,
            "content_type": content_type,
            "format_type": format_type,
            "evolution_pattern": evolution_pattern,
            "requirements": requirements,
            "constraints": constraints,
            "innovation_dimension": innovation_dimension,
            "innovation_focus": innovation_focus,
            "existing_summary": existing_summary,
            "total_existing": len(existing_iterations),
            "highest_iteration": directory_state.get("highest_iteration", 0),
        }
        
        return context
    
    async def _generate_content_with_retries(self, context: Dict[str, Any]) -> str:
        """Generate content with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                content = await self._generate_content(context, attempt)
                return content
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        raise last_error or RuntimeError("Content generation failed after all retries")
    
    async def _generate_content(self, context: Dict[str, Any], attempt: int) -> str:
        """Generate content using the language model."""
        # Prepare system prompt
        system_prompt = await self._create_system_prompt(context, attempt)
        
        # Prepare user prompt
        user_prompt = await self._create_user_prompt(context, attempt)
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        # Generate content
        response = await self.model.ainvoke(messages)
        content = response.content.strip()
        
        if not content:
            raise ValueError("Generated content is empty")
        
        return content
    
    async def _create_system_prompt(self, context: Dict[str, Any], attempt: int) -> str:
        """Create the system prompt for content generation."""
        prompt = f"""You are a specialized content generator creating iteration {context['iteration_number']} of {context['content_type']} content.

CORE MISSION:
Generate unique, high-quality content that follows the specification while introducing novel elements through the assigned innovation dimension.

CONTENT SPECIFICATIONS:
- Content Type: {context['content_type']}
- Format: {context['format_type']}
- Evolution Pattern: {context['evolution_pattern']}
- Innovation Dimension: {context['innovation_dimension']}

INNOVATION FOCUS:
{context['innovation_focus']}

EXISTING CONTENT CONTEXT:
{context['existing_summary']}

REQUIREMENTS:
{chr(10).join(f"- {req}" for req in context['requirements']) if context['requirements'] else "- Follow general best practices"}

CONSTRAINTS:
{chr(10).join(f"- {constraint}" for constraint in context['constraints']) if context['constraints'] else "- No specific constraints"}

QUALITY STANDARDS:
- Content must be genuinely unique and different from existing iterations
- Must demonstrate clear innovation in the assigned dimension
- Should maintain consistency with the overall specification
- Must be complete, functional, and well-structured
- Should add genuine value to the iteration sequence

UNIQUENESS DIRECTIVE:
This is iteration {context['iteration_number']} out of {context['total_existing'] + 1} total iterations. 
Your content MUST be distinctly different from all previous iterations while maintaining specification compliance.
Focus specifically on innovating in the "{context['innovation_dimension']}" dimension.

GENERATION GUIDELINES:
1. Start with the specification requirements as your foundation
2. Apply the innovation dimension to create unique elements
3. Ensure the content is complete and functional
4. Maintain high quality and professional standards
5. Make it genuinely valuable and different from existing iterations"""

        if attempt > 0:
            prompt += f"\n\nNOTE: This is attempt {attempt + 1}. Previous attempts failed. Please ensure the content is valid and meets all requirements."
        
        return prompt
    
    async def _create_user_prompt(self, context: Dict[str, Any], attempt: int) -> str:
        """Create the user prompt for content generation."""
        prompt = f"""Generate iteration {context['iteration_number']} focusing on the "{context['innovation_dimension']}" innovation dimension.

SPECIFIC TASK:
Create {context['content_type']} content in {context['format_type']} format that:

1. Follows the specification requirements exactly
2. Introduces novel elements through the "{context['innovation_dimension']}" dimension
3. Is distinctly different from the {context['total_existing']} existing iterations
4. Maintains high quality and completeness
5. Adds genuine value to the iteration sequence

INNOVATION DIMENSION FOCUS: {context['innovation_dimension']}
Apply this dimension creatively while maintaining specification compliance.

EVOLUTION PATTERN: {context['evolution_pattern']}
Follow this pattern in how your iteration builds upon or differs from previous work.

OUTPUT REQUIREMENTS:
- Provide ONLY the content itself, no explanations or metadata
- Ensure the content is complete and ready to use
- Make it genuinely unique and innovative
- Follow the specified format exactly

Generate the content now:"""
        
        return prompt
    
    async def _summarize_existing_iterations(self, existing_iterations: List[Dict[str, Any]]) -> str:
        """Summarize existing iterations to provide context."""
        if not existing_iterations:
            return "No existing iterations found. This will be the first iteration."
        
        summary_parts = [
            f"Found {len(existing_iterations)} existing iterations:",
        ]
        
        # Add iteration numbers
        iteration_numbers = [str(it["iteration_number"]) for it in existing_iterations[-5:]]
        summary_parts.append(f"Recent iterations: {', '.join(iteration_numbers)}")
        
        # Add size information
        if existing_iterations:
            sizes = [it["size"] for it in existing_iterations]
            avg_size = sum(sizes) / len(sizes)
            summary_parts.append(f"Average content size: {avg_size:.0f} characters")
        
        return "\n".join(summary_parts)
    
    async def _prepare_innovation_focus(
        self, innovation_dimension: str, content_type: str, iteration_number: int
    ) -> str:
        """Prepare specific innovation focus based on the dimension."""
        focus_map = {
            "functional_enhancement": "Improve core functionality, add new features, or enhance existing capabilities",
            "structural_innovation": "Reorganize structure, improve architecture, or introduce new organizational patterns",
            "interaction_patterns": "Enhance user interaction, improve interfaces, or introduce new interaction paradigms",
            "performance_optimization": "Optimize speed, efficiency, resource usage, or scalability",
            "user_experience": "Improve usability, accessibility, aesthetics, or user satisfaction",
            "integration_capabilities": "Enhance integration with other systems, APIs, or platforms",
            "scalability_improvements": "Improve ability to handle growth, load, or expansion",
            "security_enhancements": "Strengthen security, privacy, or data protection",
            "accessibility_features": "Improve accessibility for users with disabilities or diverse needs",
            "paradigm_shifts": "Introduce fundamentally new approaches or revolutionary concepts",
            "paradigm_revolution": "Completely reimagine the approach with revolutionary concepts",
            "cross_domain_synthesis": "Combine concepts from different domains or disciplines",
            "emergent_behaviors": "Introduce adaptive or self-organizing capabilities",
            "adaptive_intelligence": "Add learning, adaptation, or intelligent behavior",
            "quantum_improvements": "Make breakthrough improvements that change the game",
            "meta_optimization": "Optimize the optimization process itself",
            "holistic_integration": "Consider the entire ecosystem and interconnections",
            "future_proofing": "Prepare for future needs and technological evolution",
        }
        
        base_focus = focus_map.get(innovation_dimension, f"Focus on {innovation_dimension} improvements")
        
        # Add iteration-specific guidance
        if iteration_number <= 3:
            focus_level = "foundational"
        elif iteration_number <= 10:
            focus_level = "intermediate"
        else:
            focus_level = "advanced"
        
        return f"{base_focus}. Apply {focus_level} level innovation appropriate for iteration {iteration_number}."
    
    async def _validate_content(self, content: str, spec_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated content against requirements."""
        validation_result = {
            "valid": True,
            "reason": "",
            "checks": {},
        }
        
        # Length check
        if len(content) < self.min_content_length:
            validation_result["valid"] = False
            validation_result["reason"] = f"Content too short: {len(content)} < {self.min_content_length}"
            validation_result["checks"]["length"] = False
        elif len(content) > self.max_content_length:
            validation_result["valid"] = False
            validation_result["reason"] = f"Content too long: {len(content)} > {self.max_content_length}"
            validation_result["checks"]["length"] = False
        else:
            validation_result["checks"]["length"] = True
        
        # Non-empty check
        if not content.strip():
            validation_result["valid"] = False
            validation_result["reason"] = "Content is empty or whitespace only"
            validation_result["checks"]["non_empty"] = False
        else:
            validation_result["checks"]["non_empty"] = True
        
        # Format-specific validation
        format_type = spec_analysis.get("format", "unknown")
        validation_result["checks"]["format"] = await self._validate_format(content, format_type)
        
        if not validation_result["checks"]["format"]:
            validation_result["valid"] = False
            if not validation_result["reason"]:
                validation_result["reason"] = f"Content does not match expected format: {format_type}"
        
        return validation_result
    
    async def _validate_format(self, content: str, format_type: str) -> bool:
        """Validate content format."""
        if format_type == "json":
            try:
                import json
                json.loads(content)
                return True
            except json.JSONDecodeError:
                return False
        elif format_type == "yaml":
            try:
                import yaml
                yaml.safe_load(content)
                return True
            except yaml.YAMLError:
                return False
        elif format_type == "python":
            try:
                compile(content, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
        
        # For other formats, assume valid if non-empty
        return bool(content.strip())
    
    async def _determine_filename(
        self, iteration_number: int, spec_analysis: Dict[str, Any], output_dir: Union[str, Path]
    ) -> str:
        """Determine the filename for the generated content."""
        naming_pattern = spec_analysis.get("naming_pattern", "iteration_{number}")
        format_type = spec_analysis.get("format", "txt")
        
        # Replace placeholders
        filename = naming_pattern.format(
            number=iteration_number,
            iteration=iteration_number,
            iter=iteration_number,
        )
        
        # Add extension if not present
        if not Path(filename).suffix:
            extension_map = {
                "python": ".py",
                "javascript": ".js",
                "json": ".json",
                "yaml": ".yaml",
                "markdown": ".md",
                "html": ".html",
                "css": ".css",
                "xml": ".xml",
                "csv": ".csv",
                "sql": ".sql",
            }
            extension = extension_map.get(format_type, ".txt")
            filename += extension
        
        return filename
    
    async def _save_content(
        self, content: str, filename: str, output_dir: Union[str, Path]
    ) -> Path:
        """Save content to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        # Ensure we don't overwrite existing files
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            file_path = output_path / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Write content
        file_path.write_text(content, encoding="utf-8")
        
        self.logger.debug(f"Saved content to: {file_path}")
        return file_path
