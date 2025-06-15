"""
Mock Research Reports Agent.
This script demonstrates the functionality of the research reports agent without making external API calls.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict


class MockResearchReportsAgent:
    """Mock implementation of the Research Reports Agent."""

    def __init__(self):
        """Initialize the mock research reports agent."""
        self.reports = {}

    def generate_research_report(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """Generate a mock research report.
        
        Args:
            topic: Research topic
            depth: Research depth ("shallow", "medium", "deep")
            
        Returns:
            Generated report
        """
        print(f"Generating research report on '{topic}' with {depth} depth...")

        # Step 1: Collect data (mock)
        print("Collecting data from various sources...")
        time.sleep(1)  # Simulate API call

        # Step 2: Analyze data (mock)
        print("Analyzing collected data...")
        time.sleep(1)  # Simulate processing

        # Step 3: Generate report (mock)
        print("Generating report...")
        time.sleep(1)  # Simulate processing

        # Create a mock report
        report = self._create_mock_report(topic, depth)

        # Step 4: Format report (mock)
        print("Formatting report...")
        time.sleep(1)  # Simulate processing

        # Save the report
        timestamp = int(time.time())
        report_id = f"report_{timestamp}"
        self.reports[report_id] = report

        # Create the reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)

        # Save the report to a file
        filename = f"{topic.lower().replace(' ', '_')}_{timestamp}.md"
        filepath = os.path.join("reports", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._format_report_as_markdown(report))

        print(f"Report saved to {filepath}")

        return {
            "report_id": report_id,
            "topic": topic,
            "filepath": filepath,
            "timestamp": timestamp
        }

    def _create_mock_report(self, topic: str, depth: str) -> Dict[str, Any]:
        """Create a mock report.
        
        Args:
            topic: Research topic
            depth: Research depth
            
        Returns:
            Mock report
        """
        # Create sections based on depth
        sections = {
            "Introduction": f"This report provides a comprehensive overview of {topic}. It explores the key aspects, current trends, and future directions in this field.",
            "Background": f"{topic} has a rich history dating back several decades. It has evolved significantly over time, with major developments occurring in recent years.",
            "Methodology": "This research was conducted using a combination of literature review, data analysis, and expert interviews. Multiple sources were consulted to ensure a comprehensive understanding of the subject.",
            "Findings": f"Our research has uncovered several key findings about {topic}. These include emerging trends, challenges, and opportunities in the field."
        }

        # Add more sections for medium and deep depth
        if depth in ["medium", "deep"]:
            sections["Analysis"] = f"Analysis of the findings reveals important patterns and insights about {topic}. These have significant implications for various stakeholders."
            sections["Conclusion"] = f"In conclusion, {topic} represents a dynamic and evolving field with substantial potential for future development and impact."

        # Add more sections for deep depth
        if depth == "deep":
            sections["Recommendations"] = f"Based on our research, we recommend the following actions regarding {topic}: 1) Increase investment in research and development, 2) Foster collaboration between stakeholders, 3) Develop comprehensive policies and frameworks."
            sections["Future Directions"] = f"Future research on {topic} should focus on addressing current gaps in knowledge, exploring emerging trends, and developing innovative approaches to existing challenges."

        # Create executive summary
        executive_summary = f"This report examines {topic} in depth, providing a comprehensive analysis of its current state, key challenges, and future prospects. Our research indicates that {topic} is a rapidly evolving field with significant implications for various sectors. The report outlines major findings and offers recommendations for stakeholders."

        # Create bibliography
        bibliography = [
            f"Smith, J. (2023). 'The Future of {topic}'. Journal of Research Studies, 45(2), 123-145.",
            f"Johnson, A. & Williams, B. (2022). 'A Comprehensive Analysis of {topic}'. Annual Review, 12, 78-92.",
            f"Brown, C. et al. (2021). 'Emerging Trends in {topic}'. International Journal, 8(3), 201-215."
        ]

        # Create the report
        report = {
            "topic": topic,
            "executive_summary": executive_summary,
            "sections": sections,
            "bibliography": bibliography,
            "metadata": {
                "generated_at": time.time(),
                "depth": depth
            }
        }

        return report

    def _format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format a report as Markdown.
        
        Args:
            report: Research report
            
        Returns:
            Markdown content
        """
        # Create Markdown content
        markdown = f"# {report['topic']}\n\n"

        # Add executive summary
        markdown += "## Executive Summary\n\n"
        markdown += f"{report['executive_summary']}\n\n"

        # Add sections
        for section_name, section_content in report["sections"].items():
            markdown += f"## {section_name}\n\n"
            markdown += f"{section_content}\n\n"

        # Add bibliography
        markdown += "## Bibliography\n\n"
        for entry in report["bibliography"]:
            markdown += f"- {entry}\n"

        # Add timestamp
        markdown += f"\n\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        return markdown


def main():
    """Run the mock research reports agent."""
    agent = MockResearchReportsAgent()

    print("Welcome to the Mock Research Reports Agent!")
    print("Type 'research [topic]' to generate a research report.")
    print("Type 'exit' to quit.")
    print()

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Process the user input
        if user_input.lower().startswith("research "):
            # Extract the topic
            topic = user_input[len("research "):].strip()

            # Generate research report
            result = agent.generate_research_report(topic)

            print(f"\nAgent: Research report on '{topic}' has been generated and saved to {result['filepath']}.\n")
        else:
            print("\nAgent: I can help you generate comprehensive research reports. Type 'research [topic]' to get started.\n")


if __name__ == "__main__":
    main()
