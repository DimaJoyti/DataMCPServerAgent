"""
Visualization tools for SEO.

This module provides tools for generating visualizations of SEO data.
"""

import os
import json
import base64
import io
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud
from langchain.tools import Tool

# Directory for storing visualizations
VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "visualizations", "seo")
os.makedirs(VIS_DIR, exist_ok=True)


class SEOVisualizationTool:
    """Tool for generating visualizations of SEO data."""
    
    def __init__(self):
        """Initialize the SEO visualization tool."""
        # Set up matplotlib style
        plt.style.use('ggplot')
        
        # Define color palettes
        self.color_palettes = {
            "blue": sns.color_palette("Blues_d"),
            "green": sns.color_palette("Greens_d"),
            "red": sns.color_palette("Reds_d"),
            "purple": sns.color_palette("Purples_d"),
            "orange": sns.color_palette("Oranges_d"),
            "default": sns.color_palette("husl", 8)
        }
    
    def generate_keyword_rankings_chart(self, domain: str, keywords: List[Dict[str, Any]], 
                                        chart_type: str = "bar", color_palette: str = "blue") -> str:
        """
        Generate a chart of keyword rankings.
        
        Args:
            domain: The domain the keywords are for
            keywords: List of keyword data (each with 'keyword' and 'position' keys)
            chart_type: Type of chart ('bar', 'line', 'horizontal')
            color_palette: Color palette to use
            
        Returns:
            Path to the generated chart image
        """
        print(f"Generating keyword rankings chart for {domain}...")
        
        # Create a DataFrame from the keywords
        df = pd.DataFrame(keywords)
        
        # Sort by position (ascending, better rankings first)
        df = df.sort_values('position')
        
        # Take top 10 keywords
        if len(df) > 10:
            df = df.head(10)
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Get the color palette
        colors = self.color_palettes.get(color_palette, self.color_palettes["default"])
        
        # Generate the chart based on type
        if chart_type == "bar":
            ax = sns.barplot(x='keyword', y='position', data=df, palette=colors)
            plt.xticks(rotation=45, ha='right')
            plt.gca().invert_yaxis()  # Invert Y-axis so lower (better) rankings are higher
            
        elif chart_type == "horizontal":
            ax = sns.barplot(y='keyword', x='position', data=df, palette=colors)
            plt.gca().invert_xaxis()  # Invert X-axis so lower (better) rankings are to the right
            
        elif chart_type == "line":
            # For line chart, we need historical data
            # If 'history' is in the keyword data, use it
            if 'history' in df.columns:
                # Reshape data for line chart
                history_data = []
                for _, row in df.iterrows():
                    keyword = row['keyword']
                    for date, position in row['history'].items():
                        history_data.append({
                            'keyword': keyword,
                            'date': date,
                            'position': position
                        })
                
                history_df = pd.DataFrame(history_data)
                history_df['date'] = pd.to_datetime(history_df['date'])
                
                # Plot line chart
                sns.lineplot(x='date', y='position', hue='keyword', data=history_df, palette=colors)
                plt.gca().invert_yaxis()  # Invert Y-axis so lower (better) rankings are higher
                plt.xticks(rotation=45, ha='right')
            else:
                # If no history, fall back to bar chart
                ax = sns.barplot(x='keyword', y='position', data=df, palette=colors)
                plt.xticks(rotation=45, ha='right')
                plt.gca().invert_yaxis()
        
        # Add labels and title
        plt.title(f'Keyword Rankings for {domain}', fontsize=16)
        plt.xlabel('Keyword', fontsize=12)
        plt.ylabel('Position in Search Results', fontsize=12)
        
        # Add a horizontal line at position 10 (first page)
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7)
        plt.text(0, 10.5, 'First Page Cutoff', color='red', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(VIS_DIR, f"{domain.replace('.', '_')}_keyword_rankings_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_seo_score_comparison(self, domain: str, competitors: List[Dict[str, Any]], 
                                      color_palette: str = "green") -> str:
        """
        Generate a chart comparing SEO scores.
        
        Args:
            domain: The main domain
            competitors: List of competitor data (each with 'domain' and 'seo_score' keys)
            color_palette: Color palette to use
            
        Returns:
            Path to the generated chart image
        """
        print(f"Generating SEO score comparison chart for {domain} and competitors...")
        
        # Create a DataFrame from the competitors
        df = pd.DataFrame(competitors)
        
        # Sort by SEO score (descending)
        df = df.sort_values('seo_score', ascending=False)
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Get the color palette
        colors = self.color_palettes.get(color_palette, self.color_palettes["default"])
        
        # Generate the bar chart
        ax = sns.barplot(x='domain', y='seo_score', data=df, palette=colors)
        
        # Highlight the main domain
        for i, d in enumerate(df['domain']):
            if d == domain:
                ax.patches[i].set_facecolor('gold')
                ax.patches[i].set_edgecolor('black')
                break
        
        # Add labels and title
        plt.title(f'SEO Score Comparison: {domain} vs. Competitors', fontsize=16)
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('SEO Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.1f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10, color='black')
        
        # Set y-axis range from 0 to 100
        plt.ylim(0, 100)
        
        # Add horizontal lines for score ranges
        plt.axhline(y=80, color='green', linestyle='--', alpha=0.7)
        plt.text(0, 81, 'Excellent', color='green', alpha=0.7)
        
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
        plt.text(0, 61, 'Good', color='orange', alpha=0.7)
        
        plt.axhline(y=40, color='red', linestyle='--', alpha=0.7)
        plt.text(0, 41, 'Needs Improvement', color='red', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(VIS_DIR, f"{domain.replace('.', '_')}_seo_score_comparison_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_backlink_profile(self, domain: str, backlink_data: Dict[str, Any], 
                                 chart_type: str = "pie", color_palette: str = "purple") -> str:
        """
        Generate a chart of backlink profile.
        
        Args:
            domain: The domain the backlinks are for
            backlink_data: Backlink data including domain authority distribution
            chart_type: Type of chart ('pie', 'bar', 'donut')
            color_palette: Color palette to use
            
        Returns:
            Path to the generated chart image
        """
        print(f"Generating backlink profile chart for {domain}...")
        
        # Extract domain authority distribution
        da_distribution = backlink_data.get("domain_authority_distribution", {})
        
        # Create lists for the chart
        categories = list(da_distribution.keys())
        values = list(da_distribution.values())
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Get the color palette
        colors = self.color_palettes.get(color_palette, self.color_palettes["default"])
        
        # Generate the chart based on type
        if chart_type == "pie":
            plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
        elif chart_type == "donut":
            # Create a donut chart (pie chart with a hole in the middle)
            plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors,
                   wedgeprops=dict(width=0.5))
            plt.axis('equal')
            
        elif chart_type == "bar":
            # Create a bar chart
            plt.bar(categories, values, color=colors)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # Add title and legend
        plt.title(f'Backlink Profile: Domain Authority Distribution for {domain}', fontsize=16)
        
        if chart_type in ["pie", "donut"]:
            plt.legend(title="Domain Authority Ranges", loc="best")
        
        # Add additional information as text
        total_backlinks = backlink_data.get("total_backlinks", 0)
        plt.figtext(0.5, 0.01, f"Total Backlinks: {total_backlinks}", ha="center", fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(VIS_DIR, f"{domain.replace('.', '_')}_backlink_profile_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_content_analysis(self, content: str, keywords: List[str], 
                                 color_palette: str = "orange") -> str:
        """
        Generate a visualization of content analysis.
        
        Args:
            content: The content to analyze
            keywords: List of target keywords
            color_palette: Color palette to use
            
        Returns:
            Path to the generated visualization image
        """
        print(f"Generating content analysis visualization...")
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get the color palette
        colors = self.color_palettes.get(color_palette, self.color_palettes["default"])
        
        # 1. Word Cloud (top-left)
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap=color_palette, max_words=100).generate(content)
        
        axs[0, 0].imshow(wordcloud, interpolation='bilinear')
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Content Word Cloud', fontsize=14)
        
        # 2. Keyword Density (top-right)
        keyword_counts = {}
        for keyword in keywords:
            # Count occurrences (case insensitive)
            count = content.lower().count(keyword.lower())
            keyword_counts[keyword] = count
        
        # Sort by count (descending)
        keyword_counts = {k: v for k, v in sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)}
        
        # Calculate total words
        total_words = len(content.split())
        
        # Calculate density
        keyword_density = {k: (v / total_words) * 100 for k, v in keyword_counts.items()}
        
        # Plot keyword density
        kw_keys = list(keyword_density.keys())
        kw_values = list(keyword_density.values())
        
        axs[0, 1].bar(kw_keys, kw_values, color=colors)
        axs[0, 1].set_title('Keyword Density (%)', fontsize=14)
        axs[0, 1].set_ylabel('Density (%)')
        axs[0, 1].set_xticklabels(kw_keys, rotation=45, ha='right')
        
        # Add a horizontal line at 2% (optimal density)
        axs[0, 1].axhline(y=2, color='green', linestyle='--', alpha=0.7)
        axs[0, 1].text(0, 2.1, 'Optimal Density', color='green', alpha=0.7)
        
        # 3. Content Structure (bottom-left)
        # Count headings, paragraphs, etc.
        h1_count = content.count('# ')
        h2_count = content.count('## ')
        h3_count = content.count('### ')
        paragraphs = content.count('\n\n')
        sentences = content.count('. ') + content.count('! ') + content.count('? ')
        
        structure_labels = ['H1', 'H2', 'H3', 'Paragraphs', 'Sentences']
        structure_values = [h1_count, h2_count, h3_count, paragraphs, sentences]
        
        axs[1, 0].bar(structure_labels, structure_values, color=colors)
        axs[1, 0].set_title('Content Structure', fontsize=14)
        axs[1, 0].set_ylabel('Count')
        
        # 4. Readability Score (bottom-right)
        # Calculate a simple readability score (Flesch Reading Ease)
        words = content.split()
        word_count = len(words)
        sentence_count = max(1, sentences)
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        if sentence_count > 0 and word_count > 0:
            flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            flesch_score = min(100, max(0, flesch_score))
        else:
            flesch_score = 0
        
        # Create a gauge chart for readability
        gauge_colors = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
        gauge_positions = [0, 30, 50, 70, 90, 100]
        
        # Create the gauge
        for i in range(len(gauge_colors)):
            axs[1, 1].barh(0, gauge_positions[i+1] - gauge_positions[i], left=gauge_positions[i], 
                          height=0.5, color=gauge_colors[i])
        
        # Add the needle
        axs[1, 1].plot([flesch_score, flesch_score], [0, 0.5], color='black', linewidth=2)
        axs[1, 1].scatter(flesch_score, 0, color='black', s=100, zorder=5)
        
        # Add labels
        axs[1, 1].text(10, -0.2, 'Very Difficult', ha='center', va='top')
        axs[1, 1].text(40, -0.2, 'Difficult', ha='center', va='top')
        axs[1, 1].text(60, -0.2, 'Standard', ha='center', va='top')
        axs[1, 1].text(80, -0.2, 'Easy', ha='center', va='top')
        axs[1, 1].text(95, -0.2, 'Very Easy', ha='center', va='top')
        
        axs[1, 1].text(50, 0.7, f'Readability Score: {flesch_score:.1f}', ha='center', fontsize=12)
        
        # Set limits and remove ticks
        axs[1, 1].set_xlim(0, 100)
        axs[1, 1].set_ylim(-0.5, 1)
        axs[1, 1].set_title('Readability (Flesch Reading Ease)', fontsize=14)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        
        # Add overall title
        plt.suptitle('Content Analysis', fontsize=18)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = os.path.join(VIS_DIR, f"content_analysis_{timestamp}.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return vis_path
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word.
        
        Args:
            word: The word to count syllables for
            
        Returns:
            Number of syllables
        """
        word = word.lower()
        
        # Remove non-alphabetic characters
        word = ''.join(c for c in word if c.isalpha())
        
        if not word:
            return 0
        
        # Count vowel groups
        count = len([m for m in re.findall(r'[aeiouy]+', word)])
        
        # Adjust for silent e at the end
        if word.endswith('e'):
            count -= 1
        
        # Ensure at least one syllable
        return max(1, count)
    
    def generate_visualization(self, data_type: str, data: Dict[str, Any], 
                              chart_type: str = "default", color_palette: str = "default") -> str:
        """
        Generate a visualization based on data type.
        
        Args:
            data_type: Type of data to visualize
            data: Data to visualize
            chart_type: Type of chart
            color_palette: Color palette to use
            
        Returns:
            Path to the generated visualization image
        """
        print(f"Generating {data_type} visualization...")
        
        try:
            if data_type == "keyword_rankings":
                domain = data.get("domain", "example.com")
                keywords = data.get("keywords", [])
                return self.generate_keyword_rankings_chart(domain, keywords, chart_type, color_palette)
                
            elif data_type == "seo_score_comparison":
                domain = data.get("domain", "example.com")
                competitors = data.get("competitors", [])
                return self.generate_seo_score_comparison(domain, competitors, color_palette)
                
            elif data_type == "backlink_profile":
                domain = data.get("domain", "example.com")
                backlink_data = data.get("backlink_data", {})
                return self.generate_backlink_profile(domain, backlink_data, chart_type, color_palette)
                
            elif data_type == "content_analysis":
                content = data.get("content", "")
                keywords = data.get("keywords", [])
                return self.generate_content_analysis(content, keywords, color_palette)
                
            else:
                return f"Unsupported data type: {data_type}"
                
        except Exception as e:
            return f"Error generating visualization: {str(e)}"
    
    def run(self, data_type: str, data_json: str, chart_type: str = "default", 
           color_palette: str = "default", format: str = "png") -> str:
        """
        Run the visualization tool and return formatted results.
        
        Args:
            data_type: Type of data to visualize
            data_json: JSON string with data to visualize
            chart_type: Type of chart
            color_palette: Color palette to use
            format: Output format ('png', 'base64')
            
        Returns:
            Formatted string with visualization results
        """
        try:
            # Parse the JSON data
            data = json.loads(data_json)
            
            # Generate the visualization
            vis_path = self.generate_visualization(data_type, data, chart_type, color_palette)
            
            if format == "base64":
                # Convert the image to base64
                with open(vis_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                return f"data:image/png;base64,{encoded_string}"
            else:
                return f"Visualization generated and saved to: {vis_path}"
                
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data"
        except Exception as e:
            return f"Error: {str(e)}"


# Create tool instance
seo_visualization = SEOVisualizationTool()

# Create LangChain tool
seo_visualization_tool = Tool(
    name="seo_visualization",
    func=seo_visualization.run,
    description="Generate visualizations of SEO data. Creates charts and graphs for SEO metrics.",
)
