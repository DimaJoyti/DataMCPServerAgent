"""
AI & Machine Learning Infrastructure for Institutional Trading

Advanced AI/ML capabilities for next-generation trading systems:
- Real-time ML inference
- Price prediction models
- Sentiment analysis
- Reinforcement learning
- Alternative data integration
- AI-powered strategies
"""

from .ml_engine import MLEngine
from .feature_engineering import FeatureEngineer
from .model_manager import ModelManager
from .data_pipeline import MLDataPipeline

__all__ = [
    'MLEngine',
    'FeatureEngineer', 
    'ModelManager',
    'MLDataPipeline'
]
