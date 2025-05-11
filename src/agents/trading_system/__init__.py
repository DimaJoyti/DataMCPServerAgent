"""
Fetch.ai Advanced Crypto Trading System.

This module implements a sophisticated crypto leverage advisory system that combines
n8n workflows with fetch.ai agents to create an intelligent trading ecosystem.
"""

from .sentiment_agent import SentimentIntelligenceAgent
from .technical_agent import TechnicalAnalysisAgent
from .risk_agent import RiskManagementAgent
from .regulatory_agent import RegulatoryComplianceAgent
from .macro_agent import MacroCorrelationAgent
from .learning_agent import LearningOptimizationAgent
from .trading_system import AdvancedCryptoTradingSystem

__all__ = [
    'SentimentIntelligenceAgent',
    'TechnicalAnalysisAgent',
    'RiskManagementAgent',
    'RegulatoryComplianceAgent',
    'MacroCorrelationAgent',
    'LearningOptimizationAgent',
    'AdvancedCryptoTradingSystem',
]
