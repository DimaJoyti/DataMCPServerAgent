"""
Risk Management Agent for the Fetch.ai Advanced Crypto Trading System.

This agent implements position sizing, stop-loss calculation, and a curve.fi-inspired
soft liquidation process.
"""

import asyncio
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from uagents import Agent, Context, Model, Protocol

from .base_agent import BaseAgent, BaseAgentState

class RiskLevel(str, Enum):
    """Risk levels for trading."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class PositionSizing(Model):
    """Model for position sizing recommendations."""
    
    symbol: str
    account_balance: float
    recommended_position_size: float
    max_position_size: float
    risk_level: RiskLevel
    risk_percentage: float
    timestamp: str

class StopLossRecommendation(Model):
    """Model for stop-loss recommendations."""
    
    symbol: str
    entry_price: float
    position_size: float
    stop_loss_price: float
    take_profit_price: Optional[float] = None
    risk_reward_ratio: float
    max_loss_amount: float
    max_loss_percentage: float
    timestamp: str

class LiquidationInfo(Model):
    """Model for liquidation information."""
    
    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    leverage: float
    liquidation_price: float
    distance_to_liquidation_percentage: float
    soft_liquidation_threshold: float
    soft_liquidation_recommended: bool
    timestamp: str

class RiskAssessment(Model):
    """Model for overall risk assessment."""
    
    symbol: str
    position_sizing: PositionSizing
    stop_loss: StopLossRecommendation
    liquidation: Optional[LiquidationInfo] = None
    overall_risk_level: RiskLevel
    timestamp: str

class RiskAgentState(BaseAgentState):
    """State model for the Risk Management Agent."""
    
    default_risk_percentage: float = 0.02  # 2% of account balance
    max_risk_percentage: float = 0.05  # 5% of account balance
    default_risk_reward_ratio: float = 2.0  # 1:2 risk-reward ratio
    soft_liquidation_threshold: float = 0.8  # 80% of distance to liquidation
    symbols_to_track: List[str] = ["BTC/USD", "ETH/USD"]
    recent_assessments: List[RiskAssessment] = []

class RiskManagementAgent(BaseAgent):
    """Agent for managing trading risk."""
    
    def __init__(
        self,
        name: str = "risk_agent",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Risk Management Agent.
        
        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
        """
        super().__init__(name, seed, port, endpoint, logger)
        
        # Initialize agent state
        self.state = RiskAgentState()
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for the agent."""
        
        @self.agent.on_message(model=dict)
        async def handle_trade_request(ctx: Context, sender: str, msg: Dict[str, Any]):
            """Handle trade request from other agents."""
            if "type" in msg and msg["type"] == "trade_request":
                ctx.logger.info(f"Received trade request from {sender}")
                
                # Extract trade parameters
                symbol = msg.get("symbol")
                entry_price = msg.get("entry_price")
                account_balance = msg.get("account_balance", 10000.0)  # Default if not provided
                leverage = msg.get("leverage", 1.0)  # Default to no leverage
                
                if not symbol or not entry_price:
                    ctx.logger.warning("Invalid trade request: missing symbol or entry_price")
                    return
                
                # Perform risk assessment
                assessment = await self._assess_risk(
                    symbol, float(entry_price), float(account_balance), float(leverage)
                )
                
                # Send response
                await ctx.send(sender, assessment.dict())
    
    async def _assess_risk(
        self, 
        symbol: str, 
        entry_price: float, 
        account_balance: float,
        leverage: float = 1.0
    ) -> RiskAssessment:
        """Assess risk for a potential trade.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            account_balance: Account balance
            leverage: Leverage multiplier
            
        Returns:
            Risk assessment
        """
        # Calculate position sizing
        position_sizing = self._calculate_position_sizing(
            symbol, entry_price, account_balance, leverage
        )
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            symbol, entry_price, position_sizing.recommended_position_size
        )
        
        # Calculate liquidation info if using leverage
        liquidation_info = None
        if leverage > 1.0:
            liquidation_info = self._calculate_liquidation_info(
                symbol, entry_price, position_sizing.recommended_position_size, leverage
            )
        
        # Determine overall risk level
        overall_risk_level = self._determine_overall_risk_level(
            position_sizing, stop_loss, liquidation_info
        )
        
        # Create assessment
        assessment = RiskAssessment(
            symbol=symbol,
            position_sizing=position_sizing,
            stop_loss=stop_loss,
            liquidation=liquidation_info,
            overall_risk_level=overall_risk_level,
            timestamp=datetime.now().isoformat()
        )
        
        # Update state
        self.state.recent_assessments.append(assessment)
        if len(self.state.recent_assessments) > 10:
            self.state.recent_assessments.pop(0)
        
        return assessment
    
    def _calculate_position_sizing(
        self, 
        symbol: str, 
        entry_price: float, 
        account_balance: float,
        leverage: float = 1.0
    ) -> PositionSizing:
        """Calculate position sizing based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            account_balance: Account balance
            leverage: Leverage multiplier
            
        Returns:
            Position sizing recommendation
        """
        # Determine risk level based on market conditions
        # This would typically use market data and volatility
        risk_level = RiskLevel.MEDIUM
        
        # Adjust risk percentage based on risk level
        if risk_level == RiskLevel.LOW:
            risk_percentage = self.state.default_risk_percentage * 0.5
        elif risk_level == RiskLevel.MEDIUM:
            risk_percentage = self.state.default_risk_percentage
        else:  # HIGH
            risk_percentage = self.state.default_risk_percentage * 1.5
        
        # Ensure risk percentage doesn't exceed maximum
        risk_percentage = min(risk_percentage, self.state.max_risk_percentage)
        
        # Calculate maximum risk amount
        max_risk_amount = account_balance * risk_percentage
        
        # Calculate position size based on risk amount and stop loss distance
        # For simplicity, we'll use a fixed 5% stop loss distance
        stop_loss_percentage = 0.05
        stop_loss_distance = entry_price * stop_loss_percentage
        
        # Calculate position size
        position_size = max_risk_amount / stop_loss_distance
        
        # Adjust for leverage
        position_size = position_size * leverage
        
        # Calculate maximum position size (50% of account balance)
        max_position_size = (account_balance * 0.5) / entry_price * leverage
        
        # Ensure position size doesn't exceed maximum
        recommended_position_size = min(position_size, max_position_size)
        
        return PositionSizing(
            symbol=symbol,
            account_balance=account_balance,
            recommended_position_size=recommended_position_size,
            max_position_size=max_position_size,
            risk_level=risk_level,
            risk_percentage=risk_percentage,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_stop_loss(
        self, 
        symbol: str, 
        entry_price: float, 
        position_size: float
    ) -> StopLossRecommendation:
        """Calculate stop loss recommendation.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_size: Position size
            
        Returns:
            Stop loss recommendation
        """
        # For simplicity, we'll use a fixed 5% stop loss
        stop_loss_percentage = 0.05
        stop_loss_price = entry_price * (1 - stop_loss_percentage)
        
        # Calculate take profit based on risk-reward ratio
        take_profit_price = entry_price * (1 + (stop_loss_percentage * self.state.default_risk_reward_ratio))
        
        # Calculate maximum loss
        max_loss_amount = position_size * (entry_price - stop_loss_price)
        max_loss_percentage = (entry_price - stop_loss_price) / entry_price
        
        return StopLossRecommendation(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_reward_ratio=self.state.default_risk_reward_ratio,
            max_loss_amount=max_loss_amount,
            max_loss_percentage=max_loss_percentage,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_liquidation_info(
        self, 
        symbol: str, 
        entry_price: float, 
        position_size: float,
        leverage: float
    ) -> LiquidationInfo:
        """Calculate liquidation information.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_size: Position size
            leverage: Leverage multiplier
            
        Returns:
            Liquidation information
        """
        # Calculate liquidation price
        # This is a simplified calculation; actual liquidation price depends on
        # the exchange's specific formula and maintenance margin requirements
        maintenance_margin = 0.05  # 5% maintenance margin
        liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
        
        # Simulate current price (for demonstration)
        current_price = entry_price * 0.98  # 2% below entry
        
        # Calculate distance to liquidation
        distance_to_liquidation = current_price - liquidation_price
        distance_percentage = distance_to_liquidation / current_price
        
        # Determine if soft liquidation is recommended
        soft_liquidation_threshold = distance_percentage * self.state.soft_liquidation_threshold
        soft_liquidation_recommended = distance_percentage < soft_liquidation_threshold
        
        return LiquidationInfo(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            leverage=leverage,
            liquidation_price=liquidation_price,
            distance_to_liquidation_percentage=distance_percentage,
            soft_liquidation_threshold=soft_liquidation_threshold,
            soft_liquidation_recommended=soft_liquidation_recommended,
            timestamp=datetime.now().isoformat()
        )
    
    def _determine_overall_risk_level(
        self,
        position_sizing: PositionSizing,
        stop_loss: StopLossRecommendation,
        liquidation_info: Optional[LiquidationInfo]
    ) -> RiskLevel:
        """Determine overall risk level.
        
        Args:
            position_sizing: Position sizing recommendation
            stop_loss: Stop loss recommendation
            liquidation_info: Liquidation information
            
        Returns:
            Overall risk level
        """
        # Start with position sizing risk level
        risk_level = position_sizing.risk_level
        
        # Adjust based on stop loss
        if stop_loss.max_loss_percentage > 0.1:  # More than 10% loss
            risk_level = RiskLevel.HIGH
        
        # Adjust based on liquidation info
        if liquidation_info:
            if liquidation_info.soft_liquidation_recommended:
                risk_level = RiskLevel.HIGH
            elif liquidation_info.leverage > 5:
                risk_level = RiskLevel.HIGH
            elif liquidation_info.leverage > 2:
                risk_level = max(RiskLevel.MEDIUM, risk_level)
        
        return risk_level
