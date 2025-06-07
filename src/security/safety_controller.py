"""
Safety Controller for Penetration Testing

This module provides comprehensive safety controls and ethical guidelines
for penetration testing operations, ensuring responsible and legal testing.
"""

import asyncio
import logging
import ipaddress
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .target_validator import TargetValidator, ValidationResult
from .command_filter import CommandFilter
from .audit_logger import AuditLogger
from .resource_monitor import ResourceMonitor


class SafetyLevel(Enum):
    """Safety levels for penetration testing operations"""
    LOW = "low"           # Basic safety checks
    MEDIUM = "medium"     # Standard safety checks
    HIGH = "high"         # Strict safety checks
    CRITICAL = "critical" # Maximum safety checks


@dataclass
class SafetyCheck:
    """Represents a safety check result"""
    approved: bool
    reason: str
    safety_level: SafetyLevel
    timestamp: datetime
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class SafetyLimits:
    """Safety limits for penetration testing operations"""
    max_concurrent_scans: int = 5
    max_scan_rate: int = 100  # packets per second
    max_session_duration: int = 3600  # seconds
    max_memory_usage: int = 1024  # MB
    max_cpu_usage: float = 0.8  # 80%
    max_network_bandwidth: int = 10  # Mbps
    allowed_ports: Set[int] = None
    blocked_ports: Set[int] = None
    
    def __post_init__(self):
        if self.allowed_ports is None:
            # Common safe ports for testing
            self.allowed_ports = {
                21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995,
                8080, 8443, 3389, 5432, 3306, 1433, 27017
            }
        
        if self.blocked_ports is None:
            # Critical system ports to avoid
            self.blocked_ports = {
                0, 1, 7, 9, 13, 17, 19, 20, 37, 42, 43, 49, 135, 136, 137, 138, 139, 445
            }


class SafetyController:
    """
    Comprehensive safety controller for penetration testing operations
    
    This controller ensures that all penetration testing activities are:
    1. Legally authorized
    2. Within defined scope
    3. Ethically conducted
    4. Resource-limited
    5. Properly audited
    """
    
    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.HIGH,
        safety_limits: Optional[SafetyLimits] = None,
        target_validator: Optional[TargetValidator] = None,
        command_filter: Optional[CommandFilter] = None,
        audit_logger: Optional[AuditLogger] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        self.safety_level = safety_level
        self.safety_limits = safety_limits or SafetyLimits()
        self.target_validator = target_validator or TargetValidator()
        self.command_filter = command_filter or CommandFilter()
        self.audit_logger = audit_logger or AuditLogger()
        self.resource_monitor = resource_monitor or ResourceMonitor()
        
        self.logger = logging.getLogger(__name__)
        
        # Active sessions tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.emergency_stop_triggered = False
        
        # Rate limiting
        self.scan_history: List[datetime] = []
        
        # Blocked targets (emergency blacklist)
        self.blocked_targets: Set[str] = set()
    
    async def pre_phase_check(self, session, phase: str) -> SafetyCheck:
        """
        Perform comprehensive safety check before starting a testing phase
        
        Args:
            session: PentestSession object
            phase: Testing phase to validate
            
        Returns:
            SafetyCheck result
        """
        timestamp = datetime.now()
        
        # Check if emergency stop is active
        if self.emergency_stop_triggered:
            return SafetyCheck(
                approved=False,
                reason="Emergency stop is active",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Validate target authorization
        target_validation = await self.target_validator.validate_target(session.target)
        if not target_validation.is_valid:
            return SafetyCheck(
                approved=False,
                reason=f"Target validation failed: {target_validation.reason}",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Check if target is blocked
        for ip in session.target.ip_addresses:
            if ip in self.blocked_targets:
                return SafetyCheck(
                    approved=False,
                    reason=f"Target {ip} is in emergency blacklist",
                    safety_level=self.safety_level,
                    timestamp=timestamp
                )
        
        # Phase-specific checks
        phase_check = await self._validate_phase(session, phase)
        if not phase_check.approved:
            return phase_check
        
        # Resource checks
        resource_check = await self._check_resources()
        if not resource_check.approved:
            return resource_check
        
        # Rate limiting check
        rate_check = await self._check_rate_limits()
        if not rate_check.approved:
            return rate_check
        
        # Log the approval
        await self.audit_logger.log_safety_check(session.session_id, phase, True, "Approved")
        
        return SafetyCheck(
            approved=True,
            reason="All safety checks passed",
            safety_level=self.safety_level,
            timestamp=timestamp,
            additional_info={
                "session_id": session.session_id,
                "phase": phase,
                "target": session.target.name
            }
        )
    
    async def validate_target(self, target: str) -> SafetyCheck:
        """
        Validate a target for penetration testing
        
        Args:
            target: Target IP address, hostname, or range
            
        Returns:
            SafetyCheck result
        """
        timestamp = datetime.now()
        
        # Check if target is blocked
        if target in self.blocked_targets:
            return SafetyCheck(
                approved=False,
                reason=f"Target {target} is in emergency blacklist",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Validate IP address/range
        try:
            # Try to parse as IP address or network
            if '/' in target:
                network = ipaddress.ip_network(target, strict=False)
                # Check for private networks
                if network.is_private:
                    if self.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
                        return SafetyCheck(
                            approved=False,
                            reason="Private network scanning requires explicit authorization",
                            safety_level=self.safety_level,
                            timestamp=timestamp
                        )
            else:
                ip = ipaddress.ip_address(target)
                # Check for localhost, private, or reserved addresses
                if ip.is_loopback or ip.is_private or ip.is_reserved:
                    if self.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
                        return SafetyCheck(
                            approved=False,
                            reason="Special IP address requires explicit authorization",
                            safety_level=self.safety_level,
                            timestamp=timestamp
                        )
        except ValueError:
            # Not an IP address, might be hostname
            if self.safety_level == SafetyLevel.CRITICAL:
                return SafetyCheck(
                    approved=False,
                    reason="Hostname targets require explicit authorization in critical mode",
                    safety_level=self.safety_level,
                    timestamp=timestamp
                )
        
        # Additional target validation through TargetValidator
        validation_result = await self.target_validator.validate_target_string(target)
        if not validation_result.is_valid:
            return SafetyCheck(
                approved=False,
                reason=validation_result.reason,
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        return SafetyCheck(
            approved=True,
            reason="Target validation passed",
            safety_level=self.safety_level,
            timestamp=timestamp
        )
    
    async def validate_command(self, command: str, context: Dict[str, Any] = None) -> SafetyCheck:
        """
        Validate a command before execution
        
        Args:
            command: Command to validate
            context: Additional context for validation
            
        Returns:
            SafetyCheck result
        """
        timestamp = datetime.now()
        
        # Use command filter
        filter_result = await self.command_filter.validate_command(command, context)
        
        if not filter_result.allowed:
            return SafetyCheck(
                approved=False,
                reason=f"Command blocked: {filter_result.reason}",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Log command validation
        await self.audit_logger.log_command_validation(command, True, context)
        
        return SafetyCheck(
            approved=True,
            reason="Command validation passed",
            safety_level=self.safety_level,
            timestamp=timestamp
        )
    
    async def emergency_stop(self, session_id: str, reason: str):
        """
        Trigger emergency stop for all operations
        
        Args:
            session_id: Session that triggered the stop
            reason: Reason for emergency stop
        """
        self.emergency_stop_triggered = True
        
        # Log emergency stop
        await self.audit_logger.log_emergency_stop(session_id, reason)
        
        # Stop all active sessions
        for active_session_id in list(self.active_sessions.keys()):
            await self._stop_session(active_session_id, "Emergency stop triggered")
        
        self.logger.critical(f"EMERGENCY STOP triggered by session {session_id}: {reason}")
    
    async def add_blocked_target(self, target: str, reason: str):
        """
        Add target to emergency blacklist
        
        Args:
            target: Target to block
            reason: Reason for blocking
        """
        self.blocked_targets.add(target)
        await self.audit_logger.log_target_blocked(target, reason)
        self.logger.warning(f"Target {target} added to blacklist: {reason}")
    
    async def remove_blocked_target(self, target: str):
        """
        Remove target from emergency blacklist
        
        Args:
            target: Target to unblock
        """
        self.blocked_targets.discard(target)
        await self.audit_logger.log_target_unblocked(target)
        self.logger.info(f"Target {target} removed from blacklist")
    
    async def reset_emergency_stop(self):
        """Reset emergency stop state"""
        self.emergency_stop_triggered = False
        await self.audit_logger.log_emergency_reset()
        self.logger.info("Emergency stop reset")
    
    async def _validate_phase(self, session, phase: str) -> SafetyCheck:
        """Validate specific testing phase"""
        timestamp = datetime.now()
        
        # Check session duration
        session_duration = (timestamp - session.start_time).total_seconds()
        if session_duration > self.safety_limits.max_session_duration:
            return SafetyCheck(
                approved=False,
                reason=f"Session duration exceeded limit ({self.safety_limits.max_session_duration}s)",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Phase-specific validations
        if phase == "exploitation":
            if self.safety_level == SafetyLevel.CRITICAL:
                return SafetyCheck(
                    approved=False,
                    reason="Exploitation phase blocked in critical safety mode",
                    safety_level=self.safety_level,
                    timestamp=timestamp
                )
        
        return SafetyCheck(
            approved=True,
            reason="Phase validation passed",
            safety_level=self.safety_level,
            timestamp=timestamp
        )
    
    async def _check_resources(self) -> SafetyCheck:
        """Check system resource usage"""
        timestamp = datetime.now()
        
        resource_status = await self.resource_monitor.get_current_usage()
        
        # Check memory usage
        if resource_status.memory_usage_mb > self.safety_limits.max_memory_usage:
            return SafetyCheck(
                approved=False,
                reason=f"Memory usage exceeded limit ({resource_status.memory_usage_mb}MB > {self.safety_limits.max_memory_usage}MB)",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Check CPU usage
        if resource_status.cpu_usage > self.safety_limits.max_cpu_usage:
            return SafetyCheck(
                approved=False,
                reason=f"CPU usage exceeded limit ({resource_status.cpu_usage:.1%} > {self.safety_limits.max_cpu_usage:.1%})",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        return SafetyCheck(
            approved=True,
            reason="Resource check passed",
            safety_level=self.safety_level,
            timestamp=timestamp
        )
    
    async def _check_rate_limits(self) -> SafetyCheck:
        """Check rate limiting"""
        timestamp = datetime.now()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = timestamp - timedelta(minutes=1)
        self.scan_history = [t for t in self.scan_history if t > cutoff_time]
        
        # Check rate limit
        if len(self.scan_history) >= self.safety_limits.max_scan_rate:
            return SafetyCheck(
                approved=False,
                reason=f"Scan rate limit exceeded ({len(self.scan_history)} > {self.safety_limits.max_scan_rate} per minute)",
                safety_level=self.safety_level,
                timestamp=timestamp
            )
        
        # Add current scan to history
        self.scan_history.append(timestamp)
        
        return SafetyCheck(
            approved=True,
            reason="Rate limit check passed",
            safety_level=self.safety_level,
            timestamp=timestamp
        )
    
    async def _stop_session(self, session_id: str, reason: str):
        """Stop a specific session"""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info["status"] = "stopped"
            session_info["stop_reason"] = reason
            session_info["stop_time"] = datetime.now()
            
            await self.audit_logger.log_session_stopped(session_id, reason)
            self.logger.info(f"Session {session_id} stopped: {reason}")
