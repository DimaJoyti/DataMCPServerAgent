# DataMCPServerAgent Penetration Testing System

## 🎯 Overview

The DataMCPServerAgent Penetration Testing System is a comprehensive, AI-powered penetration testing platform that combines advanced LLM capabilities with industry-standard security tools and Bright Data's OSINT capabilities. This system provides automated, ethical, and legally compliant penetration testing with robust safety controls.

## ✨ Key Features

### 🧠 **LLM Integration & Optimization**
- **Multi-Model Support**: Claude Sonnet 4, local models, and API-based solutions
- **Fine-Tuned Models**: Specialized models trained on penetration testing data
- **Intelligent Reasoning**: Advanced decision-making for attack path planning
- **Cost Optimization**: Automatic fallback mechanisms for cost and latency optimization

### 🛠️ **Tool Control Layer**
- **Network Scanning**: Nmap integration with safety controls
- **Web Application Testing**: Burp Suite API integration (planned)
- **Exploitation Framework**: Metasploit integration (planned)
- **Custom Tools**: Extensible framework for custom security tools
- **Result Normalization**: Standardized output across all tools

### 🛡️ **Safety & Ethics Guard-Rails**
- **Target Validation**: Comprehensive authorization checking
- **Command Filtering**: Whitelist/blacklist command validation
- **Resource Monitoring**: CPU, memory, and network usage limits
- **Emergency Controls**: Kill switches and emergency stop mechanisms
- **Audit Logging**: Complete audit trail of all operations

### 🔍 **OSINT Capabilities (Bright Data)**
- **Social Media Intelligence**: LinkedIn, Twitter, Facebook, Instagram
- **Domain Intelligence**: Subdomain discovery, certificate transparency
- **Dark Web Monitoring**: Threat intelligence and credential monitoring
- **Company Intelligence**: Employee enumeration, technology stack analysis
- **Threat Intelligence**: IP/domain reputation, malware analysis

### 🏗️ **System Engineering**
- **Containerized Execution**: Docker-based isolation and security
- **Async Operations**: High-performance asynchronous processing
- **Scalable Architecture**: Horizontal scaling capabilities
- **Cloud Integration**: AWS and Cloudflare Workers support

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Docker** (for containerized execution)
3. **API Keys**:
   - Anthropic API key for Claude
   - Bright Data API token
4. **Authorization**: Written authorization for all targets

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
   cd DataMCPServerAgent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure penetration testing**:
   ```bash
   # Review and customize the configuration
   nano configs/pentest_config.yaml
   ```

### Basic Usage

1. **Run the interactive system**:
   ```bash
   python src/core/pentest_main.py
   ```

2. **Run the demo**:
   ```bash
   python examples/pentest_example.py
   ```

3. **Docker deployment**:
   ```bash
   cd docker/pentest
   docker-compose up -d
   ```

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Penetration Testing System                  │
├─────────────────────────────────────────────────────────────┤
│                    Safety Controller                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Target    │ │   Command   │ │  Resource   │           │
│  │ Validator   │ │   Filter    │ │  Monitor    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                 Pentest Coordinator                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │    Recon    │ │    Vuln     │ │   Exploit   │           │
│  │    Agent    │ │ Scan Agent  │ │    Agent    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                    Tool Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │    Nmap     │ │ Bright Data │ │   Custom    │           │
│  │   Toolkit   │ │    OSINT    │ │    Tools    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                  LLM Integration                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Claude    │ │    Local    │ │  Fallback   │           │
│  │   Sonnet    │ │   Models    │ │   Models    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## 🔒 Security & Safety

### Authorization Requirements

**⚠️ CRITICAL**: You MUST have explicit written authorization before testing any targets.

1. **Written Permission**: Signed authorization document
2. **Scope Definition**: Clear boundaries of what can be tested
3. **Time Windows**: Defined testing periods
4. **Emergency Contacts**: 24/7 contact information

### Safety Controls

1. **Target Validation**:
   - IP address and domain verification
   - Private network protection
   - Blacklist enforcement

2. **Command Filtering**:
   - Whitelist-only command execution
   - Dangerous command blocking
   - Argument validation

3. **Resource Limits**:
   - CPU and memory quotas
   - Network bandwidth limits
   - Concurrent operation limits

4. **Emergency Controls**:
   - Manual emergency stop
   - Automatic safety triggers
   - Session termination

## 🛠️ Configuration

### Main Configuration (`configs/pentest_config.yaml`)

```yaml
# Safety settings
safety:
  level: "high"  # low, medium, high, critical
  limits:
    max_concurrent_scans: 5
    max_scan_rate: 100
    max_session_duration: 3600

# Target validation
target_validation:
  require_authorization: true
  blocked_networks:
    - "127.0.0.0/8"
    - "169.254.0.0/16"

# Tool configuration
nmap:
  profiles:
    discovery: "-sn"
    quick: "-T4 -F"
    comprehensive: "-T4 -A -v"
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your_claude_api_key
BRIGHT_DATA_API_TOKEN=your_bright_data_token

# Optional
SAFETY_LEVEL=high
LOG_LEVEL=INFO
REDIS_PASSWORD=your_redis_password
MONGO_PASSWORD=your_mongo_password
```

## 📊 Usage Examples

### 1. Basic Reconnaissance

```python
from src.core.pentest_main import create_pentest_system

# Initialize system
coordinator = await create_pentest_system()

# Create session
session_id = await coordinator.create_pentest_session(
    target_name="Authorized Target",
    ip_addresses=["192.168.1.100"],
    domains=["target.example.com"],
    scope={"description": "Authorized penetration test"},
    authorization_token="AUTH_TOKEN_123"
)

# Execute reconnaissance
results = await coordinator.execute_pentest_phase(session_id, "reconnaissance")
```

### 2. OSINT Intelligence Gathering

```python
# Gather social media intelligence
social_intel = await bright_data_toolkit.social_media_intelligence(
    target_name="Target Company",
    platforms="linkedin"
)

# Domain intelligence
domain_intel = await bright_data_toolkit.domain_intelligence(
    domain="target.com",
    intel_type="comprehensive"
)
```

### 3. Network Scanning

```python
# Host discovery
discovery_results = await nmap_toolkit.host_discovery(
    target="192.168.1.0/24",
    scan_type="discovery"
)

# Port scanning
port_results = await nmap_toolkit.port_scan(
    target="192.168.1.100",
    ports="1-1000",
    scan_type="quick"
)
```

## 📈 Monitoring & Logging

### Log Files

- `logs/pentest_main.log` - Main application logs
- `logs/pentest_security.log` - Security events
- `logs/pentest_audit.log` - Audit trail
- `logs/pentest_errors.log` - Error logs

### Metrics

- Session statistics
- Tool performance
- Resource usage
- Safety violations

## 🔧 Extending the System

### Adding Custom Tools

1. **Create tool class**:
   ```python
   class CustomTool(BaseTool):
       name = "custom_tool"
       description = "Custom security tool"
       
       async def _run(self, target: str) -> str:
           # Tool implementation
           return results
   ```

2. **Register with coordinator**:
   ```python
   custom_tools = [CustomTool()]
   coordinator = PentestCoordinatorAgent(tools=custom_tools, ...)
   ```

### Adding New Agents

1. **Inherit from base agent**:
   ```python
   class CustomAgent:
       def __init__(self, model, tools, memory):
           self.model = model
           self.tools = tools
           self.memory = memory
       
       async def perform_custom_task(self, target):
           # Agent implementation
           return results
   ```

## 🚨 Legal & Ethical Considerations

### Legal Requirements

1. **Authorization**: Always obtain written permission
2. **Scope Compliance**: Stay within defined boundaries
3. **Data Protection**: Handle sensitive data appropriately
4. **Incident Response**: Have procedures for unexpected findings

### Ethical Guidelines

1. **Responsible Disclosure**: Report vulnerabilities responsibly
2. **Minimal Impact**: Avoid disrupting services
3. **Data Privacy**: Respect privacy and confidentiality
4. **Professional Conduct**: Maintain professional standards

## 🆘 Support & Troubleshooting

### Common Issues

1. **Permission Denied**: Check authorization and target validation
2. **Tool Failures**: Verify tool installation and configuration
3. **Network Issues**: Check connectivity and firewall rules
4. **Resource Limits**: Monitor system resources

### Getting Help

1. **Documentation**: Check the docs/ directory
2. **Examples**: Review examples/ directory
3. **Logs**: Check log files for detailed error information
4. **Issues**: Report bugs on GitHub

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for authorized penetration testing only. Users are responsible for ensuring they have proper authorization before testing any targets. The developers are not responsible for any misuse of this tool.
