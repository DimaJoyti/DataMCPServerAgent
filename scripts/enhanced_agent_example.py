"""
Enhanced Agent Example demonstrating all new capabilities:
- Persistent state handling with Cloudflare
- Long-running tasks and horizontal scaling
- Email APIs for human-in-the-loop workflows
- WebRTC for voice and video communication
- Self-hosting capabilities
"""

import asyncio
import logging
from datetime import datetime, timezone

# Import our enhanced integrations
from cloudflare_mcp_integration import (
    enhanced_cloudflare_integration, 
    AgentType, 
    TaskStatus
)
from email_integration import (
    email_integration, 
    EmailProvider, 
    ApprovalStatus
)
from webrtc_integration import (
    webrtc_integration, 
    CallDirection, 
    MediaType
)
from self_hosting_config import (
    self_hosting_manager, 
    Environment, 
    DeploymentType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAgentDemo:
    """Demonstration of enhanced agent capabilities."""
    
    def __init__(self):
        self.agent_id = "demo_agent_001"
        self.user_id = "user_001"
        self.approver_email = "admin@yourdomain.com"
        
    async def run_complete_demo(self):
        """Run complete demonstration of all enhanced features."""
        logger.info("🚀 Starting Enhanced Agent Demo")
        
        try:
            # 1. Persistent State Management
            await self.demo_persistent_state()
            
            # 2. Long-running Tasks
            await self.demo_long_running_tasks()
            
            # 3. Horizontal Scaling
            await self.demo_horizontal_scaling()
            
            # 4. Email Integration & Human-in-the-Loop
            await self.demo_email_integration()
            
            # 5. WebRTC Communication
            await self.demo_webrtc_integration()
            
            # 6. Self-hosting Configuration
            await self.demo_self_hosting()
            
            logger.info("✅ Enhanced Agent Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise

    async def demo_persistent_state(self):
        """Demonstrate persistent state management."""
        logger.info("\n📦 === PERSISTENT STATE MANAGEMENT ===")
        
        # Save agent state
        state_data = {
            "conversation_history": [
                {"role": "user", "content": "Hello, I need help with my account"},
                {"role": "assistant", "content": "I'd be happy to help you with your account. What specific issue are you experiencing?"}
            ],
            "user_preferences": {
                "language": "en",
                "timezone": "UTC",
                "notification_email": self.approver_email
            },
            "context": {
                "current_task": "account_assistance",
                "session_start": datetime.now(timezone.utc).isoformat()
            }
        }
        
        success = await enhanced_cloudflare_integration.save_agent_state(
            self.agent_id, 
            AgentType.ANALYTICS, 
            state_data
        )
        logger.info(f"💾 State saved: {success}")
        
        # Load agent state
        loaded_state = await enhanced_cloudflare_integration.load_agent_state(self.agent_id)
        if loaded_state:
            logger.info(f"📖 State loaded: version {loaded_state.version}")
            logger.info(f"📝 Conversation history: {len(loaded_state.state_data['conversation_history'])} messages")

    async def demo_long_running_tasks(self):
        """Demonstrate long-running task management."""
        logger.info("\n⏳ === LONG-RUNNING TASKS ===")
        
        # Create a long-running task
        task_id = await enhanced_cloudflare_integration.create_long_running_task(
            agent_id=self.agent_id,
            task_type="data_analysis",
            metadata={
                "dataset_size": "10GB",
                "analysis_type": "sentiment_analysis",
                "estimated_duration": "30 minutes"
            }
        )
        logger.info(f"📋 Created task: {task_id}")
        
        # Simulate task progress
        progress_steps = [10, 25, 50, 75, 90, 100]
        for progress in progress_steps:
            await enhanced_cloudflare_integration.update_task_progress(
                task_id, 
                progress, 
                TaskStatus.RUNNING if progress < 100 else TaskStatus.COMPLETED
            )
            logger.info(f"📊 Task progress: {progress}%")
            await asyncio.sleep(0.5)  # Simulate processing time
        
        # Complete the task
        result = {
            "sentiment_scores": {
                "positive": 0.65,
                "neutral": 0.25,
                "negative": 0.10
            },
            "total_processed": 50000,
            "processing_time": "28 minutes"
        }
        
        await enhanced_cloudflare_integration.complete_task(task_id, result)
        logger.info(f"✅ Task completed with results")

    async def demo_horizontal_scaling(self):
        """Demonstrate horizontal scaling capabilities."""
        logger.info("\n📈 === HORIZONTAL SCALING ===")
        
        # Scale agent to 3 instances
        scaling_result = await enhanced_cloudflare_integration.scale_agent_horizontally(
            self.agent_id, 
            target_instances=3
        )
        logger.info(f"🔄 Scaling result: {scaling_result['current_instances']} instances")
        
        # Get load metrics
        load_metrics = await enhanced_cloudflare_integration.get_agent_load_metrics(self.agent_id)
        logger.info(f"📊 Load metrics: {load_metrics['total_instances']} instances")
        logger.info(f"📊 Average load: {load_metrics['average_load']:.2f}")
        logger.info(f"💡 Recommendation: {load_metrics['scaling_recommendation']}")

    async def demo_email_integration(self):
        """Demonstrate email integration and human-in-the-loop workflows."""
        logger.info("\n📧 === EMAIL INTEGRATION ===")
        
        # Create approval request
        approval_id = await email_integration.create_approval_request(
            agent_id=self.agent_id,
            task_id="sensitive_data_access",
            title="Access Sensitive Customer Data",
            description="The agent needs to access sensitive customer financial data to complete the analysis. This requires human approval for compliance reasons.",
            data={
                "customer_id": "CUST_12345",
                "data_type": "financial_records",
                "purpose": "fraud_detection_analysis"
            },
            approver_email=self.approver_email,
            expires_in_hours=24
        )
        logger.info(f"📋 Approval request created: {approval_id}")
        
        # Simulate approval process
        await asyncio.sleep(1)  # Simulate time for human to review
        
        # Process approval (simulate human clicking approve)
        approval_success = await email_integration.process_approval_response(
            approval_id, 
            "approve", 
            self.approver_email,
            "Approved for fraud detection analysis"
        )
        logger.info(f"✅ Approval processed: {approval_success}")
        
        # Check approval status
        approval_status = await email_integration.get_approval_status(approval_id)
        if approval_status:
            logger.info(f"📊 Approval status: {approval_status.status.value}")

    async def demo_webrtc_integration(self):
        """Demonstrate WebRTC voice and video capabilities."""
        logger.info("\n🎥 === WEBRTC INTEGRATION ===")
        
        # Create call session
        call_id = await webrtc_integration.create_call_session(
            agent_id=self.agent_id,
            user_id=self.user_id,
            direction=CallDirection.INBOUND,
            enable_recording=True,
            enable_video=True
        )
        logger.info(f"📞 Call session created: {call_id}")
        
        # Start the call
        call_started = await webrtc_integration.start_call(call_id)
        logger.info(f"🔊 Call started: {call_started}")
        
        # Simulate voice-to-text processing
        voice_result = await webrtc_integration.process_voice_to_text(
            call_id, 
            b"mock_audio_data",
            participant_id="user_participant"
        )
        logger.info(f"🎤 Voice-to-text: {voice_result.text}")
        
        # Simulate agent response via text-to-speech
        await webrtc_integration.play_speech_in_call(
            call_id,
            "Thank you for calling. I understand you need help with your account. Let me assist you with that.",
            voice="en-US-Neural2-A"
        )
        logger.info(f"🔊 Agent responded via speech")
        
        # Simulate call duration
        await asyncio.sleep(2)
        
        # End the call
        call_ended = await webrtc_integration.end_call(call_id)
        logger.info(f"📞 Call ended: {call_ended}")
        
        # Get call session details
        call_session = await webrtc_integration.get_call_session(call_id)
        if call_session:
            logger.info(f"📊 Call duration: {call_session.duration_seconds} seconds")

    async def demo_self_hosting(self):
        """Demonstrate self-hosting configuration generation."""
        logger.info("\n🏠 === SELF-HOSTING CONFIGURATION ===")
        
        # Generate deployment files for different environments
        environments = ["development", "production"]
        
        for env in environments:
            logger.info(f"📦 Generating {env} deployment files...")
            
            # Save deployment files
            output_dir = f"./deployment_{env}"
            self_hosting_manager.save_deployment_files(env, output_dir)
            
            # Get deployment config
            config = self_hosting_manager.get_deployment_config(env)
            if config:
                logger.info(f"🔧 {env.title()} config:")
                logger.info(f"   - Deployment type: {config.deployment_type.value}")
                logger.info(f"   - Services: {len(config.services)}")
                logger.info(f"   - Databases: {len(config.databases)}")
                logger.info(f"   - Ingress enabled: {config.ingress_config.get('enabled', False)}")
        
        logger.info("📁 Deployment files generated successfully")

async def main():
    """Main function to run the enhanced agent demo."""
    demo = EnhancedAgentDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())
