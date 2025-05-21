"""
Scheduled reporting tools for SEO.

This module provides tools for scheduling and generating regular SEO reports.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import requests
import schedule
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import pandas as pd
import matplotlib.pyplot as plt
from langchain.tools import Tool

from src.tools.seo_bulk_tools import BulkAnalysisTool
from src.tools.seo_advanced_tools import RankTrackingTool
from src.tools.seo_ml_tools import MLRankingPredictionTool

# Directory for storing scheduled reports
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "seo")
os.makedirs(REPORTS_DIR, exist_ok=True)


class ScheduledReportingTool:
    """Tool for scheduling and generating regular SEO reports."""
    
    def __init__(self):
        """Initialize the scheduled reporting tool."""
        self.bulk_analysis = BulkAnalysisTool()
        self.rank_tracking = RankTrackingTool()
        self.ml_ranking = MLRankingPredictionTool()
        self.scheduled_reports = {}
        self.scheduler_thread = None
        self.running = False
    
    def schedule_report(self, domain: str, frequency: str, report_type: str, email: Optional[str] = None) -> Dict[str, Any]:
        """
        Schedule a regular SEO report.
        
        Args:
            domain: The domain to generate reports for
            frequency: Report frequency ('daily', 'weekly', 'monthly')
            report_type: Type of report ('basic', 'comprehensive')
            email: Optional email address to send reports to
            
        Returns:
            Dictionary with scheduling details
        """
        print(f"Scheduling {report_type} SEO report for {domain} with {frequency} frequency...")
        
        # Generate a unique ID for the report
        report_id = hashlib.md5(f"{domain}_{frequency}_{report_type}_{time.time()}".encode()).hexdigest()
        
        # Determine the schedule
        if frequency == "daily":
            schedule_time = "00:00"  # Midnight
        elif frequency == "weekly":
            schedule_time = "Monday 00:00"  # Monday at midnight
        elif frequency == "monthly":
            schedule_time = "1st 00:00"  # 1st of the month at midnight
        else:
            return {"error": f"Invalid frequency: {frequency}. Must be 'daily', 'weekly', or 'monthly'."}
        
        # Store the report configuration
        self.scheduled_reports[report_id] = {
            "domain": domain,
            "frequency": frequency,
            "report_type": report_type,
            "email": email,
            "schedule_time": schedule_time,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": self._calculate_next_run(frequency)
        }
        
        # Start the scheduler if not already running
        if not self.running:
            self._start_scheduler()
        
        return {
            "report_id": report_id,
            "domain": domain,
            "frequency": frequency,
            "report_type": report_type,
            "email": email,
            "schedule_time": schedule_time,
            "next_run": self.scheduled_reports[report_id]["next_run"].isoformat()
        }
    
    def _calculate_next_run(self, frequency: str) -> datetime:
        """
        Calculate the next run time based on frequency.
        
        Args:
            frequency: Report frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            Datetime of the next run
        """
        now = datetime.now()
        
        if frequency == "daily":
            # Next day at midnight
            next_run = datetime(now.year, now.month, now.day) + timedelta(days=1)
        elif frequency == "weekly":
            # Next Monday at midnight
            days_until_monday = 7 - now.weekday() if now.weekday() > 0 else 7
            next_run = datetime(now.year, now.month, now.day) + timedelta(days=days_until_monday)
        elif frequency == "monthly":
            # 1st of next month at midnight
            if now.month == 12:
                next_run = datetime(now.year + 1, 1, 1)
            else:
                next_run = datetime(now.year, now.month + 1, 1)
        else:
            # Default to tomorrow
            next_run = datetime(now.year, now.month, now.day) + timedelta(days=1)
        
        return next_run
    
    def _start_scheduler(self) -> None:
        """Start the scheduler thread."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        while self.running:
            # Check for reports that need to be run
            now = datetime.now()
            
            for report_id, report in list(self.scheduled_reports.items()):
                if report["next_run"] <= now:
                    # Generate and send the report
                    self._generate_report(report_id)
                    
                    # Update the last run time and calculate the next run
                    self.scheduled_reports[report_id]["last_run"] = now.isoformat()
                    self.scheduled_reports[report_id]["next_run"] = self._calculate_next_run(report["frequency"])
            
            # Sleep for a minute before checking again
            time.sleep(60)
    
    def _generate_report(self, report_id: str) -> None:
        """
        Generate a scheduled report.
        
        Args:
            report_id: ID of the report to generate
        """
        report = self.scheduled_reports.get(report_id)
        if not report:
            print(f"Report {report_id} not found")
            return
        
        domain = report["domain"]
        report_type = report["report_type"]
        
        print(f"Generating {report_type} SEO report for {domain}...")
        
        try:
            # Generate the report based on type
            if report_type == "basic":
                result = self.bulk_analysis.analyze_site(domain, max_pages=10, depth="basic")
            else:  # comprehensive
                result = self.bulk_analysis.analyze_site(domain, max_pages=50, depth="comprehensive")
            
            # Add ranking data
            ranking_data = self.rank_tracking.track_rankings(domain)
            result["rankings"] = ranking_data
            
            # Save the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(REPORTS_DIR, f"{domain.replace('.', '_')}_{report_type}_{timestamp}.json")
            
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Generate visualizations
            visualization_path = self._generate_visualizations(result, domain, timestamp)
            
            # Send the report by email if an email address is provided
            if report["email"]:
                self._send_report_email(report["email"], domain, report_type, report_path, visualization_path)
            
            print(f"Report generated and saved to {report_path}")
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
    
    def _generate_visualizations(self, report_data: Dict[str, Any], domain: str, timestamp: str) -> str:
        """
        Generate visualizations for the report.
        
        Args:
            report_data: Report data
            domain: Domain name
            timestamp: Timestamp string
            
        Returns:
            Path to the visualization file
        """
        # Create a directory for visualizations
        vis_dir = os.path.join(REPORTS_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create a PDF file for visualizations
        vis_path = os.path.join(vis_dir, f"{domain.replace('.', '_')}_{timestamp}_visualizations.pdf")
        
        # Create visualizations using matplotlib
        plt.figure(figsize=(12, 8))
        
        # SEO Score Distribution
        plt.subplot(2, 2, 1)
        scores = [page.get("seo_score", 0) for page in report_data.get("pages", [])]
        plt.hist(scores, bins=10, range=(0, 100), alpha=0.7, color='blue')
        plt.title('SEO Score Distribution')
        plt.xlabel('SEO Score')
        plt.ylabel('Number of Pages')
        
        # Issues by Category
        plt.subplot(2, 2, 2)
        issue_categories = {}
        for page in report_data.get("pages", []):
            for issue in page.get("issues", []):
                category = issue.get("category", "Other")
                issue_categories[category] = issue_categories.get(category, 0) + 1
        
        categories = list(issue_categories.keys())
        counts = list(issue_categories.values())
        plt.bar(categories, counts, color='red', alpha=0.7)
        plt.title('Issues by Category')
        plt.xlabel('Category')
        plt.ylabel('Number of Issues')
        plt.xticks(rotation=45, ha='right')
        
        # Keyword Rankings
        plt.subplot(2, 2, 3)
        rankings = report_data.get("rankings", {}).get("rankings", [])
        keywords = [r.get("keyword", "") for r in rankings]
        positions = [r.get("current_rank", 0) for r in rankings]
        
        plt.bar(keywords, positions, color='green', alpha=0.7)
        plt.title('Keyword Rankings')
        plt.xlabel('Keyword')
        plt.ylabel('Position')
        plt.xticks(rotation=45, ha='right')
        plt.gca().invert_yaxis()  # Invert Y-axis so lower (better) rankings are higher
        
        # Page Speed
        plt.subplot(2, 2, 4)
        mobile_speeds = [page.get("page_speed", {}).get("mobile", 0) for page in report_data.get("pages", [])]
        desktop_speeds = [page.get("page_speed", {}).get("desktop", 0) for page in report_data.get("pages", [])]
        
        if mobile_speeds and desktop_speeds:
            labels = ['Mobile', 'Desktop']
            speeds = [sum(mobile_speeds) / len(mobile_speeds), sum(desktop_speeds) / len(desktop_speeds)]
            plt.bar(labels, speeds, color='purple', alpha=0.7)
            plt.title('Average Page Speed')
            plt.xlabel('Device Type')
            plt.ylabel('Speed Score')
            plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close()
        
        return vis_path
    
    def _send_report_email(self, email: str, domain: str, report_type: str, report_path: str, visualization_path: str) -> None:
        """
        Send a report by email.
        
        Args:
            email: Email address to send the report to
            domain: Domain name
            report_type: Type of report
            report_path: Path to the report file
            visualization_path: Path to the visualization file
        """
        # Email configuration
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        if not smtp_username or not smtp_password:
            print("SMTP credentials not configured. Email not sent.")
            return
        
        try:
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = smtp_username
            msg['To'] = email
            msg['Subject'] = f"SEO Report for {domain} - {report_type.capitalize()} - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h1>SEO Report for {domain}</h1>
                <p>Please find attached your {report_type} SEO report for {domain}.</p>
                <p>This report was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
                <p>The report includes:</p>
                <ul>
                    <li>SEO analysis of your website</li>
                    <li>Keyword rankings</li>
                    <li>Issues and recommendations</li>
                    <li>Visualizations of key metrics</li>
                </ul>
                <p>If you have any questions, please reply to this email.</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # Attach the report file
            with open(report_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='json')
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_path))
                msg.attach(attachment)
            
            # Attach the visualization file
            with open(visualization_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='pdf')
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(visualization_path))
                msg.attach(attachment)
            
            # Send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            print(f"Report sent to {email}")
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
    
    def list_scheduled_reports(self) -> List[Dict[str, Any]]:
        """
        List all scheduled reports.
        
        Returns:
            List of scheduled reports
        """
        return [
            {
                "report_id": report_id,
                "domain": report["domain"],
                "frequency": report["frequency"],
                "report_type": report["report_type"],
                "email": report["email"],
                "last_run": report["last_run"],
                "next_run": report["next_run"].isoformat() if isinstance(report["next_run"], datetime) else report["next_run"]
            }
            for report_id, report in self.scheduled_reports.items()
        ]
    
    def delete_scheduled_report(self, report_id: str) -> Dict[str, Any]:
        """
        Delete a scheduled report.
        
        Args:
            report_id: ID of the report to delete
            
        Returns:
            Dictionary with deletion status
        """
        if report_id in self.scheduled_reports:
            report = self.scheduled_reports.pop(report_id)
            return {
                "success": True,
                "report_id": report_id,
                "domain": report["domain"],
                "message": f"Report for {report['domain']} with {report['frequency']} frequency deleted"
            }
        else:
            return {
                "success": False,
                "report_id": report_id,
                "message": f"Report {report_id} not found"
            }
    
    def run(self, domain: str, frequency: str, report_type: str, email: Optional[str] = None) -> str:
        """
        Run the scheduled reporting tool and return formatted results.
        
        Args:
            domain: The domain to generate reports for
            frequency: Report frequency ('daily', 'weekly', 'monthly')
            report_type: Type of report ('basic', 'comprehensive')
            email: Optional email address to send reports to
            
        Returns:
            Formatted string with scheduling results
        """
        result = self.schedule_report(domain, frequency, report_type, email)
        
        if "error" in result:
            return f"Error scheduling report: {result['error']}"
        
        # Format the results as a readable string
        output = f"# Scheduled SEO Report\n\n"
        
        output += f"## Report Details\n"
        output += f"- Domain: {result['domain']}\n"
        output += f"- Frequency: {result['frequency']}\n"
        output += f"- Report Type: {result['report_type']}\n"
        
        if result.get('email'):
            output += f"- Email: {result['email']}\n"
        
        output += f"- Next Run: {result['next_run']}\n\n"
        
        output += f"## Report ID\n"
        output += f"`{result['report_id']}`\n\n"
        
        output += f"This report has been scheduled successfully. "
        
        if result.get('email'):
            output += f"The report will be sent to {result['email']} {result['frequency']}."
        else:
            output += f"The report will be generated {result['frequency']} and saved to the reports directory."
        
        return output


# Create tool instance
scheduled_reporting = ScheduledReportingTool()

# Create LangChain tool
scheduled_reporting_tool = Tool(
    name="scheduled_reporting",
    func=scheduled_reporting.run,
    description="Schedule regular SEO reports. Sets up automated analysis and reporting on a recurring basis.",
)
