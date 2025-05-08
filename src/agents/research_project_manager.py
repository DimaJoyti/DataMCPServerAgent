"""
Research Project Manager for the Research Assistant.

This module provides functionality for managing research projects, including
creating, updating, and retrieving projects, queries, and results.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from src.models.research_models import (
    Annotation,
    Comment,
    Permission,
    ResearchProject,
    ResearchQuery,
    ResearchResult,
    ResearchResultWithComments,
    SharedResearch,
    Source,
    User,
)


class ResearchProjectManager:
    """Manager for research projects."""
    
    def __init__(self, data_dir: str = "research_data"):
        """
        Initialize the research project manager.
        
        Args:
            data_dir: Directory for storing research data
        """
        self.data_dir = data_dir
        self.projects: Dict[str, ResearchProject] = {}
        self.users: Dict[str, User] = {}
        self.shared_research: List[SharedResearch] = []
        
        # Create the data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing projects
        self._load_projects()
    
    def _load_projects(self) -> None:
        """Load existing projects from the data directory."""
        try:
            # Check if the projects file exists
            projects_file = os.path.join(self.data_dir, "projects.json")
            if os.path.exists(projects_file):
                with open(projects_file, "r", encoding="utf-8") as f:
                    projects_data = json.load(f)
                
                # Convert the data to ResearchProject objects
                for project_data in projects_data:
                    project = ResearchProject(**project_data)
                    self.projects[project.id] = project
            
            # Check if the users file exists
            users_file = os.path.join(self.data_dir, "users.json")
            if os.path.exists(users_file):
                with open(users_file, "r", encoding="utf-8") as f:
                    users_data = json.load(f)
                
                # Convert the data to User objects
                for user_data in users_data:
                    user = User(**user_data)
                    self.users[user.id] = user
            
            # Check if the shared research file exists
            shared_file = os.path.join(self.data_dir, "shared.json")
            if os.path.exists(shared_file):
                with open(shared_file, "r", encoding="utf-8") as f:
                    shared_data = json.load(f)
                
                # Convert the data to SharedResearch objects
                for shared_item_data in shared_data:
                    shared_item = SharedResearch(**shared_item_data)
                    self.shared_research.append(shared_item)
        except Exception as e:
            print(f"Error loading projects: {str(e)}")
    
    def _save_projects(self) -> None:
        """Save projects to the data directory."""
        try:
            # Save projects
            projects_data = [project.dict() for project in self.projects.values()]
            projects_file = os.path.join(self.data_dir, "projects.json")
            with open(projects_file, "w", encoding="utf-8") as f:
                json.dump(projects_data, f, default=str, indent=2)
            
            # Save users
            users_data = [user.dict() for user in self.users.values()]
            users_file = os.path.join(self.data_dir, "users.json")
            with open(users_file, "w", encoding="utf-8") as f:
                json.dump(users_data, f, default=str, indent=2)
            
            # Save shared research
            shared_data = [shared_item.dict() for shared_item in self.shared_research]
            shared_file = os.path.join(self.data_dir, "shared.json")
            with open(shared_file, "w", encoding="utf-8") as f:
                json.dump(shared_data, f, default=str, indent=2)
        except Exception as e:
            print(f"Error saving projects: {str(e)}")
    
    def create_project(self, name: str, description: str = "", tags: List[str] = None) -> ResearchProject:
        """
        Create a new research project.
        
        Args:
            name: Project name
            description: Project description
            tags: Project tags
            
        Returns:
            ResearchProject object
        """
        # Create a new project
        project = ResearchProject(
            name=name,
            description=description,
            tags=tags or []
        )
        
        # Add the project to the dictionary
        self.projects[project.id] = project
        
        # Save the projects
        self._save_projects()
        
        return project
    
    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """
        Get a research project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            ResearchProject object or None if not found
        """
        return self.projects.get(project_id)
    
    def get_all_projects(self) -> List[ResearchProject]:
        """
        Get all research projects.
        
        Returns:
            List of ResearchProject objects
        """
        return list(self.projects.values())
    
    def update_project(self, project_id: str, name: str = None, description: str = None, tags: List[str] = None) -> Optional[ResearchProject]:
        """
        Update a research project.
        
        Args:
            project_id: Project ID
            name: New project name
            description: New project description
            tags: New project tags
            
        Returns:
            Updated ResearchProject object or None if not found
        """
        # Get the project
        project = self.get_project(project_id)
        if not project:
            return None
        
        # Update the project
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if tags is not None:
            project.tags = tags
        
        # Update the timestamp
        project.updated_at = datetime.now()
        
        # Save the projects
        self._save_projects()
        
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a research project.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if the project was deleted, False otherwise
        """
        # Check if the project exists
        if project_id not in self.projects:
            return False
        
        # Delete the project
        del self.projects[project_id]
        
        # Delete any shared research for this project
        self.shared_research = [item for item in self.shared_research if item.project_id != project_id]
        
        # Save the projects
        self._save_projects()
        
        return True
    
    def add_query(self, project_id: str, query: str, tags: List[str] = None) -> Optional[ResearchQuery]:
        """
        Add a query to a research project.
        
        Args:
            project_id: Project ID
            query: Query string
            tags: Query tags
            
        Returns:
            ResearchQuery object or None if the project was not found
        """
        # Get the project
        project = self.get_project(project_id)
        if not project:
            return None
        
        # Create a new query
        research_query = ResearchQuery(
            query=query,
            tags=tags or []
        )
        
        # Add the query to the project
        project.queries.append(research_query)
        
        # Update the timestamp
        project.updated_at = datetime.now()
        
        # Save the projects
        self._save_projects()
        
        return research_query
    
    def get_query(self, project_id: str, query_id: str) -> Optional[ResearchQuery]:
        """
        Get a query from a research project.
        
        Args:
            project_id: Project ID
            query_id: Query ID
            
        Returns:
            ResearchQuery object or None if not found
        """
        # Get the project
        project = self.get_project(project_id)
        if not project:
            return None
        
        # Find the query
        for query in project.queries:
            if query.id == query_id:
                return query
        
        return None
    
    def add_result(self, project_id: str, query_id: str, result: ResearchResult) -> bool:
        """
        Add a result to a query in a research project.
        
        Args:
            project_id: Project ID
            query_id: Query ID
            result: ResearchResult object
            
        Returns:
            True if the result was added, False otherwise
        """
        # Get the project
        project = self.get_project(project_id)
        if not project:
            return False
        
        # Find the query
        query = None
        for q in project.queries:
            if q.id == query_id:
                query = q
                break
        
        if not query:
            return False
        
        # Add the result to the query
        query.results.append(result)
        
        # Update the timestamp
        project.updated_at = datetime.now()
        
        # Save the projects
        self._save_projects()
        
        return True
    
    def get_result(self, project_id: str, query_id: str, result_id: str) -> Optional[ResearchResult]:
        """
        Get a result from a query in a research project.
        
        Args:
            project_id: Project ID
            query_id: Query ID
            result_id: Result ID
            
        Returns:
            ResearchResult object or None if not found
        """
        # Get the query
        query = self.get_query(project_id, query_id)
        if not query:
            return None
        
        # Find the result
        for result in query.results:
            if result.id == result_id:
                return result
        
        return None
    
    def add_user(self, name: str, email: str = None) -> User:
        """
        Add a user to the system.
        
        Args:
            name: User name
            email: User email
            
        Returns:
            User object
        """
        # Create a new user
        user_id = f"user_{len(self.users) + 1}"
        user = User(
            id=user_id,
            name=name,
            email=email
        )
        
        # Add the user to the dictionary
        self.users[user.id] = user
        
        # Save the projects
        self._save_projects()
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None if not found
        """
        return self.users.get(user_id)
    
    def share_project(self, project_id: str, user_id: str, permission: Permission = Permission.READ) -> Optional[SharedResearch]:
        """
        Share a project with a user.
        
        Args:
            project_id: Project ID
            user_id: User ID
            permission: Permission level
            
        Returns:
            SharedResearch object or None if the project or user was not found
        """
        # Check if the project and user exist
        if project_id not in self.projects or user_id not in self.users:
            return None
        
        # Create a new shared research item
        shared_item = SharedResearch(
            project_id=project_id,
            user_id=user_id,
            permission=permission
        )
        
        # Add the shared item to the list
        self.shared_research.append(shared_item)
        
        # Save the projects
        self._save_projects()
        
        return shared_item
    
    def get_shared_projects(self, user_id: str) -> List[Dict]:
        """
        Get projects shared with a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of dictionaries containing project and permission information
        """
        # Find shared projects for the user
        shared_projects = []
        for shared_item in self.shared_research:
            if shared_item.user_id == user_id:
                project = self.get_project(shared_item.project_id)
                if project:
                    shared_projects.append({
                        "project": project,
                        "permission": shared_item.permission
                    })
        
        return shared_projects
    
    def add_comment(self, project_id: str, query_id: str, result_id: str, user_id: str, content: str) -> Optional[Comment]:
        """
        Add a comment to a research result.
        
        Args:
            project_id: Project ID
            query_id: Query ID
            result_id: Result ID
            user_id: User ID
            content: Comment content
            
        Returns:
            Comment object or None if the result was not found
        """
        # Get the result
        result = self.get_result(project_id, query_id, result_id)
        if not result:
            return None
        
        # Check if the user exists
        if user_id not in self.users:
            return None
        
        # Convert the result to a ResearchResultWithComments if needed
        if not isinstance(result, ResearchResultWithComments):
            result_with_comments = ResearchResultWithComments(**result.dict())
            
            # Replace the result in the query
            query = self.get_query(project_id, query_id)
            for i, r in enumerate(query.results):
                if r.id == result_id:
                    query.results[i] = result_with_comments
                    break
            
            result = result_with_comments
        
        # Create a new comment
        comment = Comment(
            user_id=user_id,
            content=content
        )
        
        # Add the comment to the result
        result.comments.append(comment)
        
        # Update the project timestamp
        project = self.get_project(project_id)
        project.updated_at = datetime.now()
        
        # Save the projects
        self._save_projects()
        
        return comment
    
    def add_annotation(self, project_id: str, query_id: str, result_id: str, user_id: str, content: str, target_text: str) -> Optional[Annotation]:
        """
        Add an annotation to a research result.
        
        Args:
            project_id: Project ID
            query_id: Query ID
            result_id: Result ID
            user_id: User ID
            content: Annotation content
            target_text: Text being annotated
            
        Returns:
            Annotation object or None if the result was not found
        """
        # Get the result
        result = self.get_result(project_id, query_id, result_id)
        if not result:
            return None
        
        # Check if the user exists
        if user_id not in self.users:
            return None
        
        # Convert the result to a ResearchResultWithComments if needed
        if not isinstance(result, ResearchResultWithComments):
            result_with_comments = ResearchResultWithComments(**result.dict())
            
            # Replace the result in the query
            query = self.get_query(project_id, query_id)
            for i, r in enumerate(query.results):
                if r.id == result_id:
                    query.results[i] = result_with_comments
                    break
            
            result = result_with_comments
        
        # Create a new annotation
        annotation = Annotation(
            user_id=user_id,
            content=content,
            target_text=target_text
        )
        
        # Add the annotation to the result
        result.annotations.append(annotation)
        
        # Update the project timestamp
        project = self.get_project(project_id)
        project.updated_at = datetime.now()
        
        # Save the projects
        self._save_projects()
        
        return annotation
    
    def search_projects(self, query: str) -> List[ResearchProject]:
        """
        Search for projects matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching ResearchProject objects
        """
        query = query.lower()
        matching_projects = []
        
        for project in self.projects.values():
            # Check if the query matches the project name or description
            if query in project.name.lower() or query in project.description.lower():
                matching_projects.append(project)
                continue
            
            # Check if the query matches any tags
            if any(query in tag.lower() for tag in project.tags):
                matching_projects.append(project)
                continue
            
            # Check if the query matches any queries
            if any(query in q.query.lower() for q in project.queries):
                matching_projects.append(project)
                continue
        
        return matching_projects
    
    def search_results(self, query: str) -> List[Dict]:
        """
        Search for results matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of dictionaries containing project, query, and result information
        """
        query = query.lower()
        matching_results = []
        
        for project in self.projects.values():
            for research_query in project.queries:
                for result in research_query.results:
                    # Check if the query matches the result topic or summary
                    if query in result.topic.lower() or query in result.summary.lower():
                        matching_results.append({
                            "project": project,
                            "query": research_query,
                            "result": result
                        })
                        continue
                    
                    # Check if the query matches any tags
                    if any(query in tag.lower() for tag in result.tags):
                        matching_results.append({
                            "project": project,
                            "query": research_query,
                            "result": result
                        })
                        continue
                    
                    # Check if the query matches any sources
                    if any(isinstance(source, str) and query in source.lower() for source in result.sources):
                        matching_results.append({
                            "project": project,
                            "query": research_query,
                            "result": result
                        })
                        continue
                    
                    # Check if the query matches any Source objects
                    if any(isinstance(source, Source) and (
                        query in source.title.lower() or
                        (source.authors and any(query in author.lower() for author in source.authors))
                    ) for source in result.sources):
                        matching_results.append({
                            "project": project,
                            "query": research_query,
                            "result": result
                        })
                        continue
        
        return matching_results


# Create a singleton instance
research_project_manager = ResearchProjectManager()