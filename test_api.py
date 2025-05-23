"""
Test script for the API.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create a simple FastAPI application
app = FastAPI(
    title="DataMCPServerAgent API Test",
    description="Test API for DataMCPServerAgent",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DataMCPServerAgent API Test",
        "status": "ok",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "test_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
