"""
Debug authentication system.
"""

import requests
from auth_system import auth_system

def debug_auth():
    print("🔍 Debugging Authentication System")
    print("=" * 50)
    
    # Show current users and their API keys
    print("Current users in auth system:")
    for user_id, user in auth_system.users.items():
        print(f"  {user.username}: {user.api_key}")
    
    print()
    
    # Test authentication directly
    print("Testing authentication directly:")
    user_key = auth_system.users['user_001'].api_key
    print(f"Testing key: {user_key}")
    
    user = auth_system.authenticate_api_key(user_key)
    if user:
        print(f"✅ Direct auth successful: {user.username}")
    else:
        print("❌ Direct auth failed")
    
    print()
    
    # Test API endpoint
    print("Testing API endpoint:")
    headers = {"Authorization": f"Bearer {user_key}"}
    
    try:
        response = requests.get("http://localhost:8002/v1/auth/me", headers=headers)
        print(f"API response status: {response.status_code}")
        print(f"API response: {response.text}")
    except Exception as e:
        print(f"API request failed: {e}")

if __name__ == "__main__":
    debug_auth()
