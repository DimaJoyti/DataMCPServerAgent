"""
Get API keys from the auth system for testing.
"""

from auth_system import auth_system

def main():
    print("🔑 API Keys for Testing:")
    print("=" * 50)
    
    for user_id, user in auth_system.users.items():
        print(f"👤 {user.username} ({user.role.value}):")
        print(f"   API Key: {user.api_key}")
        print(f"   User ID: {user.user_id}")
        print(f"   Email: {user.email}")
        print(f"   Permissions: {[p.value for p in user.permissions]}")
        print()
    
    print("🧪 Test Commands:")
    print("=" * 50)
    
    admin_key = auth_system.users['admin_001'].api_key
    dev_key = auth_system.users['dev_001'].api_key
    user_key = auth_system.users['user_001'].api_key
    
    print(f"# Test with Admin key:")
    print(f'curl -H "Authorization: Bearer {admin_key}" http://localhost:8002/v1/tools')
    print()
    
    print(f"# Test with Developer key:")
    print(f'curl -H "Authorization: Bearer {dev_key}" http://localhost:8002/v1/tools')
    print()
    
    print(f"# Test with User key:")
    print(f'curl -H "Authorization: Bearer {user_key}" http://localhost:8002/v1/tools')
    print()
    
    print(f"# Create agent with User key:")
    print(f'curl -X POST -H "Authorization: Bearer {user_key}" -H "Content-Type: application/json" -d \'{{"agent_type": "cloudflare_worker", "configuration": {{"name": "Test Agent"}}}}\' http://localhost:8002/v1/agents')
    print()
    
    print(f"# Execute tool with User key:")
    print(f'curl -X POST -H "Authorization: Bearer {user_key}" -H "Content-Type: application/json" -d \'{{"parameters": {{}}, "session_id": "test_session"}}\' http://localhost:8002/v1/tools/workers_list')

if __name__ == "__main__":
    main()
