import os
import sys
from dotenv import load_dotenv
import requests

def check_environment():
    """Check if all required environment variables are set"""
    print("Checking environment configuration...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Required variables
    required_vars = {
        'DEVTO_API_KEY': 'Dev.to API key for publishing articles',
        'SECRET_KEY': 'Flask application secret key',
    }
    
    # Optional variables with defaults
    optional_vars = {
        'DEBUG': ('True', 'Debug mode flag (True/False)'),
        'CACHE_TYPE': ('simple', 'Flask cache type'),
        'CACHE_DEFAULT_TIMEOUT': ('300', 'Cache timeout in seconds'),
        'CACHE_THRESHOLD': ('500', 'Maximum cache entries'),
        'DEFAULT_MODEL': ('flux', 'Default image generation model'),
        'IMAGE_WIDTH': ('800', 'Image width for generated images'),
        'IMAGE_HEIGHT': ('400', 'Image height for generated images')
    }
    
    # Check required variables
    missing_vars = []
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if not value or value == 'your_devto_api_key':
            missing_vars.append(f"{var}: {description}")
        else:
            # Mask sensitive values
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
            print(f"✓ {var}: {masked_value}")
    
    # Check optional variables
    for var, (default, description) in optional_vars.items():
        value = os.environ.get(var, default)
        os.environ[var] = value  # Set default if not present
        print(f"✓ {var}: {value} {'(default)' if os.environ.get(var) is None else ''}")
    
    # Report missing variables
    if missing_vars:
        print("\n❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    # Check if Dev.to API key is valid
    if 'DEVTO_API_KEY' in os.environ and os.environ['DEVTO_API_KEY'] != 'your_devto_api_key':
        try:
            print("\nVerifying Dev.to API key...")
            response = requests.get(
                'https://dev.to/api/articles/me', 
                headers={'api-key': os.environ['DEVTO_API_KEY']}
            )
            if response.status_code == 200:
                print("✓ Dev.to API key is valid")
            else:
                print(f"❌ Dev.to API key validation failed with status code: {response.status_code}")
                if response.status_code == 401:
                    print("  The API key is invalid or expired. Please check your key.")
                return False
        except Exception as e:
            print(f"❌ Error verifying Dev.to API key: {str(e)}")
            return False
    
    print("\n✓ Environment configuration looks good!")
    return True

if __name__ == "__main__":
    if check_environment():
        print("\nAll systems go! 🚀")
        sys.exit(0)
    else:
        print("\nEnvironment check failed. Please fix the issues above. ❌")
        sys.exit(1) 