#!/usr/bin/env python3
"""
Test script to verify Docker configuration for the Medical Dataset Processor web interface.
"""
import subprocess
import time
import requests
import sys
import os


def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            timeout=60
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return None


def test_docker_build():
    """Test Docker image build."""
    print("Testing Docker image build...")
    
    result = run_command("docker build -t medical-dataset-processor-test .")
    
    if result and result.returncode == 0:
        print("✅ Docker image built successfully")
        return True
    else:
        print("❌ Docker image build failed")
        if result:
            print(f"Error: {result.stderr}")
        return False


def test_docker_compose_config():
    """Test docker-compose configuration."""
    print("Testing docker-compose configuration...")
    
    result = run_command("docker-compose config")
    
    if result and result.returncode == 0:
        print("✅ docker-compose.yml configuration is valid")
        return True
    else:
        print("❌ docker-compose.yml configuration is invalid")
        if result:
            print(f"Error: {result.stderr}")
        return False


def test_environment_variables():
    """Test that required environment variables are documented."""
    print("Testing environment variables documentation...")
    
    required_vars = [
        'DEEPL_API_KEY',
        'SECRET_KEY',
        'TARGET_LANGUAGE',
        'WEB_HOST',
        'WEB_PORT'
    ]
    
    try:
        with open('.env.example', 'r') as f:
            env_content = f.read()
        
        missing_vars = []
        for var in required_vars:
            if var not in env_content:
                missing_vars.append(var)
        
        if not missing_vars:
            print("✅ All required environment variables are documented in .env.example")
            return True
        else:
            print(f"❌ Missing environment variables in .env.example: {missing_vars}")
            return False
            
    except FileNotFoundError:
        print("❌ .env.example file not found")
        return False


def test_docker_run_dry():
    """Test Docker container startup (dry run without actual API keys)."""
    print("Testing Docker container startup (dry run)...")
    
    # Create a temporary .env file for testing
    test_env_content = """
DEEPL_API_KEY=test_key_for_docker_test
OPENAI_API_KEY=test_key_for_docker_test
SECRET_KEY=test-secret-key-for-docker
TARGET_LANGUAGE=FR
WEB_HOST=0.0.0.0
WEB_PORT=5000
"""
    
    with open('.env.test', 'w') as f:
        f.write(test_env_content)
    
    try:
        # Try to start the container with test environment using test Flask app
        print("Starting container with test environment...")
        result = run_command(
            "docker run --rm -d --name medical-dataset-test --env-file .env.test -p 5001:5000 medical-dataset-processor-test python test_flask_startup.py",
            capture_output=True
        )
        
        if result and result.returncode == 0:
            container_id = result.stdout.strip()
            print(f"✅ Container started successfully: {container_id}")
            
            # Wait a moment for the container to start
            time.sleep(10)
            
            # Check if container is still running
            check_result = run_command(f"docker ps -q -f id={container_id}")
            if check_result and check_result.stdout.strip():
                print("✅ Container is running")
                
                # Try to access health endpoint
                try:
                    response = requests.get("http://localhost:5001/api/health", timeout=5)
                    if response.status_code == 200:
                        print("✅ Health endpoint accessible")
                        health_success = True
                    else:
                        print(f"❌ Health endpoint returned status {response.status_code}")
                        health_success = False
                except requests.exceptions.RequestException as e:
                    print(f"❌ Could not access health endpoint: {e}")
                    health_success = False
                
                # Stop the container
                run_command(f"docker stop {container_id}")
                print("Container stopped")
                
                return health_success
            else:
                print("❌ Container stopped unexpectedly")
                # Get container logs
                logs_result = run_command(f"docker logs {container_id}")
                if logs_result:
                    print("Container logs:")
                    print(logs_result.stdout)
                    print(logs_result.stderr)
                return False
        else:
            print("❌ Failed to start container")
            if result:
                print(f"Error: {result.stderr}")
            return False
            
    finally:
        # Clean up test environment file
        if os.path.exists('.env.test'):
            os.remove('.env.test')
        
        # Make sure container is stopped
        run_command("docker stop medical-dataset-test 2>/dev/null || true")


def main():
    """Run all Docker tests."""
    print("🐳 Testing Docker configuration for Medical Dataset Processor Web Interface")
    print("=" * 70)
    
    tests = [
        ("Docker Build", test_docker_build),
        ("Docker Compose Config", test_docker_compose_config),
        ("Environment Variables", test_environment_variables),
        ("Docker Container Startup", test_docker_run_dry),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 50)
        success = test_func()
        results.append((test_name, success))
        print()
    
    print("=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 All Docker tests passed! The configuration is ready for deployment.")
        print("\nTo run the application:")
        print("1. Copy .env.example to .env and fill in your API keys")
        print("2. Run: docker-compose up -d")
        print("3. Access the web interface at http://localhost:5000")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        sys.exit(1)


if __name__ == '__main__':
    main()