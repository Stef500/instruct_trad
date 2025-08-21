#!/usr/bin/env python3
"""
Test script to verify Docker configuration for the Medical Dataset Processor web interface.
"""
import subprocess
import time
import requests
import sys
import os
import shutil
def require_docker_or_fail():
    """Ensure docker CLI and daemon are available, else fail with clear message."""
    if shutil.which("docker") is None:
        raise AssertionError(
            "Docker CLI introuvable dans le PATH. Installe Docker Desktop et réessaie."
        )
    # Vérifie l'accès au démon
    info = run_command("docker info")
    if info is None:
        raise AssertionError("La commande 'docker info' a expiré (timeout)")
    if info.returncode != 0:
        raise AssertionError(
            "Docker daemon inaccessible. Assure-toi que Docker Desktop est démarré.\n"
            f"Sortie: {info.stdout}\nErreurs: {info.stderr}"
        )

def _docker_compose_config_command() -> str:
    """Return the available docker compose config command (plugin or legacy)."""
    if shutil.which("docker-compose") is not None:
        return "docker-compose config"
    # Fallback to docker compose (plugin)
    return "docker compose config"


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
    require_docker_or_fail()
    
    result = run_command("docker build -t medical-dataset-processor-test .")
    
    assert result is not None, "docker build command timed out"
    assert result.returncode == 0, (
        "Docker image build failed.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    print("✅ Docker image built successfully")


def test_docker_compose_config():
    """Test docker-compose configuration."""
    print("Testing docker-compose configuration...")
    require_docker_or_fail()
    
    cmd = _docker_compose_config_command()
    result = run_command(cmd)
    
    assert result is not None, f"{cmd} command timed out"
    assert result.returncode == 0, (
        "docker-compose.yml configuration is invalid.\n"
        f"Command: {cmd}\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    print("✅ docker-compose.yml configuration is valid")


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
        
        assert not missing_vars, f"Missing environment variables in .env.example: {missing_vars}"
        print("✅ All required environment variables are documented in .env.example")
            
    except FileNotFoundError:
        raise AssertionError(".env.example file not found")


def test_docker_run_dry():
    """Test Docker container startup (dry run without actual API keys)."""
    print("Testing Docker container startup (dry run)...")
    require_docker_or_fail()
    
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
            "docker run --rm -d --name medical-dataset-test --env-file .env.test -p 5001:5000 medical-dataset-processor-test python /app/test_flask_startup.py",
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
                
                assert health_success, "Health endpoint not accessible"
            else:
                print("❌ Container stopped unexpectedly")
                # Get container logs
                logs_result = run_command(f"docker logs {container_id}")
                if logs_result:
                    print("Container logs:")
                    print(logs_result.stdout)
                    print(logs_result.stderr)
                raise AssertionError("Container stopped unexpectedly")
        else:
            print("❌ Failed to start container")
            if result:
                print(f"Error: {result.stderr}")
            raise AssertionError("Failed to start container")
            
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