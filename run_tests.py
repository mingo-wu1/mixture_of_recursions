#!/usr/bin/env python3
import subprocess
import sys

def run_tests():
    """Run all tests."""
    print("Running MoR Model Tests...")
    print("="*50)
    
    # Run model tests
    try:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'test_model', '-v'], 
                              capture_output=True, text=True, cwd='.')
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error running model tests: {e}")
    
    print("\nRunning Scheduler Tests...")
    print("="*50)
    
    # Run scheduler tests
    try:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'test_scheduler', '-v'], 
                              capture_output=True, text=True, cwd='.')
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error running scheduler tests: {e}")

if __name__ == "__main__":
    run_tests()