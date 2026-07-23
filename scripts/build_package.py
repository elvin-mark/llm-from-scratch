import os
import subprocess
import sys

def build_package():
    print("📦 Building Python package (wheel & source distribution)...")
    
    # Ensure build & dist clean state or build via uv
    cmd = ["uv", "build"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Error building package:")
        print(e.stderr)
        sys.exit(1)
        
    dist_dir = "dist"
    if os.path.exists(dist_dir):
        files = os.listdir(dist_dir)
        print("\n✅ Package built successfully inside dist/:")
        for f in sorted(files):
            file_path = os.path.join(dist_dir, f)
            size_kb = os.path.getsize(file_path) / 1024.0
            print(f"  - {f} ({size_kb:.2f} KB)")

if __name__ == "__main__":
    build_package()
