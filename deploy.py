#!/usr/bin/env python3
"""
Deployment script for the Customer Churn Prediction project.

This script will:
1. Clean previous outputs
2. Run the complete training pipeline
3. Launch the Streamlit web application
4. Display results and visualizations
"""

import os
import sys
import subprocess
import shutil
import logging
import argparse
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_DATASET = "Dataset_Cay quyet dinh_HV.xlsx"


def clean_outputs():
    """Clean previous output directories."""
    logger.info("🧹 Cleaning previous outputs...")
    
    # Remove outputs directory if it exists
    if OUTPUTS_DIR.exists():
        shutil.rmtree(OUTPUTS_DIR)
        logger.info(f"Removed existing outputs directory: {OUTPUTS_DIR}")
    
    # Create fresh outputs directory structure
    (OUTPUTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "results").mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Created fresh outputs directory structure")


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("🔍 Checking dependencies...")
    
    # Map package names to their import names
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit',
        'imbalanced-learn': 'imblearn',
        'joblib': 'joblib',
        'openpyxl': 'openpyxl',
        'plotly': 'plotly',
        'pillow': 'PIL'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True


def find_dataset():
    """Find the dataset file."""
    logger.info("📁 Looking for dataset...")
    
    dataset_path = DATASETS_DIR / DEFAULT_DATASET
    
    if dataset_path.exists():
        logger.info(f"✅ Found dataset: {dataset_path}")
        return str(dataset_path)
    
    # Look for any Excel files in datasets directory
    excel_files = list(DATASETS_DIR.glob("*.xlsx")) + list(DATASETS_DIR.glob("*.xls"))
    
    if excel_files:
        dataset_path = excel_files[0]
        logger.info(f"✅ Found alternative dataset: {dataset_path}")
        return str(dataset_path)
    
    logger.error(f"❌ No dataset found in {DATASETS_DIR}")
    logger.info("Please place your dataset file in the datasets/ directory")
    return None


def run_training_pipeline(dataset_path, quick_mode=True):
    """Run the complete training pipeline."""
    logger.info("🚀 Starting training pipeline...")
    
    cmd = [
        sys.executable, "scripts/train_model.py",
        "--data", dataset_path,
        "--output-dir", str(OUTPUTS_DIR),
        "--visualize-tree",
        "--max-depth", "4"
    ]
    
    logger.info("Running baseline model training pipeline...")
    
    try:
        # Run the training script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("✅ Training pipeline completed successfully!")
        
        # Log some output for debugging
        if result.stdout:
            logger.info("Pipeline output:")
            for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    logger.info(f"  {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training pipeline failed with error: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during training: {e}")
        return False


def verify_outputs():
    """Verify that all expected outputs were generated."""
    logger.info("🔍 Verifying outputs...")
    
    # Core baseline files that should always be present
    expected_files = [
        OUTPUTS_DIR / "models" / "baseline_model.pkl",
        OUTPUTS_DIR / "plots" / "baseline_roc_curve.png",
        OUTPUTS_DIR / "plots" / "baseline_confusion_matrix.png"
    ]
    
    missing_files = []
    existing_files = []
    
    # Check required files
    for file_path in expected_files:
        if file_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    logger.info(f"✅ Generated {len(existing_files)} output files")
    
    if missing_files:
        logger.warning(f"⚠️  Missing {len(missing_files)} core files:")
        for file_path in missing_files:
            logger.warning(f"  - {file_path}")
    
    # List all generated files
    all_outputs = []
    for subfolder in ["models", "plots", "results"]:
        subfolder_path = OUTPUTS_DIR / subfolder
        if subfolder_path.exists():
            files = list(subfolder_path.glob("*.*"))
            all_outputs.extend(files)
    
    logger.info(f"📊 Total generated files: {len(all_outputs)}")
    
    # Success if we have at least the baseline model
    baseline_model_exists = (OUTPUTS_DIR / "models" / "baseline_model.pkl").exists()
    return baseline_model_exists


def launch_streamlit():
    """Launch the Streamlit application."""
    logger.info("🌐 Launching Streamlit application...")
    
    try:
        # Launch Streamlit in a new process
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        
        logger.info("🚀 Starting Streamlit server...")
        logger.info("📱 Open your browser and navigate to: http://localhost:8501")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        # Run Streamlit (this will block until interrupted)
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("🛑 Streamlit server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch Streamlit: {e}")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Customer Churn Prediction Project')
    parser.add_argument('--no-streamlit', action='store_true',
                       help='Skip launching Streamlit (training only)')
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset file (optional)')
    
    args = parser.parse_args()
    
    print("🎯 Customer Churn Prediction - Deployment Script")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return 1
    
    # Step 2: Clean previous outputs
    clean_outputs()
    
    # Step 3: Find dataset
    if args.dataset:
        dataset_path = args.dataset
        if not Path(dataset_path).exists():
            logger.error(f"❌ Dataset file not found: {dataset_path}")
            return 1
    else:
        dataset_path = find_dataset()
        if not dataset_path:
            return 1
    
    # Step 4: Run training pipeline
    if not run_training_pipeline(dataset_path):
        return 1
    
    # Step 5: Verify outputs
    if not verify_outputs():
        logger.error("❌ Training completed but outputs are missing")
        return 1
    
    # Step 6: Launch Streamlit (unless disabled)
    if not args.no_streamlit:
        print("\n" + "=" * 50)
        print("🎉 Training completed successfully!")
        print("🌐 Launching Streamlit web application...")
        print("=" * 50)
        
        time.sleep(2)  # Brief pause for user to read
        launch_streamlit()
    else:
        print("\n" + "=" * 50)
        print("🎉 Training completed successfully!")
        print(f"📁 Outputs saved in: {OUTPUTS_DIR}")
        print("🌐 To launch Streamlit manually: streamlit run app.py")
        print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 