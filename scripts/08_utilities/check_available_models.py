"""
Check available detection models in runs/detect
"""
import os
from pathlib import Path
import pandas as pd

print("="*70)
print("📁 Checking Detection Models")
print("="*70)

# Check runs/detect folder
runs_dir = Path('runs/detect')

if runs_dir.exists():
    print(f"\n📂 Found runs/detect folder")
    
    # List all subdirectories
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    
    print(f"   Total runs: {len(runs)}\n")
    
    best_models = []
    
    for run in runs:
        print(f"\n{'='*70}")
        print(f"📂 Run: {run.name}")
        print(f"{'='*70}")
        
        # Check for weights
        weights_dir = run / 'weights'
        if weights_dir.exists():
            weights = list(weights_dir.glob('*.pt'))
            print(f"\n✅ Weights folder found: {len(weights)} models")
            
            for w in weights:
                size_mb = w.stat().st_size / (1024*1024)
                print(f"   📦 {w.name}: {size_mb:.1f} MB")
                
                if 'best' in w.name:
                    best_models.append((run.name, str(w)))
        
        # Check for results
        results_csv = run / 'results.csv'
        if results_csv.exists():
            print(f"\n✅ Results file found")
            
            try:
                df = pd.read_csv(results_csv)
                
                # Show column names
                print(f"\n   Available metrics:")
                for col in df.columns:
                    col_clean = col.strip()
                    print(f"      - {col_clean}")
                
                # Show last row (final metrics)
                if len(df) > 0:
                    print(f"\n   📊 Final Metrics (Epoch {len(df)-1}):")
                    last_row = df.iloc[-1]
                    
                    # Show key metrics
                    for col in df.columns:
                        col_clean = col.strip()
                        if any(key in col_clean for key in ['mAP', 'precision', 'recall', 'loss']):
                            value = last_row[col]
                            print(f"      {col_clean}: {value:.4f}")
                
            except Exception as e:
                print(f"   ⚠️ Could not read results: {e}")
        
        # Check for args.yaml
        args_yaml = run / 'args.yaml'
        if args_yaml.exists():
            print(f"\n✅ Config file found: {args_yaml}")
    
    # Summary
    if best_models:
        print(f"\n\n{'='*70}")
        print(f"🏆 Best Models Found:")
        print(f"{'='*70}")
        for run_name, model_path in best_models:
            print(f"\n   Run: {run_name}")
            print(f"   Model: {model_path}")
    
else:
    print("\n❌ runs/detect folder not found")

# Also check models folder
print(f"\n\n{'='*70}")
print("📁 Checking models/ folder")
print(f"{'='*70}")

models_dir = Path('models')
if models_dir.exists():
    models = sorted([f for f in models_dir.iterdir() if f.suffix == '.pt'])
    
    print(f"\n   Total .pt files: {len(models)}\n")
    
    for model in models:
        size_mb = model.stat().st_size / (1024*1024)
        mod_time = model.stat().st_mtime
        
        from datetime import datetime
        mod_datetime = datetime.fromtimestamp(mod_time)
        
        print(f"\n📦 {model.name}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Modified: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*70)
print("✅ Check complete!")
print("="*70)
