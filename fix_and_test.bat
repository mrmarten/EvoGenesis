@echo off
echo ===== EvoGenesis Fix Script =====
echo.
echo Step 1: Creating strategic_observatory.json in config directory...
python -c "import os, json, sys; src = os.path.join('data', 'strategic_observatory', 'strategic_observatory.json'); dst = os.path.join('config', 'strategic_observatory.json'); os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'w').write(open(src, 'r').read()); print('Successfully created config file!')"
echo.

echo Step 2: Running the perception_action_demo example...
python examples\perception_action_demo.py
echo.

echo ===== Fix complete =====
pause
