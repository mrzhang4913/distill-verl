#!/bin/bash
# fix_path.sh
# 自动修复所有入口文件的 Python 路径问题

cd /home/mengrui/distill

echo "=========================================="
echo "Fixing Python path (no install needed)"
echo "=========================================="

# ── 1. 创建 tests/conftest.py（解决所有测试的路径问题）──
echo "Creating tests/conftest.py..."
cat > tests/conftest.py << 'EOF'
# tests/conftest.py
# pytest 自动加载此文件，将项目根目录加入 Python 路径
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
EOF
echo "✓ tests/conftest.py created"


# ── 2. 修复 train_distill.py ──
echo "Fixing train_distill.py..."
HEADER='import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n'
# 检查是否已经有路径修复
if ! grep -q "sys.path.insert" train_distill.py; then
    # 在第一行插入
    sed -i "1s/^/$HEADER/" train_distill.py
    echo "✓ train_distill.py fixed"
else
    echo "  train_distill.py already fixed, skipping"
fi


# ── 3. 修复 eval_math500.py ──
echo "Fixing eval_math500.py..."
if ! grep -q "sys.path.insert" eval_math500.py; then
    sed -i "1s/^/$HEADER/" eval_math500.py
    echo "✓ eval_math500.py fixed"
else
    echo "  eval_math500.py already fixed, skipping"
fi


# ── 4. 修复 analyze_results.py ──
echo "Fixing analyze_results.py..."
if ! grep -q "sys.path.insert" analyze_results.py; then
    sed -i "1s/^/$HEADER/" analyze_results.py
    echo "✓ analyze_results.py fixed"
else
    echo "  analyze_results.py already fixed, skipping"
fi


# ── 5. 修复 test_setup.py ──
echo "Fixing test_setup.py..."
if ! grep -q "sys.path.insert" test_setup.py; then
    sed -i "1s/^/$HEADER/" test_setup.py
    echo "✓ test_setup.py fixed"
else
    echo "  test_setup.py already fixed, skipping"
fi


# ── 6. 验证 ──
echo ""
echo "=========================================="
echo "Verifying fixes..."
echo "=========================================="
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/home/mengrui/distill')))

results = {}

try:
    from src.data import MATHDataset, MATHDataCollator
    results['src.data'] = '✓'
except Exception as e:
    results['src.data'] = f'✗ {e}'

try:
    from src.models import load_teacher_model
    results['src.models'] = '✓'
except Exception as e:
    results['src.models'] = f'✗ {e}'

try:
    from src.losses import get_distillation_loss
    results['src.losses'] = '✓'
except Exception as e:
    results['src.losses'] = f'✗ {e}'

try:
    from src.utils import load_config, set_seed
    results['src.utils'] = '✓'
except Exception as e:
    results['src.utils'] = f'✗ {e}'

for k, v in results.items():
    print(f'  {v}  {k}')

if all(v.startswith('✓') for v in results.values()):
    print('')
    print('All imports OK!')
else:
    print('')
    print('Some imports failed, check error messages above')
"

echo ""
echo "=========================================="
echo "Done! You can now run:"
echo "  python train_distill.py --config configs/forward_kl.yaml"
echo "  pytest tests/ -v"
echo "=========================================="
