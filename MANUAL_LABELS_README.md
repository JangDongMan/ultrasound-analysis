# Manual Labels Feature

## Overview
The visualization script now supports overlaying manual thickness measurements on the ultrasound analysis plots. Manual measurements are displayed as **blue dots** on the filtered signal plot.

## Label File Format

### CSV Format (label.csv)
```csv
filename,thickness1,thickness2
bhjung-5M-1.csv,0.0015,0.0035
bhjung-5M-2.csv,0.0016,0.0038
bhjung-5M-3.csv,0.0014,0.0032
```

### Excel Format (label.xlsx)
Same structure as CSV, with columns:
- `filename`: Name of the ultrasound data file (e.g., "bhjung-5M-1.csv")
- `thickness1`: Distance to Position 1 (진피층 시작) in **meters**
- `thickness2`: Distance to Position 2 (근막층 시작) in **meters**

## Important Notes

1. **Units**: The thickness values in the label file must be in **meters** (m), not millimeters
   - Example: 1.5mm should be entered as `0.0015`
   - Example: 3.5mm should be entered as `0.0035`

2. **File Location**: The label file can be placed in:
   - Root directory: `label.csv` or `label.xlsx`
   - Data directory: `data/label.csv` or `data/label.xlsx`
   - Alternative names: `labels.csv` or `labels.xlsx`

3. **Visualization**:
   - Automatic detection positions are shown as **red dots**
   - Manual measurements are shown as **blue dots**
   - Both include depth annotations in millimeters

## Usage

### Automatic (Main Script)
```bash
python3 visualize_signal.py
```
The script automatically searches for label files and applies them if found.

### Programmatic
```python
from visualize_signal import visualize_ultrasound_signal

# With manual labels
manual_labels = {
    'thickness1': 0.0015,  # 1.5mm in meters
    'thickness2': 0.0035   # 3.5mm in meters
}

visualize_ultrasound_signal(
    'data/bhjung-5M-1.csv',
    'results/output.png',
    manual_labels=manual_labels
)

# Without manual labels
visualize_ultrasound_signal(
    'data/bhjung-5M-1.csv',
    'results/output.png'
)
```

## Example Output
The visualization will show:
- **Panel 3 (Filtered Signal)**:
  - Red dots with yellow boxes: Automatic position detection
  - Blue dots with light blue boxes: Manual measurements
  - Both annotations show the depth in mm

## Creating Your Label File

1. Measure the two positions manually from your ultrasound data
2. Convert measurements from mm to meters (divide by 1000)
3. Create a CSV or Excel file with the format shown above
4. Place the file in the workspace directory
5. Run `visualize_signal.py`

## Position Definitions
- **Position 1 (thickness1)**: 진피층 시작 (Start of dermis layer)
- **Position 2 (thickness2)**: 근막층 시작 (Start of fascia layer)

Both positions represent the **depth from the skin surface** (i.e., from the pulse start point).
