# @Author: mac
# @Date:   2024-05-09 23:13:16
# @Last Modified by:   mac
# @Last Modified time: 2024-06-12 10:35:50


# Check if Python is installed
if command -v python3 &>/dev/null; then
    python_cmd="python3"
elif command -v python &>/dev/null; then
    python_cmd="python"
else
    echo "Error: Python is not installed. Please install Python to run this script."
    exit 1
fi

# cd 

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the directory of the script
cd "$DIR"

# Get the directory path of the YAML file
YAML_DIR="$DIR/configs/path.yaml"
S1_JSON_DIR="$DIR/configs/config_s1.json"
S2_JSON_DIR="$DIR/configs/config_s2.json"


$python_cmd ./benchmarks/1_behavior_cls/uwb/train_uwb_s1.py --path_dir "$YAML_DIR" --config_dir "$S1_JSON_DIR"
$python_cmd ./benchmarks/1_behavior_cls/uwb/train_uwb_s2.py --path_dir "$YAML_DIR" --config_dir "$S2_JSON_DIR"

$python_cmd ./benchmarks/1_behavior_cls/immu/train_immu_s0.py --path_dir "$YAML_DIR"
$python_cmd ./benchmarks/1_behavior_cls/immu/train_immu_s1.py --path_dir "$YAML_DIR" --config_dir "$S1_JSON_DIR"
$python_cmd ./benchmarks/1_behavior_cls/immu/train_immu_s2.py --path_dir "$YAML_DIR" --config_dir "$S2_JSON_DIR"

$python_cmd ./benchmarks/1_behavior_cls/uwb_hd/train_uwb_hd_s1.py --path_dir "$YAML_DIR" --config_dir "$S1_JSON_DIR"
$python_cmd ./benchmarks/1_behavior_cls/uwb_hd/train_uwb_hd_s2.py --path_dir "$YAML_DIR" --config_dir "$S2_JSON_DIR"

$python_cmd ./benchmarks/1_behavior_cls/uwb_hd_akl/train_uwb_hd_akl_s1.py --path_dir "$YAML_DIR" --config_dir "$S1_JSON_DIR"
$python_cmd ./benchmarks/1_behavior_cls/uwb_hd_akl/train_uwb_hd_akl_s2.py --path_dir "$YAML_DIR" --config_dir "$S2_JSON_DIR"


# Run S-RGB inference here


# S-RGB and M-RGB evaluation
# $python_cmd ./benchmarks/1_behavior_cls/rgb/eval/eval_srgb_s2.py --path_dir "$YAML_DIR" --config_dir "$S2_JSON_DIR"
# $python_cmd ./benchmarks/1_behavior_cls/rgb/eval/eval_mrgb_s2.py --path_dir "$YAML_DIR" --config_dir "$S2_JSON_DIR"


echo "-------TRAINING HAS BEEN COMPLETED------"
sleep 360000

