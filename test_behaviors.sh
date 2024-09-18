# @Author: mac
# @Date:   2024-05-09 23:13:16
# @Last Modified by:   mac
# @Last Modified time: 2024-06-12 10:04:34


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

$python_cmd ./benchmarks/2_beahvior_analysis/test_uwb_hd_akl.py --path_dir "$YAML_DIR"
$python_cmd ./benchmarks/2_beahvior_analysis/eval_behavior_stats.py --path_dir "$YAML_DIR"


echo "-------TESTING HAS BEEN COMPLETED------"
sleep 36000

