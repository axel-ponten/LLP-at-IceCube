# setup venv from scratch for reconstruction

echo "This script will setup a new virtual environment and install the required packages."
echo "Your current python version is: $(python --version)"
read -p "Are you sure you want to run this script? (y/n): " confirm
if [[ $confirm != "y" ]]; then
    echo "Script execution aborted."
    exit 0
fi


# remove .venv if it exists
if [ -d .venv ]; then
    echo "removing existing .venv"
    rm -rf .venv
fi

# create new venv
echo "creating new venv"
python -m venv .venv

# activate venv
source .venv/bin/activate

# install jammy flows
pip install git+https://github.com/thoglu/jammy_flows.git

# install requirements
pip install -r llp_gap_reco/requirements.txt

# install llp_gap_reco as editable
pip install -e llp_gap_reco

# deactivate venv
echo "venv setup complete. deactivating venv."
deactivate
