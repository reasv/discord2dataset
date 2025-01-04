#!/bin/bash
set -e  # Exit on any error

# Trap errors to provide a message
trap 'echo "An error occurred. Exiting..."; exit 1;' ERR

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Non-interactive apt installation
export DEBIAN_FRONTEND=noninteractive

# Update package list
apt update

# Install required packages if not already installed
REQUIRED_PKGS=("git" "python3" "python3-pip" "tmux" "nvtop")
for pkg in "${REQUIRED_PKGS[@]}"; do
  if ! dpkg -l | grep -qw "$pkg"; then
    echo "Installing package: $pkg"
    apt install -y "$pkg"
  else
    echo "Package '$pkg' is already installed. Skipping."
  fi
done

SESSION_NAME="huggingface_login"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Check if the script is already running inside tmux
if [ -z "$TMUX" ] && [ "$1" != "inside_tmux" ]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching to it."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
  else
    echo "Starting new tmux session: $SESSION_NAME"
    tmux new-session -s "$SESSION_NAME" -d "$SCRIPT_PATH inside_tmux"
    tmux attach-session -t "$SESSION_NAME"
    exit 0
  fi
fi

# Actual script logic inside tmux
echo "Running inside tmux session: $SESSION_NAME"

# Set the repo URL and target directory
REPO_URL="https://github.com/axolotl-ai-cloud/axolotl.git"
TARGET_DIR="axolotl"

# Clone the repository if it doesn't exist
if [ -d "$TARGET_DIR" ]; then
  echo "Directory '$TARGET_DIR' already exists. Skipping clone."
else
  echo "Cloning repository from $REPO_URL..."
  git clone "$REPO_URL" "$TARGET_DIR"
  echo "Repository cloned successfully."
fi

# Change to the repo directory
cd "$TARGET_DIR"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install packaging ninja

# Verify the presence of setup.py or pyproject.toml
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
  pip3 install --no-build-isolation -e '.[flash-attn,deepspeed]'
else
  echo "setup.py or pyproject.toml not found. Skipping pip installation."
  exit 1
fi

# Set the Hugging Face cache directory
export HF_HOME="/workspace/huggingface"
mkdir -p "$HF_HOME"

# Ensure tokens are set via environment variables
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY environment variable is not set."
  exit 1
fi

# Set up Git credentials and login to Hugging Face
git config --global credential.helper store
huggingface-cli login --add-to-git-credential --token "$HF_TOKEN"

# Set up Weights & Biases API key and login
wandb login --relogin "$WANDB_API_KEY"

echo "Setup complete. Keeping the tmux session alive."

# Keep the tmux session alive by starting an interactive shell
bash
