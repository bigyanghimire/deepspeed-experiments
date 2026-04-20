
echo "=== Setting up Micromamba and Environment ==="

# Install micromamba to user's home directory
PROJECT_DIR="$SCRATCH/bigyan_project"
mkdir -p $PROJECT_DIR
MICROMAMBA_DIR="$PROJECT_DIR/.local/bin"
mkdir -p "$MICROMAMBA_DIR"

echo "Downloading micromamba..."
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C "$MICROMAMBA_DIR" --strip-components=1 bin/micromamba

# Verify installation
if [ ! -f "$MICROMAMBA_DIR/micromamba" ]; then
    echo "ERROR: Micromamba installation failed"
    exit 1
fi

echo "Micromamba installed successfully at $MICROMAMBA_DIR/micromamba"

# Initialize micromamba
eval "$("$MICROMAMBA_DIR/micromamba" shell hook -s bash)"

# Create environment
ENV_NAME="ds-hf"
echo "Creating environment: $ENV_NAME"

micromamba create -n $ENV_NAME python=3.11.5 -y
micromamba activate $ENV_NAME
micromamba install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge -y
# micromamba install gcc=9.3.0 gxx=9.3.0 -c conda-forge
cd $PROJECT_DIR # Now we are in bigyan_project
git clone git@github.com:bigyanghimire/deepspeed-mod.git
cd deepspeed-mod
python -m pip install -e . --no-build-isolation
pip show deepspeed
cd $PROJECT_DIR
git clone git@github.com:huggingface/transformers.git
cd transformers
git checkout v4.51.3
python -m pip install -e .
cd $PROJECT_DIR
git clone git@github.com:bigyanghimire/deepspeed-experiments.git