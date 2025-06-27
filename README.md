# Protein Contact Map Prediction with ESM2 + LoRA

This project implements a protein contact map prediction system based on the pre-trained ESM2 language model with LoRA (Low-Rank Adaptation) adapters. The project uses a modern approach to model training with minimal computational resources by fine-tuning only a small portion of parameters.

## Main Results

Main models results:

|                                                                                               |           |           |           |           |           |           |           |
| --------------------------------------------------------------------------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Name                                                                                          | P\_l\@L/5 | P\_l\@L/1 | P\_f\@L/5 | P\_f\@L/1 | P\_f      | Recall\_f | f1\_f     |
| [ESN2-Template: all seq](https://wandb.ai/stepan-v-kuznetsov/deep-origin-task/runs/7bn0eyvu)  | **0.93**  | **0.741** | **0.966** | **0.873** | **0.802** | **0.685** | **0.729** |
| [ESN2-LoRA Conv: all seq](https://wandb.ai/stepan-v-kuznetsov/deep-origin-task/runs/osev9png) | 0.929     | 0.738     | 0.965     | 0.871     | 0.808     | 0.673     | 0.723     |
| [ESN2-LoRA base: all seq](https://wandb.ai/stepan-v-kuznetsov/deep-origin-task/runs/jqojj03g) | 0.901     | 0.683     | 0.948     | 0.826     | 0.78      | 0.595     | 0.664     |
| [ESN2-LoRA base: no homo](https://wandb.ai/stepan-v-kuznetsov/deep-origin-task/runs/w6as1xvg) | 0.855     | 0.588     | 0.92      | 0.75      | 0.699     | 0.503     | 0.574     |
| [ESN2-Rao: no homo](https://wandb.ai/stepan-v-kuznetsov/deep-origin-task/runs/ykthsvvz)       | 0.714     | 0.431     | 0.83      | 0.579     | 0.673     | 0.248     | 0.35      |


## Project Architecture

### Core Components

1. **ESM2 + LoRA**: Pre-trained ESM2 language model with LoRA adapters for efficient fine-tuning
2. **Contact Head**: Specialized head for contact prediction with attention mechanism and APC (Average Product Correction)
3. **DVC Pipeline**: Automated data processing and training pipeline
4. **Hydra configuration**: Configuration based on Hydra
5. **MLOps**: Lightning and DVC integation with Weights & Biases for experiment tracking

### Model Architecture

- **Backbone**: ESM2 (facebook/esm2_t33_650M_UR50D or other variants)
- **Adaptation**: LoRA adapters with rank 8-16 on query, key, value layers
- **Contact Head**: Multi-head attention with symmetrization and APC correction
- **Loss Functions**: BCE or Focal Loss for handling imbalanced data

## Project Structure

```
esm-2-contact-map/
├── data/                           # Data (managed by DVC)
│   ├── train/                      # Training PDB files
│   ├── test/                       # Test PDB files
│   └── *.json                      # Validation metadata
├── pipeline/                       # Main pipeline code
│   ├── conf/                       # Configurations based on Hydra
│   │   ├── config.yaml             # Main configuration
│   │   ├── model/                  # Model configurations
│   │   ├── datamodule/             # Data configurations
│   │   └── stages/                 # Stage configurations
│   ├── src/                        # Source code
│   │   ├── models/                 # Model implementations
│   │   ├── datamodules/            # Data loading and processing
│   │   ├── trainers/               # Training logic
│   │   └── utils/                  # Utilities and metrics
│   ├── dvc.yaml                    # DVC pipeline definition
│   ├── dvc.lock/                   # DVC lock file
│   └── params.yaml                 # Experiment parameters
└── ipynb/                          # Jupyter notebooks for analysis
```

## Installation and Setup

### 1. Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd esm-2-contact-map

# Run the project setup script
bash prepare_project.sh
```

The `prepare_project.sh` script will:
- Configure Git and GCloud
- Install Python dependencies
- Authenticate with Weights & Biases
- Download data via DVC
- Extract data archive

### 2. Manual Installation

If automatic installation doesn't work:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure authentication
wandb login  # For experiment tracking and logging
gcloud auth application-default login --no-browser  # For DVC data storage access, you can change remote storage to your own type and credentials

# Download data
dvc pull ./data/*.dvc  # If you have access to the data, else you can download data manually
unzip ./data/ml_interview_task_data.zip -d ./data/ 
```

### 3. Dependencies

Main packages:
- `transformers` - for ESM2 model
- `peft` - for LoRA adapters
- `lightning` - for training
- `dvc` - for data and pipeline management
- `wandb` - for experiment tracking
- `biopython` - for working with protein structures

### Additional Dependencies

**MMseqs2** is required for sequence clustering functionality. Install following the instructions at [MMseqs2 GitHub](https://github.com/soedinglab/MMseqs2?tab=readme-ov-file#installation). MMseqs2 is used in the pipeline to cluster sequences and prevent data leakage between training and validation sets.

## Usage

### Running the Full Pipeline

There is several ways to run the pipeline:
1. Reproduce the entire pipeline with configuration from params.yaml
```bash
dvc repro
```

You can also run specific stages of the pipeline:
```bash
dvc repro parse_pdb
dvc repro train
```
For more information about DVC, see [DVC Experiments](https://dvc.org/doc/user-guide/).

2. Run new experiments with custom parameters
```bash
cd pipeline

# For example, change model to esm2_lora_contact_600m_focal
dvc exp run -S "model=esm2_lora_contact_600m_focal" -n "test_run"

# Or more specific parameters
dvc exp run -S "model.learning_rate=1e-4" -S "model.lora_rank=16" -n "test_run_2"
```
3. Queue experiments
```bash
dvc exp run --queue -S "model=esm2_lora_contact_600m_focal" -n "test_run"
dvc queue start -j ${number_of_jobs_or_gpus}
dvc queue status
```

### Pipeline Stages
#todo add dag image
1. **parse_pdb**: Extract sequences and structural information from PDB files
2. **filter_sequences**: Filter sequences by length and quality
3. **mmseqs2**: Cluster sequences and compute similarity between them
4. **prepare_dataset**: Prepare datasets for training
5. **train**: Train the model
6. **predict**: Generate predictions
7. **evaluate**: Evaluate model performance


## Configuration

### Models

Available model configurations (in `pipeline/conf/model/`):
- `baseline_unsupervised.yaml` - Baseline model without LoRA (semi-supervised contact prediction)
- `esm2_lora_contact_650m.yaml` - Base model with ESM2-650M and LoRA and contact head
- `esm2_lora_contact_600m_focal.yaml` - Model with Focal Loss and contact head
- `esm2_lora_contactconv_650m.yaml` - Model with lora and contact head with CNN refiner
- `esm2_wolora_contact_650m.yaml` - Model without LoRA (full fine-tuning) - for reference

### Data

Available data configurations (in `pipeline/conf/datamodule/`):
- `base_allseq.yaml` - All sequences
- `base_20seq.yaml` - Only 20 sequences in train, for sime-supervised contact prediction
- `base_allseq_exc_cluster.yaml` - Exclude clusters from training


## Monitoring and Results

### Weights & Biases

All experiments are automatically logged to W&B:
- Training and validation metrics
- Contact map visualizations
- Boxplot and lineplot metrics
- Hyperparameters and configurations

### Metrics

Main evaluation metrics:
- **Precision@L/5, L/2, L**: Precision for top contacts at different sequence separation ranges:
  - Short range: 6-12 residues
  - Medium range: 12-24 residues 
  - Long range: >24 residues
  - Full range: >6 residues
- **AUPR**: Area Under Precision-Recall curve
- **Precision ,Recall, F1-score**: Default metrics for different ranges

### Results

Results are saved in:
- `pipeline/output/train_checkpoints/` - Model checkpoints
- `pipeline/output/predictions/` - Model predictions
- `pipeline/output/evaluation/` - Evaluation results

## Performance

### System Requirements

- **GPU**: Recommended 16+ GB VRAM (RTX 3090, A100, V100)
- **RAM**: 32+ GB for processing large batches
- **Disk**: 50+ GB free space

### Optimization

- Use gradient accumulation (`accumulate_grad_batches`) for larger effective batches
- LoRA significantly reduces memory requirements compared to full fine-tuning

## Project Development

### Adding New Models or DataModules

1. Create a new file in `pipeline/src/models/`
2. Register the model using `@register_model` decorator
3. Add configuration in `pipeline/conf/model/`

### Adding New Metrics

1. Extend the `ContactPredictionMetrics` class in `utils/metrics.py`
2. Add logging to the model

### Custom Loss Functions

Implement new loss functions in the model and add them to configuration.

## License

The project uses open components under respective licenses (MIT, Apache 2.0).
