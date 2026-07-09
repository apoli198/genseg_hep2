import os
import pickle
import sys
sys.path.append('.')
import logging
import subprocess
import torch
import gc
import signal
import statistics

from options.train_options import TrainOptions

# ---------------------------
# Training options parsing and setting
# ---------------------------
opt = TrainOptions().parse()
assert opt.cuda_index == int(opt.gpu_ids[0]), "GPU indices should be the same"
device = torch.device(f'cuda:{opt.cuda_index}' if torch.cuda.is_available() else 'cpu')
seg_save_path = f'./baseline_HEp2_model/baseline-HEp2-{opt.seg_model}'
os.makedirs(seg_save_path, exist_ok=True)
opt_path = os.path.join(seg_save_path, "train_options.pkl")
with open(opt_path, "wb") as f:
    pickle.dump(opt, f)
log_file_path = os.path.join("./plots/baseline", "output.log")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def cleanup():
    """Cleanup function to free up GPU memory and collect garbage."""
    torch.cuda.empty_cache()
    gc.collect()

current_process = None
num_folds = 5

def signal_handler(sig, frame):
    """Signal handler to terminate subprocesses on interrupt or termination signals."""
    logger.info("Interrupt signal received. Terminating subprocesses and cleaning up...")
    global current_process
    if current_process is not None:
        try:
            current_process.kill()
            logger.info("Subprocess terminated.")
        except Exception as e:
            logger.error(f"Error terminating subprocess: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # ---------------------------
    # Launch segmentation training subprocess for each fold
    # ---------------------------
    for fold in range(0, num_folds):
        command = [
            "python", "running_files/baseline_hep2_fold.py",
            "--fold", str(fold),
            "--opt_path", opt_path,
            "--save_path", seg_save_path,
        ]
        logger.info(f"Starting segmentation training for fold {fold}...")

        current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        while True:
            output = current_process.stdout.readline()
            if output == '' and current_process.poll() is not None:
                break
            if output:
                logger.info(output.strip())

        stderr = current_process.stderr.read()
        if stderr:
            logger.error(stderr.strip())
        
        current_process.wait()
        if current_process.returncode != 0:
            logger.error(f"Fold {fold} failed with exit code {current_process.returncode}. Aborting.")
            break
        cleanup()
        current_process = None

# ---------------------------
# Collect metrics from all folds
# ---------------------------
fold_metrics = {}
for fold in range(0, num_folds):
    metrics_file = os.path.join(f"{seg_save_path}-fold{fold}", "metrics.pkl")
    if os.path.exists(metrics_file):
        with open(metrics_file, "rb") as f:
            fold_metrics[fold] = pickle.load(f)
    else:
        logger.error(f"Metrics file not found for fold {fold}")

# ---------------------------
# Calculate and log average validation loss across folds
# ---------------------------
if fold_metrics:
    loss_values = []
    for fold, metrics in fold_metrics.items():
        if "val_loss" in metrics:
            loss_values.append(metrics["val_loss"])
    if loss_values:
        unet_loss_values = [float(loss) if isinstance(loss, torch.Tensor) else loss for loss in loss_values]
        avg_loss = statistics.mean(unet_loss_values)
        std_loss = statistics.stdev(unet_loss_values) if len(unet_loss_values) > 1 else 0.0
        logger.info(f"Average validation loss: {avg_loss:.4f} ± {std_loss:.4f}")
    else:
        logger.error("No valid loss values found")

logger.info("All segmentation training folds completed.")
