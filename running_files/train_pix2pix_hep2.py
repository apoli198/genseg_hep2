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
save_path = f'./pix2pix_HEp2_model/pix2pix-HEp2'
os.makedirs(save_path, exist_ok=True)
opt_path = os.path.join(save_path, "train_options.pkl")
with open(opt_path, "wb") as f:
    pickle.dump(opt, f)
log_file_path = os.path.join("./plots/pix2pix", "output.log")
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
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    # ---------------------------
    # Launch training subprocess for each fold
    # ---------------------------
    for fold in range(0, num_folds):
        command = [
            "python", "running_files/train_pix2pix_hep2_fold.py",
            "--fold", str(fold),
            "--opt_path", opt_path,
            "--save_path", save_path
        ]
        
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
    metrics_file = os.path.join(f"{save_path}-fold{fold}", "metrics.pkl")
    if os.path.exists(metrics_file):
        with open(metrics_file, "rb") as f:
            fold_metrics[fold] = pickle.load(f)
    else:
        logger.error(f"Metrics file not found for fold {fold}")

# ---------------------------
# Calculate and log average validation losses for G and D
# ---------------------------
if fold_metrics:
    loss_G_values = []
    loss_D_values = []
    for fold, metrics in fold_metrics.items():
        if "val_G" in metrics and "val_D" in metrics:
            loss_G_values.append(metrics["val_G"])
            loss_D_values.append(metrics["val_D"])
    if loss_G_values and loss_D_values:
        avg_loss_G = statistics.mean(loss_G_values)
        std_loss_G = statistics.stdev(loss_G_values) if len(loss_G_values) > 1 else 0.0
        avg_loss_D = statistics.mean(loss_D_values)
        std_loss_D = statistics.stdev(loss_D_values) if len(loss_D_values) > 1 else 0.0
        logger.info(f"Average validation loss G: {avg_loss_G:.4f} ± {std_loss_G:.4f}")
        logger.info(f"Average validation loss D: {avg_loss_D:.4f} ± {std_loss_D:.4f}")
    else:
        logger.error("No valid loss values found")
        
logger.info("All folds completed.")
