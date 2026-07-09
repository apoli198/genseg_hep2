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
save_path = f'./test_HEp2/test-HEp2'
os.makedirs(save_path, exist_ok=True)
opt_path = os.path.join(save_path, "train_options.pkl")
with open(opt_path, "wb") as f:
    pickle.dump(opt, f)
log_file_path = os.path.join(save_path, "output.log")
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

# ---------------------------
# Launch testing subprocess for each fold
# ---------------------------
if __name__ == '__main__':
    for fold in range(0, num_folds):
        command = [
            "python", "running_files/test_hep2_fold.py",
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
# Collect and process metrics from all folds
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
# Calculate and log average unet_score across folds
# ---------------------------
if fold_metrics:
    dice_values = []
    accuracy_values = []
    for fold, metrics in fold_metrics.items():
        if "dice_score" in metrics and "seg_accuracy" in metrics:
            dice = metrics["dice_score"]
            accuracy = metrics["seg_accuracy"]
            dice_values.append(dice.item() if hasattr(dice, "item") else float(dice))
            accuracy_values.append(accuracy.item() if hasattr(accuracy, "item") else float(accuracy))
    if dice_values and accuracy_values:
        avg_dice = statistics.mean(dice_values)
        avg_accuracy = statistics.mean(accuracy_values)
        std_dice = statistics.stdev(dice_values) if len(dice_values) > 1 else 0.0
        std_accuracy = statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0
        logger.info(f"DS: {avg_dice:.2%} ± {std_dice:.2%}")
        logger.info(f"SA: {avg_accuracy:.2%} ± {std_accuracy:.2%}")
    else:
        logger.error("No valid scores values found")
        
logger.info("All folds completed.")
