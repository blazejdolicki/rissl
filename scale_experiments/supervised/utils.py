import logging
import os
import sys

def setup_logging(output_dir):
    """
    Setup various logging streams: stdout and file handlers.
    Suitable for single GPU only.
    """
    # get the filename if we want to log to the file as well
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"{output_dir}/log.txt"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger('PIL').setLevel(logging.WARNING)


def accuracy(predictions, targets):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    else:
        predictions = (predictions > 0.)

    predictions = predictions.to(dtype=targets.dtype)
    acc = float((targets == predictions).sum()) / predictions.numel()
    return acc