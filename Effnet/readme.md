# EfficientNet Deepfake Detection - Training Script

This code fine-tunes an EfficientNet model for binary deepfake classification (0 = fake, 1 = real). It uses weighted sampling, a freeze/unfreeze training strategy, early stopping, and automatic evaluation.

## Expected Dataset Structure

The model expects the data to be in train, test, and val folders, with subfolders labeled 0 and 1 where 0 contains deepfakes and 1 contains real images. 
---

##  Running on OSCAR


1.  Make sure your environment has been set up ( `venv_effnet/` has been set up in the terminal or wherever you are running the code)
2.  Prepare your data under a `Dataset/` folder, or otherwise and just adjust the code
3.  Submit the batch job via SLURM:
    ```bash
    sbatch kag_effnet.sh
    ```

The `kag_effnet.sh` file includes all the commands to:

* Load Python
* Activate the virtual environment
* Run `train_kag_effnet.py` with appropriate params

## Training output

The script will log validation accuracy for each epoch, test accuracy and classification report, a confusion matrix. It automatically saves the best model and metrics.


##  Notes

* Default image size is 300 (for EfficientNet-B3).
* The training script uses weighted sampling because of class imbalance.
* Early stopping stops training if val acc doesn't improve for `--patience` epochs. This is just to save time since running all the epochs would take probably 12 hours or so.

## Colab visualizations 

* Just run the colab file as you normally would. Just make sure all directories are properly set up in advance to run. 