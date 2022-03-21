# Gray Zone Assessment

## Get started

1. Clone github repository   
```bash
git clone https://github.com/rknahmed0/gray_zone_assessment.git
```
2. Build docker image  
```bash
docker build -t gray_zone docker/
```
3. Run docker container  
```bash
docker run -it -v tunnel/to/local/folder:/tunnel --gpus 0 gray_zone:latest bash
```
4. Run the following command at the root of the repository to install the modules  
```bash
cd path/to/gray_zone_assessment
```
```bash
pip install -e .
```
5. Train model  
```bash
python run_model_rakin.py -o <outpath/path> -p <resources/training_configs/config.json> -d <image/data/path> -c <path/csv/file.csv>
```
For more information on the different flags: `python run_model_rakin.py --help`. Additional options/flags added to store loss values in a csv (for subsequent plotting of train/val learning curves), and to allow for user input for num_class when running inference, since the csv file which is used for inference (--test) may contain an empty ground truth columns. Updated run_model.py script stored as run_model_rakin.py (if you want previous version, use run_model.py from https://github.com/rknahmed0/gray_zone_assessment/tree/main/gray_zone/old_versions - run_model.py script here references train.py and loss.py so make sure you move all 3 to your gray_zone directory first in your local clone)

For binary classification, process_model_output.py updated to account for determining 'predicted_class'/'predicted_mc_class' and 'soft_prediction'/'soft_mc_prediction' from 1-element lists of pred and mc_pred, using similar functions as used for ordinal classification.
  
## Configuration file (flag -p or --param-path)  
The configuration file is a json file containing the main training parameters.  
Some json file examples are located in `gray_zone/resources/training_configs/`  

### Required configuration parameters
Sample config file with all new configs: https://github.com/rknahmed0/gray_zone_assessment/blob/main/gray_zone/resources/training_configs/foc_qwk_LC_classification_all_params.json  

|    Parameter   | Description |  
| -------- | --- |  
| architecture |   Architecture id contained in Densenet or Resnet family. Choice between: 'densenet121', 'densenet169', 'densenet201', 'densenet264', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'    |  
| model_type | Choice between "classification", "ordinal", "regression". |  
| loss |  Loss function id. Choice between 'ce' (Cross entropy), 'mse' (Mean square error), 'l1' (L1), 'bce' (Binary cross entropy), 'coral' (Ordinal loss), 'qwk' (Quadratic weighted kappa), 'foc' (Focal), 'foc_qwk' (Focal adjustment to qwk), 'foc_qwk_LC (Linear combination of focal and qwk with coefficients for each term)'. |  
| foc_gamma | gamma parameter for foc, foc_qwk, foc_qwk_LC losses (float or None). |  
| foc_kappa_adjustment | focal adjustment to qwk: either None or 'num' or 'num_denom' for foc_qwk and foc_qwk_LC losses (str or None). |  
| foc_coeff | coefficient for focal loss term for each ground truth class for foc_qwk_LC loss (list len=3 or None). |  
| kappa_coeff | coefficient for kappa loss term for each ground truth class for foc_qwk_LC loss (list len=3 or None). |  
| batch_size | Batch size (int). |  
| lr | Learning rate (float). |  
| n_epochs | Number of training epochs (int). |  
| device | Device id (e.g., 'cuda:0', 'cpu') (str).  |   
| val_metric | Choice between "auc" (average ROC AUC over all classes), "val_loss" (minimum validation loss), "kappa" (linear Cohen's kappa), default "accuracy". |  
| dropout_rate | Dropout rate (Necessary for Monte Carlo model's). A dropout rate of 0 will disable dropout. (float). |  
| is_weighted_loss | Indicates if the loss is weighted by the number of cases by class (bool). |  
| is_weighted_sampling |  Indicates if the sampling is weighted by the number of cases by class (bool). |  
| seed | Random seed (int).  |  
| train_frac | Fraction of cases used for training if splitting not already done in csv file, or else the parameter is ignored (float). |  
| test_frac | Fraction of cases used for testing if splitting not already done in csv file, or else the parameter is ignored (float). |  
| train_transforms / val_transforms | monai training / validation transforms with parameters. Validation transforms are also used during testing (see https://docs.monai.io/en/latest/transforms.html for transform list)  |  


## csv file (flag -c or --csv-path)
The provided csv file contains the filename of the images used for training, GT labels (int from 0-n_class), patient ID 
(str) and split column (containing 'train', 'val' or 'test') (optional). 

Example of csv file with the default column names. If the column names are different from the default values,
the flags `--label-colname`, `--image-colname`, `--patient-colname`, and `--split-colname` can 
be used to indicate the custom column names. There can be more columns in the csv file. All this
metadata will be included in `predictions.csv` and `split_df.csv`.

|    image   | label | patient  |  dataset  |
| :--------: | :---: | :------: |  :------: |  
| patient1_000.png |   0   |  patient1  |    train  |
| patient1_001.png |   0   |  patient1  |    train  |
| patient2_000.png |   2   |  patient2  |    val  |
| patient2_001.png |   2   |  patient2  |    val  |
| patient2_002.png |   2   |  patient2  |    val  |
| patient3_000.png |   1   |  patient3  |    test  |
| patient3_001.png |   1   |  patient3  |    test  |

## Output directory (flag -o or --output-path)
```

└── output directory                # Output directory specified with `-o`  
    ├──   checkpoints               # All models (one .pth per epoch)  
    |     ├──  checkpoint0.pth   
    |     ├──  ...  
    |     └──  checkpointn.pth   
    ├──   best_metric_model.pth     # Best model based on validation metric  
    ├──   params.json               # Parameters used for training (configuration file)  
    ├──   predictions.csv           # Test results  
    ├──   split_df.csv              # csv file containing image filenames, labels, split and patient id  
    └──   train_record.json         # Record of CLI used to train and other info for reproducibility  
```
