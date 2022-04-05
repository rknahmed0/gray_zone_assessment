"""A test function."""
import click
import json
import os
import pandas as pd
import torch

from gray_zone.loader import loader, get_unbalanced_loader
from gray_zone.utils import load_transforms
from gray_zone.models.model import get_model
from gray_zone.train_rakin import train # 02/2022 changed source of import for train i.e. instead of importing from gray_zone.train import from gray_zone.train_rakin
from gray_zone.evaluate import evaluate_model
from gray_zone.loss_rakin import get_loss # 03/13/2022 changed source of import for get_loss i.e. instead of importing from gray_zone.loss import from gray_zone.loss_rakin
from gray_zone.records import get_job_record, save_job_record
from gray_zone.process_model_output import process_output

def _run_model(output_path: str,
               param_path: str,
               data_path: str,
               csv_path: str,
               label_colname: str,
               image_colname: str,
               split_colname: str,
               patient_colname: str,
               transfer_learning: str,
               test: bool,
               num_class: int,
               make_plots: bool) -> None:
    """ Run deep learning model for training and evaluation for classification tasks. """
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Create directory for checkpoints if it doesn't exist
    path_to_checkpoints = os.path.join(output_path, "checkpoints")
    if not os.path.isdir(path_to_checkpoints):
        os.makedirs(path_to_checkpoints)

    # Save configuration file in output directory
    param_dict = json.load(open(param_path, 'r'))
    df = pd.read_csv(csv_path)
    ##################################################################################################
    if test: 
        param_dict['n_class'] = int(num_class) # num_class provided by click.option n_class
    else:
        param_dict['n_class'] = int(df[label_colname].max() + 1)
    ##################################################################################################
    json.dump(param_dict, open(os.path.join(output_path, "params.json"), 'w'), indent=4)

    # Record environment and CLI
    job_record = get_job_record(param_dict['seed'])
    save_job_record(output_path, record=job_record, name='train_record.json')

    # Convert transforms from config file
    train_transforms = load_transforms(param_dict["train_transforms"])
    val_transforms = load_transforms(param_dict["val_transforms"])

    # Get train, val, test loaders and test dataframe
    weights = (torch.Tensor(param_dict['weights'])).float() if "weights" in param_dict else None

    is_balanced = param_dict['is_weighted_sampling'] or param_dict['is_weighted_loss']
    train_loader, val_loader, test_loader, val_df, test_df, weights = loader(data_path=data_path,
                                                                             output_path=output_path,
                                                                             train_transforms=train_transforms,
                                                                             val_transforms=val_transforms,
                                                                             metadata_path=csv_path,
                                                                             label_colname=label_colname,
                                                                             image_colname=image_colname,
                                                                             split_colname=split_colname,
                                                                             patient_colname=patient_colname,
                                                                             train_frac=param_dict['train_frac'],
                                                                             test_frac=param_dict['test_frac'],
                                                                             seed=param_dict['seed'],
                                                                             batch_size=param_dict['batch_size'],
                                                                             balanced=is_balanced,
                                                                             weights=weights)

    # Get model
    img_dim = list(test_loader.dataset.__getitem__(0)[0].size())
    model, act = get_model(architecture=param_dict['architecture'],
                           model_type=param_dict['model_type'],
                           dropout_rate=param_dict['dropout_rate'],
                           n_class=param_dict['n_class'],
                           device=param_dict['device'],
                           transfer_learning=transfer_learning,
                           img_dim=img_dim)

    optimizer = torch.optim.Adam(model.parameters(), param_dict['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_function = get_loss(param_dict['loss'], param_dict['n_class'], param_dict['foc_gamma'], param_dict['foc_kappa_adjustment'],
                            param_dict['foc_coeff'], param_dict['kappa_coeff'], param_dict['is_weighted_loss'], weights, param_dict['device']) 
                            # 4 additional parameters added: foc_gamma (used in foc, foc_qwk, foc_qwk_LC losses),
                            # foc_kappa_adjustment (used in foc_qwk and foc_qwk_LC losses), foc_coeff, kappa_coeff (used in foc_qwk_LC loss)

    if not test:
        train(model=model,
              act=act,
              train_loader=train_loader,
              val_loader=val_loader,
              loss_function=loss_function,
              optimizer=optimizer,
              device=param_dict['device'],
              n_epochs=param_dict['n_epochs'],
              output_path=output_path,
              scheduler=scheduler,
              n_class=param_dict['n_class'],
              model_type=param_dict['model_type'],
              val_metric=param_dict['val_metric'],
              make_plots=make_plots)

    val_loader = get_unbalanced_loader(val_df, data_path, param_dict['batch_size'], val_transforms,
                                       label_colname, image_colname)
    for data_loader, data_df, suffix in zip([test_loader, val_loader], [test_df, val_df], ['', '_validation']):
        if data_loader:
            df = evaluate_model(model=model,
                                loader=data_loader,
                                output_path=output_path,
                                device=param_dict['device'],
                                act=act,
                                transforms=val_transforms,
                                df=data_df,
                                is_mc=param_dict['dropout_rate'] > 0,
                                image_colname=image_colname,
                                suffix=suffix)
            is_ordinal = param_dict['model_type'] == 'ordinal'
            process_output(output_path, is_ordinal, "predictions" + suffix + ".csv", n_class=param_dict['n_class'])

########################################################################################################
# Class to ensure that --test flag is always accompanied by the --num-class flag i.e. mutually inclusive
class RequiredIf(click.Option):
    def __init__(self, *args, **kwargs):
        self.required_if = kwargs.pop('required_if')
        assert self.required_if, "'required_if' parameter required"
        kwargs['help'] = (kwargs.get('help', '') +
            ' NOTE: This argument is mutually inclusive with %s' %
            self.required_if
        ).strip()
        super(RequiredIf, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        we_are_present = self.name in opts
        other_present = self.required_if in opts

        if other_present:
            if we_are_present:
                self.prompt = None
            # if not(we_are_present):
            #     raise click.UsageError(
            #         "Illegal usage: `%s` is mutually inclusive with `%s`" % (
            #             self.name, self.required_if))
            else:
                self.prompt = "Error: `%s` not entered. `%s` is mutually inclusive with `%s`. Enter `%s` " % (
                        self.name, self.name, self.required_if, self.name)
        else:
            self.prompt = None

        return super(RequiredIf, self).handle_parse_result(
            ctx, opts, args)
#######################################################################################################


@click.command()
@click.option('--output-path', '-o', required=True, help='Output path.')
@click.option('--param-path', '-p', required=True, help='Path to parameter file (.json).')
@click.option('--data-path', '-d', required=True, help='Path to data (directory where images are saved).')
@click.option('--csv-path', '-c', required=True, help='Path to csv file containing image name and labels.')
@click.option('--label-colname', '-lc', default='label', help='Column name in csv associated to the labels.')
@click.option('--image-colname', '-ic', default='image', help='Column name in csv associated to the image.')
@click.option('--split-colname', '-sc', default='dataset',
              help="Column name in csv associated to the train, val, test splits. Each image needs to be associated "
                   "with `val`,`train`, or `test`")
@click.option('--patient-colname', '-pc', default='patient',
              help='Column name in csv associated to the patient id.')
@click.option('--transfer-learning', '-tf', default=None, help="Path to model (.pth) for fine-tune training (i.e., "
                                                               "start training with weights from other model.)")
@click.option('--test', '-test', default=None, is_flag=True, help="Option to test already trained model.")
# @click.option('--num-class', '-n', default=None, help="# of classes if running inference and csv has no labels, REQUIRED if test")
@click.option('--num-class', '-n', default=None, prompt=True, cls=RequiredIf, required_if='test', 
              help="# of classes if running inference (--test) and csv has no ground truth labels\
              only REQUIRED if --test is true. The RequiredIf class ensures that --test and --num-class are mutually inclusive")
@click.option('--make-plots', '-mp', default=None, is_flag=True, help="Option to store losses to allow for plotting learning curves")
def run_model(output_path: str,
              param_path: str,
              data_path: str,
              csv_path: str,
              label_colname: str,
              image_colname: str,
              split_colname: str,
              patient_colname: str,
              transfer_learning: str,
              test: bool,
              num_class: int,
              make_plots: bool) -> None:
    """Train deep learning model using CLI. """
    _run_model(output_path, param_path, data_path, csv_path, label_colname, image_colname, split_colname,
               patient_colname, transfer_learning, test, num_class, make_plots)


if __name__ == "__main__":
    run_model()
