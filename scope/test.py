import argparse, os
import torch
from tqdm import tqdm
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import ensure_dir, write_json
from utils.plot_ROC import plot_roc


def main(config, output_dir=None):
    logger = config.get_logger('test')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(config.resume), os.path.basename(config.resume).split('.')[0])
    ensure_dir(output_dir)

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config.config['test_data_path'],
        outcome_file_path=config['data_loader']['args']['outcome_file_path'],
        outcome=config['data_loader']['args']['outcome'],
        channels=config['data_loader']['args']['channels'],
        preload_data=True,
        augmentation=False,
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    config.config['checkpoint'] = str(config.resume)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    prediction_df = pd.DataFrame()

    with torch.no_grad():
        for i, (data, target, subj_id) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            binary_prediction = torch.argmax(output, dim=1)
            positive_class = 1
            soft_prediction = torch.softmax(output, dim=1)[:, positive_class]

            batch_results = {
                'subj-id': subj_id,
                'label': target,
                'soft_prediction': soft_prediction,
                'binary_prediction': binary_prediction,
            }

            for i, metric in enumerate(metric_fns):
                metric_output = metric(output, target)
                total_metrics[i] += metric_output * batch_size
            batch_df = pd.DataFrame(batch_results)
            prediction_df = pd.concat([prediction_df, batch_df], axis=0, ignore_index=True)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    result_dict = {
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    }
    log.update(result_dict)
    logger.info(log)

    plot_roc(soft_prediction=prediction_df['soft_prediction'].tolist(), label=prediction_df['label'].tolist(),
             save_path=os.path.join(output_dir, 'ROC.jpg'))

    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(output_dir, 'result_df.csv'))
    prediction_df.to_csv(os.path.join(output_dir, 'prediction_df.csv'))
    write_json(config.config, os.path.join(output_dir, 'config.json'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-t', '--test-data', default=None, type=str,
                      help='test data file path (default: None)')
    args.add_argument('-o', '--output', default=None, type=str,
                      help='path to output dir (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, save_config=False)
    args = args.parse_args()
    config.config['test_data_path'] = args.test_data
    main(config, args.output)
