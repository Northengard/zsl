import os
import torch
import cv2
import json


def _default_paths_init(cfg):
    """
    initialize output paths such as logs and snapshots inner project directory.
    :return: None
    """
    if not os.path.exists(cfg.SYSTEM.SNAPSHOT_DIR):
        os.mkdir(cfg.SYSTEM.SNAPSHOT_DIR)
    if not os.path.exists(cfg.SYSTEM.LOG_DIR):
        os.mkdir(cfg.SYSTEM.LOG_DIR)


def load_weights(model, optimizer, checkpoint_file):
    """
    Load network snapshot
    :param model: model, torch model to load snapshot
    :param optimizer: optimizer to restore its params
    :param checkpoint_file: path to checkpoint
    :return: None
    """
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['state_dict'])


def save_weights(model, prefix, model_name, epoch, save_dir, optimizer=None, parallel=True):
    """
    Save network snapshot
    :param model: model, torch model to save
    :param prefix: str, snapshot prefix part
    :param model_name: str, model name (identifier)
    :param epoch: int, snapshot epoch
    :param save_dir: str, path to save weights
    :param optimizer: model optimizer
    :param parallel: bool, set true if model learned on multiple gpu
    :return: None
    """
    file = os.path.join(save_dir,
                        '{}_{}_epoch_{}.pth'.format(prefix,
                                                    model_name,
                                                    epoch))
    if parallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    if optimizer:
        checkpoint = {"state_dict": state_dict,
                      "optimizer_state_dict": optimizer.state_dict()}
    else:
        checkpoint = {'state_dict': state_dict}
    torch.save(checkpoint, file)


def load_image(imp_path):
    """
    Load image by given path using opencv-python (cv2)
    :param imp_path: str, path to image
    :return: image
    """
    return cv2.imread(imp_path)


def save_json(path, filename, data):
    """
    save the following json-serialisable data to json file
    :param path: str, directory to save file
    :param filename: str, desired filename must contain '.json' at the end
    :param data: data to save
    :return: None
    """
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as outpf:
        json.dump(data, outpf)
