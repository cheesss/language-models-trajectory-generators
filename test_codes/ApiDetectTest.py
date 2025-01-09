from api import API
import argparse
from multiprocessing import Process, Pipe
from lang_sam import LangSAM
from XMem.model.network import XMem
import config
import torch
import multiprocessing
import logging

device = torch.device("cpu")
xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)


parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument("-lm", "--language_model", choices=["gpt-4o-mini", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], default="gpt-4o-mini", help="select language model")
parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
args = parser.parse_args()

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)
langsam_model = LangSAM()
xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)

main_connection, env_connection = Pipe()
api = API(args, main_connection, logger, langsam_model, xmem_model, device)

detect_object = api.detect_object
