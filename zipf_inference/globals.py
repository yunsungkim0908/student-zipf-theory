import os

ROOT_DIR=os.path.dirname(__file__)
DATA_DIR=os.path.join(ROOT_DIR, 'dataset')
MODEL_DIR=os.path.join(ROOT_DIR, 'models')
SCORER_DIR=os.path.join(ROOT_DIR, 'scorer')
OUT_DIR=os.path.join(ROOT_DIR, 'out')
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)
