import os
import pickle
import logging
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(fname):
    logger.info("Reading in file: " + fname)
    if os.path.exists(fname):
        if fname.endswith(".laser"):
            logger.info("Loading LASER format data.")
            dim = 1024
            data = np.fromfile(fname, dtype=np.float32, count=-1)
            data.resize(data.shape[0] // dim, dim)
        elif fname.endswith(".tl"):
            import torch
            logger.info("Loading Torch format data.")
            data = torch.load(fname)
            data = data.numpy()
        else:
            logger.info("Loading Numpy format data.")
            data = np.loadtxt(fname, delimiter=",")
        logger.info("Loaded data with shape: " + str(data.shape))
        return data
    else:
        logger.info("Input file does not exist")
        raise FileNotFoundError

def save_predictions(y_pred, fname):
    logger.info("Saving predictions to: "+ str(fname)+". Shape of predictions: " + str(y_pred.shape))
    np.savetxt(fname, y_pred, fmt="<cl%d>", delimiter=",", )



class clusterModel:
    def __init__(self, n_clusters, algorithm, seed=24):
        if algorithm.lower() == "kmeans":
            logger.info("Using KMeans clustering algorithm with num_clusters=" + str(n_clusters))
            self.model = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=seed)
        elif algorithm.lower() == "gmm":
            logger.info("Using Gaussian Mixture model algorithm with num_clusters="+ str(n_clusters))
            self.model = GaussianMixture(n_components=n_clusters, random_state=seed)
        else:
            raise NotImplementedError

    def train(self, X):
        logger.info("Training a model")
        self.model.fit(X)

    def predict(self, X):
        logger.info("Generating predictions.")
        return self.model.predict(X)

    def load_model(self, fname):
        logger.info("Loading model from path: "+ str(fname))
        with open(fname, "rb") as fin:
            self.model = pickle.load(fin)

    def save_model(self, fname):
        logger.info("Saving a model to path: " + fname)
        with open(fname, "wb") as fout:
            pickle.dump(self.model, fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=256, help="Number of clusters. Defaults to 256.")
    parser.add_argument('--algorithm', type=str, default='kmeans', help="Algorithm type. Supported: kmeans and gmm (Gaussian Mixture Models).")
    parser.add_argument('--model_file', type=str, required=True, help="Model file. In training mode, this file will be created. In predicting mode, the model will be loaded and used for predictiong")
    parser.add_argument('--in_file', type=str, required=True, help="Input file for training/predicting. N lines and D numbers per line, separated by comma.")
    parser.add_argument('--out_file', type=str, required=True, help="Output file for predictions. N lines with one class per line.")
    parser.add_argument('--predict', action="store_true", help="If this is set, only predicting part is done")
    parser.add_argument('--seed', type=int, default=24, help="Seed value. Defaults to 24.")

    # Parse arguments
    args = parser.parse_args()

    model = clusterModel(n_clusters=args.n_clusters, algorithm=args.algorithm, seed=args.seed)

    logger.info("Reading in the input file.")
    data = load_data(args.in_file)

    if args.predict:
        model.load_model(args.model_file)
        y_pred = model.predict(data)
        save_predictions(y_pred, args.out_file)
    else:
        model.train(data)
        y_pred = model.predict(data)
        save_predictions(y_pred, args.out_file)
        model.save_model(args.model_file)

    logger.info("Clustering finished.")

if __name__ == '__main__':
    main()
