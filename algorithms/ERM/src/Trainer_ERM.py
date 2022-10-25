import logging
import math
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.ERM.src.dataloaders import dataloader_factory
from algorithms.ERM.src.models import model_factory
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.load_metadata import set_tr_val_samples_labels, set_test_samples_labels
from torchmetrics.functional.classification import multiclass_calibration_error

class Classifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, z):
        y = self.classifier(z)
        return y


class Trainer_ERM:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.exp_idx = exp_idx
        self.checkpoint_name = (
            "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + self.exp_idx
        )
        self.plot_dir = (
            "algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + self.exp_idx + "/"
        )
        self.model = model_factory.get_model(self.args.model)().to(self.device)
        self.classifier = Classifier(self.args.feature_dim, self.args.n_classes).to(self.device)
        self.nn_softmax = nn.Softmax(dim=1)

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def init_training(self):
        logging.basicConfig(
            filename="algorithms/"
            + self.args.algorithm
            + "/results/logs/"
            + self.args.exp_name
            + "_"
            + self.exp_idx
            + ".log",
            filemode="w",
            level=logging.INFO,
        )
        self.writer = self.set_writer(
            log_dir="algorithms/"
            + self.args.algorithm
            + "/results/tensorboards/"
            + self.args.exp_name
            + "_"
            + self.exp_idx
            + "/"
        )
        (
            tr_sample_paths,
            tr_class_labels,
            val_sample_paths,
            val_class_labels,
        ) = set_tr_val_samples_labels(self.args.train_meta_filenames, self.args.val_size)
        self.train_loader = DataLoader(
            dataloader_factory.get_train_dataloader(self.args.dataset)(
                path=self.args.train_path,
                sample_paths=tr_sample_paths,
                class_labels=tr_class_labels,
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        if self.args.val_size != 0:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    path=self.args.train_path,
                    sample_paths=val_sample_paths,
                    class_labels=val_class_labels,
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        else:
            self.val_loader = DataLoader(
                dataloader_factory.get_test_dataloader(self.args.dataset)(
                    path=self.args.train_path, sample_paths=test_sample_paths, class_labels=test_class_labels
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        optimizer_params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.SGD(optimizer_params, lr=self.args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.val_loss_min = np.Inf
        self.val_acc_max = 0

    def init_testing(self):
        test_sample_paths, test_class_labels = set_test_samples_labels(self.args.test_meta_filenames)
        self.test_loader = DataLoader(
            dataloader_factory.get_test_dataloader(self.args.dataset)(
                path=self.args.test_path, sample_paths=test_sample_paths, class_labels=test_class_labels
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
        )

    def train(self):
        self.init_training()
        self.model.train()
        self.classifier.train()
        n_class_corrected, total_classification_loss, total_samples = 0, 0, 0
        self.train_iter_loader = iter(self.train_loader)
        for iteration in range(self.args.iterations):
            if (iteration % len(self.train_iter_loader)) == 0:
                self.train_iter_loader = iter(self.train_loader)
            samples, labels = self.train_iter_loader.next()
            samples, labels = samples.to(self.device), labels.to(self.device)
            predicted_classes = self.classifier(self.model(samples))
            classification_loss = self.criterion(predicted_classes, labels)
            total_classification_loss += classification_loss.item()
            _, predicted_classes = torch.max(predicted_classes, 1)
            n_class_corrected += (predicted_classes == labels).sum().item()
            total_samples += len(samples)
            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()
            if iteration % self.args.step_eval == (self.args.step_eval - 1):
                self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_samples, iteration)
                self.writer.add_scalar("Loss/train", total_classification_loss / self.args.step_eval, iteration)
                logging.info(
                    "Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                        iteration + 1,
                        self.args.iterations,
                        n_class_corrected,
                        total_samples,
                        100.0 * n_class_corrected / total_samples,
                        total_classification_loss / self.args.step_eval,
                    )
                )
                self.evaluate(iteration)
                n_class_corrected, total_classification_loss, total_samples = 0, 0, 0

    def evaluate(self, n_iter):
        self.model.eval()
        self.classifier.eval()
        n_class_corrected, total_classification_loss, total_ece = 0, 0, 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.val_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                predicted_softmaxs = self.nn_softmax(predicted_classes)
                total_ece += multiclass_calibration_error(predicted_softmaxs, labels, num_classes=10, n_bins=10, norm='l1')
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        self.writer.add_scalar("Accuracy/validate", 100.0 * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar("Loss/validate", total_classification_loss / len(self.val_loader), n_iter)
        logging.info(
            "Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}\tECE: {:.6f}".format(
                n_class_corrected,
                len(self.val_loader.dataset),
                100.0 * n_class_corrected / len(self.val_loader.dataset),
                total_classification_loss / len(self.val_loader),
                total_ece / len(self.val_loader),
            )
        )
        val_acc = n_class_corrected / len(self.val_loader.dataset)
        val_loss = total_classification_loss / len(self.val_loader)
        self.model.train()
        self.classifier.train()
        if self.args.val_size != 0:
            if self.val_loss_min > val_loss:
                self.val_loss_min = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )
        else:
            if self.val_acc_max < val_acc:
                self.val_acc_max = val_acc
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
                    },
                    self.checkpoint_name + ".pt",
                )

    def test(self):
        self.init_testing()
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()
        n_class_corrected, total_ece = 0, 0
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.test_loader):
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes = self.classifier(self.model(samples))
                predicted_softmaxs = self.nn_softmax(predicted_classes)
                total_ece += multiclass_calibration_error(predicted_softmaxs, labels, num_classes=10, n_bins=10, norm='l1')
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        logging.info(
            "Test set: Accuracy: {}/{} ({:.2f}%)\tECE: {:.6f}".format(
                n_class_corrected,
                len(self.test_loader.dataset),
                100.0 * n_class_corrected / len(self.test_loader.dataset),
                total_ece / len(self.test_loader),
            )
        )


    def save_plot(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.model.eval()
        self.classifier.eval()
        Z_train, Y_train, Z_test, Y_test = [], [], [], []
        tr_nlls, tr_entropies, te_nlls, te_entropies = [], [], [], []
        with torch.no_grad():
            for iteration, (samples, labels) in enumerate(self.train_loader):
                b, c, h, w = samples.shape
                samples, labels = samples.to(self.device), labels.to(self.device)
                z = self.model(samples)
                predicted_classes = self.classifier(z)
                predicted_softmaxs = self.nn_softmax(predicted_classes)
                for predicted_softmax in predicted_softmaxs:
                    tr_entropies.append(entropy(predicted_softmax.cpu()))
                classification_loss = self.criterion(predicted_classes, labels)
                bpd = (classification_loss.item()) / (math.log(2.0) * c * h * w)
                tr_nlls.append(bpd)
                Z_train += z.tolist()
                Y_train += labels.tolist()
            for iteration, (samples, labels) in enumerate(self.test_loader):
                b, c, h, w = samples.shape
                samples, labels = samples.to(self.device), labels.to(self.device)
                z = self.model(samples)
                predicted_classes = self.classifier(z)
                predicted_softmaxs = self.nn_softmax(predicted_classes)
                for predicted_softmax in predicted_softmaxs:
                    te_entropies.append(entropy(predicted_softmax.cpu()))
                classification_loss = self.criterion(predicted_classes, labels)
                bpd = (classification_loss.item()) / (math.log(2.0) * c * h * w)
                te_nlls.append(bpd)
                Z_test += z.tolist()
                Y_test += labels.tolist()
                
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        with open(self.plot_dir + "Z_train.pkl", "wb") as fp:
            pickle.dump(Z_train, fp)
        with open(self.plot_dir + "Y_train.pkl", "wb") as fp:
            pickle.dump(Y_train, fp)
        with open(self.plot_dir + "Z_test.pkl", "wb") as fp:
            pickle.dump(Z_test, fp)
        with open(self.plot_dir + "Y_test.pkl", "wb") as fp:
            pickle.dump(Y_test, fp)
        with open(self.plot_dir + "tr_nlls.pkl", "wb") as fp:
            pickle.dump(tr_nlls, fp)
        with open(self.plot_dir + "tr_entropies.pkl", "wb") as fp:
            pickle.dump(tr_entropies, fp)
        with open(self.plot_dir + "te_nlls.pkl", "wb") as fp:
            pickle.dump(te_nlls, fp)
        with open(self.plot_dir + "te_entropies.pkl", "wb") as fp:
            pickle.dump(te_entropies, fp)
