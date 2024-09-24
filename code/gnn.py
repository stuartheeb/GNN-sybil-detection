import copy

import numpy as np
import torch_geometric.nn

from algorithms import *

import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, MessagePassing, SimpleConv, RGCNConv, APPNP, SAGEConv
from torch_geometric.nn import aggr
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn import metrics


# SEED = 1234


class SybilGNN(SybilFinder):
    def __init__(self, graph: Graph = None,
                 honest_nodes: [int] = None,
                 sybil_nodes: [int] = None,
                 threshold: float = 0.5,
                 pretrained_algorithm=None,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 train_model: bool = True,
                 fine_tune: bool = False,
                 num_epochs: int = 200,
                 patience: int = None,
                 input_width: int = 2,
                 num_layers: int = None,
                 hidden_width: int = 2,
                 num_classes: int = 2,
                 dropout: bool = True,
                 name: str = None) -> None:
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         uses_directed_graph=True,
                         uses_honest_nodes=True,
                         uses_sybil_nodes=True,
                         has_trust_values=True)
        if name is None:
            self.name = "SybilGNN"
        else:
            self.name = name

        self.threshold = threshold

        self.pretrained_algorithm = pretrained_algorithm
        if self.pretrained_algorithm is not None:
            model = self.pretrained_algorithm.model
            criterion = self.pretrained_algorithm.criterion
            optimizer = self.pretrained_algorithm.optimizer

        self.model = model
        self.initial_model = copy.deepcopy(self.model)
        self.criterion = criterion
        self.initial_criterion = copy.deepcopy(self.criterion)
        self.optimizer = optimizer
        self.initial_optimizer = copy.deepcopy(self.optimizer)

        self.train_model = train_model or fine_tune
        self.fine_tune = fine_tune

        self.num_epochs = num_epochs
        self.patience = patience if patience is not None else num_epochs

        self.train_losses = None
        self.val_losses = None

        self.input_width = input_width
        self.num_layers = num_layers
        self.hidden_width = hidden_width
        self.num_classes = num_classes

        self.dropout = dropout

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.device = torch.device("cpu")  # TODO delete, currently here as an override

    def _description(self):
        return "Base class"

    def set_graph(self, graph) -> None:
        super().set_graph(graph)
        self._transform_to_pyg_graph(graph)

    def _transform_to_pyg_graph(self, graph: Graph):  # TODO This function might be better placed in utils.py
        self.pyg_graph = from_networkx(graph.graph).to(self.device)

    def _setup_edge_type(self):
        raise Exception("No edge types")

    def generate_masks(self, indices, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

        num_nodes = self.graph.num_nodes()

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if test_ratio > 0:
            train_val_indices, test_indices = train_test_split(indices,
                                                               test_size=test_ratio)
            test_mask[test_indices] = True
        else:
            train_val_indices = indices

        # Handle the case where either train_ratio or val_ratio is 0
        if val_ratio > 0 and train_ratio > 0:
            train_indices, val_indices = train_test_split(train_val_indices,
                                                          test_size=val_ratio / (train_ratio + val_ratio))
            train_mask[train_indices] = True
            val_mask[val_indices] = True
        elif val_ratio == 0:
            train_mask[train_val_indices] = True
        elif train_ratio == 0:
            val_mask[train_val_indices] = True

        return train_mask, val_mask, test_mask

    def _setup_pyg_graph(self):
        if self.verbose:
            print("Setting up pyg_graph.x")
        if self.input_width == 1:
            x_values = torch.full(size=(self.pyg_graph.num_nodes, 1), fill_value=0.0, dtype=torch.float32,
                                  device=self.device)

            for i in range(self.graph.num_nodes()):
                if self.uses_honest_nodes:
                    if self.train_labels[i] == -1:
                        x_values[i] = -1.0
                if self.uses_sybil_nodes:
                    if self.train_labels[i] == 1:
                        x_values[i] = 1.0
        elif self.input_width == 2:
            x_values = torch.full(size=(self.pyg_graph.num_nodes, 2), fill_value=0.0, dtype=torch.float32,
                                  device=self.device)
            # TODO : [0, 1] (one-hot) or [-1, 1] ??? (for honest, vice-versa for sybil)

            for i in range(self.graph.num_nodes()):
                if self.uses_honest_nodes:
                    if self.train_labels[i] == -1:
                        x_values[i] = torch.tensor([0, 1], device=self.device)
                if self.uses_sybil_nodes:
                    if self.train_labels[i] == 1:
                        x_values[i] = torch.tensor([1, 0], device=self.device)
        else:
            raise Exception("Input width not supported")

        if self.verbose:
            print("Setting up pyg_graph.y")
        if self.num_classes == 1:
            y_values = torch.full(size=(self.pyg_graph.num_nodes, 1), fill_value=0.0, dtype=torch.long,
                                  device=self.device)

            for i in range(self.graph.num_nodes()):
                if self.uses_honest_nodes:
                    if self.train_labels[i] == -1:
                        y_values[i] = 0.0
                if self.uses_sybil_nodes:
                    if self.train_labels[i] == 1:
                        y_values[i] = 1.0
        elif self.num_classes == 2:
            y_values = torch.full(size=(self.pyg_graph.num_nodes, 2), fill_value=0.0, dtype=torch.long,
                                  device=self.device)

            for i in range(self.graph.num_nodes()):
                if self.uses_honest_nodes:
                    if self.train_labels[i] == -1:
                        y_values[i] = torch.tensor([0, 1], device=self.device)
                if self.uses_sybil_nodes:
                    if self.train_labels[i] == 1:
                        y_values[i] = torch.tensor([1, 0], device=self.device)
        else:
            raise Exception("Num. classes not supported")

        if not self.train_model and isinstance(self.model, SybilFinderRGCN):
            if self.verbose:
                print("Setting up pyg_graph.edge_type")
            self._setup_edge_type()

        self.pyg_graph.x = x_values.to(self.device)
        self.pyg_graph.y = y_values.to(self.device)

        # self.pyg_graph.train_mask = [True if ((self.uses_sybil_nodes and node in self._sybil_nodes) or (
        #        self.uses_honest_nodes and node in self._honest_nodes)) else False for node
        #                             in self.graph.nodes_list()]

        train_ratio = 0.8
        val_ratio = 0.2

        if not self.train_model:
            train_ratio = 0.9
            val_ratio = 0.1  # This is just used for optimal threshold calculation

        num_nodes = self.graph.num_nodes()
        honest_indices = np.array(
            [i for i in range(num_nodes) if np.isclose(self.train_labels[i], -1)]
        )
        honest_train_mask, honest_val_mask, honest_test_mask = self.generate_masks(honest_indices,
                                                                                   train_ratio=train_ratio,
                                                                                   val_ratio=val_ratio,
                                                                                   test_ratio=0.0)

        sybil_indices = np.array(
            [i for i in range(num_nodes) if np.isclose(self.train_labels[i], 1)]
        )
        sybil_train_mask, sybil_val_mask, sybil_test_mask = self.generate_masks(sybil_indices,
                                                                                train_ratio=train_ratio,
                                                                                val_ratio=val_ratio,
                                                                                test_ratio=0.0)

        self.pyg_graph.train_mask = honest_train_mask | sybil_train_mask
        self.pyg_graph.val_mask = honest_val_mask | sybil_val_mask
        self.pyg_graph.test_mask = honest_test_mask | sybil_test_mask

        if self.verbose:
            print(
                f"number train / val / test : {self.pyg_graph.train_mask.sum()} / {self.pyg_graph.val_mask.sum()} / {self.pyg_graph.test_mask.sum()}")

        self.pyg_graph = self.pyg_graph.to(self.device)

    def reinitialize(self):
        self.train_losses = None
        self.val_losses = None
        if self.verbose:
            print(f"Reinitializing {self.name}")
        self.model = copy.deepcopy(self.initial_model)
        self.criterion = copy.deepcopy(self.initial_criterion)
        self.optimizer = copy.deepcopy(self.initial_optimizer)

    def _model_initialization(self):
        raise Exception("This is the base SybilGNN class, it has no functionality")

    def _criterion_initialization(self):
        raise Exception("This is the base SybilGNN class, it has no functionality")

    def _optimizer_initialization(self):
        raise Exception("This is the base SybilGNN class, it has no functionality")

    def _check_model(self):
        try:
            _, _ = self.apply_model(training_mode=True)
            _, _ = self.apply_model(training_mode=False)
        except Exception:
            raise Exception("Model is not compatible with the graph")

    def apply_model(self, training_mode: bool):
        self.model.train(training_mode)  # model.train(False) is equivalent to model.eval()
        if isinstance(self.model, RGCN):
            out, h = self.model(self.pyg_graph.x, self.pyg_graph.edge_index, self.pyg_graph.edge_type)
        elif isinstance(self.model, GCN) or isinstance(self.model, GAT):
            out, h = self.model(self.pyg_graph.x, self.pyg_graph.edge_index)
        else:
            raise Exception("Unknown model")
        return out, h

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out, h = self.apply_model(training_mode=True)
        loss = self.criterion(out[self.pyg_graph.train_mask], self.pyg_graph.y[self.pyg_graph.train_mask].type_as(out))
        loss.backward()
        self.optimizer.step()
        return out, loss.item()

    def test(self, mask):
        self.model.eval()
        out, h = self.apply_model(training_mode=False)
        loss = self.criterion(out[mask], self.pyg_graph.y[mask].type_as(out))

        y_pred = self.get_predictions(out)
        pred = (y_pred > self.threshold)
        correct = pred[mask] == self.pyg_graph.y[mask, 0].detach().numpy()
        accuracy = correct.sum().item() / mask.sum().item()
        # accuracy = (pred[mask] == self.pyg_graph.y[mask]).float().mean().item()

        FPR, TPR, thresholds = metrics.roc_curve(self.pyg_graph.y[mask, 0],
                                                 torch.tensor(y_pred[mask]))
        AUC = metrics.auc(FPR, TPR)

        return loss.item(), accuracy, AUC

    def get_predictions(self, out):
        if self.num_classes == 1:
            y_pred = out.detach().cpu().numpy().flatten()
        else:
            y_pred = out[:, 0].detach().cpu().numpy()
        return y_pred

    def _find_sybils(self) -> list[int]:

        self._setup_pyg_graph()

        if self.model is None:
            self._model_initialization()
        if self.criterion is None:
            self._criterion_initialization()
        if self.optimizer is None:
            self._optimizer_initialization()

        self._check_model()

        train_losses = []
        train_accs = []
        train_AUCs = []
        val_losses = []
        val_accs = []
        val_AUCs = []

        best_val_loss = np.inf
        epochs_without_improvement = 0
        out = None
        saved_model = None
        if self.train_model and self.num_epochs > 0:
            for epoch in range(self.num_epochs):
                out, loss = self.train()

                val_loss, val_accuracy, val_AUC = self.test(self.pyg_graph.val_mask)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # if self.verbose:
                    #    print(f"New best score, saving model")
                    # torch.save(self.model.state_dict(), 'models/temp/best_model.pth')
                    saved_model = copy.deepcopy(self.model)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"Lost patience after {epoch} epochs, loading best model")
                    # self.model.load_state_dict(torch.load('models/temp/best_model.pth'))
                    self.model = saved_model
                    out, _ = self.apply_model(training_mode=False)
                    break

                train_loss, train_accuracy, train_AUC = self.test(self.pyg_graph.train_mask)

                train_losses.append(train_loss)
                train_accs.append(train_accuracy)
                train_AUCs.append(train_AUC)
                val_losses.append(val_loss)
                val_accs.append(val_accuracy)
                val_AUCs.append(val_AUC)

                if self.verbose and (epoch % 10 == 0 or epoch == self.num_epochs - 1):
                    print(
                        f"epoch = {epoch:03d}:\tloss = {loss:8f},\tval_loss = {val_loss:8f},\t val_accuracy = {val_accuracy:8f},\t val_AUC = {val_AUC:8f},\tepochs_without_improvement = {epochs_without_improvement}")
        else:
            out, _ = self.apply_model(training_mode=False)

        self.train_losses = train_losses
        self.val_losses = val_losses

        y_pred = self.get_predictions(out)

        self.trust_values = -y_pred

        if False:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.title('Loss over time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(1, 3, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(val_accs, label='Validation Accuracy')
            plt.legend()
            plt.title('Accuracy over time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

            plt.subplot(1, 3, 3)
            plt.plot(train_AUCs, label='Train AUC')
            plt.plot(val_AUCs, label='Validation AUC')
            plt.legend()
            plt.title('AUC over time')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')

            plt.tight_layout()
            plt.savefig(f"output/training_plots/{self.name}.pdf")
            plt.close()

        FPR, TPR, thresholds = metrics.roc_curve(self.pyg_graph.y[self.pyg_graph.val_mask, 0],
                                                 torch.tensor(y_pred[self.pyg_graph.val_mask]))
        val_AUC = metrics.auc(FPR, TPR)

        val_optimal_threshold = thresholds[np.argmax(TPR - FPR)]
        if self.verbose:
            print(f"val AUC = {val_AUC}")
            print(f"val optimal threshold = {val_optimal_threshold}")

        # TODO: Build threshold into the GNN ?? (but then we need access to test data ??)
        # return self.sybil_classification(y_pred, self.threshold)
        return self.sybil_classification(y_pred, val_optimal_threshold)


class SybilFinderGCN(SybilGNN):
    def __init__(self, graph: Graph = None,
                 honest_nodes: [int] = None,
                 sybil_nodes: [int] = None,
                 threshold: float = 0.5,
                 pretrained_algorithm=None,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 train_model: bool = True,
                 fine_tune: bool = False,
                 num_epochs: int = 200,
                 patience: int = None,
                 input_width: int = 1,
                 num_layers: int = None,
                 hidden_width: int = 1,
                 num_classes: int = 2,
                 dropout: bool = False,
                 name: str = None) -> None:
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         threshold=threshold,
                         pretrained_algorithm=pretrained_algorithm,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         train_model=train_model,
                         fine_tune=fine_tune,
                         num_epochs=num_epochs,
                         patience=patience,
                         input_width=input_width,
                         num_layers=num_layers,
                         hidden_width=hidden_width,
                         num_classes=num_classes,
                         dropout=dropout,
                         name=name if name is not None else f"SybilFinderGCN-E{num_epochs}")

        self.uses_honest_nodes = True
        self.uses_sybil_nodes = True

    def _model_initialization(self):
        if self.num_layers is None:
            self.num_layers = int(math.ceil(math.log2(self.pyg_graph.num_nodes)))
            if self.verbose:
                print(f"Number of layers automatically set to log_2(n) = {self.num_layers} (n = #nodes)")

        self.model = GCN(input_width=self.input_width, num_layers=self.num_layers, hidden_width=self.hidden_width,
                         num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        if self.verbose:
            print(f"Model = {self.model}")

    def _criterion_initialization(self):
        if self.num_classes == 1:
            self.criterion = torch.nn.BCELoss()
        elif self.num_classes >= 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception("Width not supported")

    def _optimizer_initialization(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)


class GCN(torch.nn.Module):
    def __init__(self, input_width: int, num_layers: int, hidden_width: int, num_classes: int, dropout: bool = True):
        # TODO: set normalize = False?, initialize weights to 1.0? set aggr = "mean"/"sum"/"mul"?
        super().__init__()
        # torch.manual_seed(SEED)
        self.num_classes = num_classes
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:  # First layer
                self.convs.append(GCNConv(input_width, hidden_width, bias=False))
            elif i == num_layers - 1:  # Last layer
                self.convs.append(GCNConv(hidden_width, num_classes, bias=False))
            else:  # Middle layers
                self.convs.append(GCNConv(hidden_width, hidden_width, bias=False))

    def forward(self, x, edge_index):
        h = x
        i = 0
        for conv in self.convs:
            if self.dropout:
                h = F.dropout(h, p=0.5, training=self.training)
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.tanh(h)
            i += 1

        if self.num_classes == 1:
            return F.sigmoid(h), h
        else:
            return F.softmax(h, dim=1), h


class SybilFinderRGCN(SybilGNN):
    def __init__(self, graph: Graph = None,
                 honest_nodes: [int] = None,
                 sybil_nodes: [int] = None,
                 threshold: float = 0.5,
                 pretrained_algorithm=None,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 train_model: bool = True,
                 fine_tune: bool = False,
                 num_epochs: int = 200,  # 200 seems to be sweet spot
                 patience: int = None,
                 input_width: int = 1,
                 num_layers: int = None,
                 hidden_width: int = 2,
                 num_classes: int = 2,
                 dropout: bool = False,
                 case_mask: dict = None,
                 name: str = None,
                 name_suffix: str = "") -> None:
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         threshold=threshold,
                         pretrained_algorithm=pretrained_algorithm,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         train_model=train_model,
                         fine_tune=fine_tune,
                         num_epochs=num_epochs,
                         patience=patience,
                         input_width=input_width,
                         num_layers=num_layers,
                         hidden_width=hidden_width,
                         num_classes=num_classes,
                         dropout=dropout,
                         name=name if name is not None else f"SybilFinderRGCN-E{num_epochs}{name_suffix}")

        if case_mask is None:
            self.case_mask = {"H-H": True, "S-S": True, "H-U": False, "S-U": False, "H-S": False}  # Seems to work best
        else:
            self.case_mask = case_mask

    def _setup_pyg_graph(self):
        super()._setup_pyg_graph()
        self._setup_edge_type()

    def _setup_edge_type(self):
        if self.case_mask.keys() != {"H-H", "S-S", "H-U", "S-U", "H-S"}:
            raise Exception("Invalid case mask")

        edge_type = torch.zeros(self.pyg_graph.edge_index.shape[1], dtype=torch.long, device=self.device)

        if self.case_mask.values() == {False, False, False, False, False}:
            # No relations, no need to iterate below
            # print(edge_type)
            self.pyg_graph.edge_type = edge_type.to(self.device)
            return

        idx_mapping = [-1, -1, -1, -1, -1]
        case_counts = {"H-H": 0, "S-S": 0, "H-U": 0, "S-U": 0, "H-S": 0, "U-U": 0}
        effective_case_counts = {"H-H": 0, "S-S": 0, "H-U": 0, "S-U": 0, "H-S": 0, "U-U": 0}
        current_idx = 1
        for i in range(self.pyg_graph.edge_index.shape[1]):
            u = self.pyg_graph.edge_index[0, i]
            v = self.pyg_graph.edge_index[1, i]
            if self.train_labels[u] == 0 and self.train_labels[v] == 0:  # Unknown-Unknown
                # This case it done first since it is by far the most common
                case_counts["U-U"] += 1
                effective_case_counts["U-U"] += 1
                continue
            elif self.train_labels[u] == -1 and self.train_labels[v] == -1:  # Honest-Honest
                case_counts["H-H"] += 1
                if self.case_mask["H-H"]:
                    if idx_mapping[0] == -1:
                        idx_mapping[0] = current_idx
                        current_idx += 1
                    edge_type[i] = idx_mapping[0]
                    effective_case_counts["H-H"] += 1
            elif self.train_labels[u] == 1 and self.train_labels[v] == 1:  # Sybil-Sybil
                case_counts["S-S"] += 1
                if self.case_mask["S-S"]:
                    if idx_mapping[1] == -1:
                        idx_mapping[1] = current_idx
                        current_idx += 1
                    edge_type[i] = idx_mapping[1]
                    effective_case_counts["S-S"] += 1

            elif self.train_labels[u] == -1 and self.train_labels[v] == 1 or self.train_labels[u] == 1 and \
                    self.train_labels[
                        v] == -1:  # Honest-Sybil or Sybil-Honest
                case_counts["H-S"] += 1
                if self.case_mask["H-S"]:
                    if idx_mapping[2] == -1:
                        idx_mapping[2] = current_idx
                        current_idx += 1
                    edge_type[i] = idx_mapping[2]
                    effective_case_counts["H-S"] += 1

            elif self.train_labels[u] == -1 or self.train_labels[v] == -1:  # Honest-Unknown or Unknown-Honest
                case_counts["H-U"] += 1
                if self.case_mask["H-U"]:
                    if idx_mapping[3] == -1:
                        idx_mapping[3] = current_idx
                        current_idx += 1
                    edge_type[i] = idx_mapping[3]
                    effective_case_counts["H-U"] += 1

            elif self.train_labels[u] == 1 or self.train_labels[v] == 1:  # Sybil-Unknown or Unknown-Sybil
                case_counts["S-U"] += 1
                if self.case_mask["S-U"]:
                    if idx_mapping[4] == -1:
                        idx_mapping[4] = current_idx
                        current_idx += 1
                    edge_type[i] = idx_mapping[4]
                    effective_case_counts["S-U"] += 1

        total_count = sum(case_counts.values())
        if self.verbose:
            print(f"case counts = {case_counts}")
            print(f"effective case counts = {effective_case_counts}")
            print(f"Total count = {total_count}")

        # print(edge_type)
        self.pyg_graph.edge_type = edge_type.to(self.device)

    def _model_initialization(self):
        if self.num_layers is None:
            self.num_layers = int(math.ceil(math.log2(self.pyg_graph.num_nodes)))
            if self.verbose:
                print(f"Number of layers automatically set to log_2(n) = {self.num_layers} (n = #nodes)")

        num_relations = torch.unique(self.pyg_graph.edge_type).shape[0]
        self.model = RGCN(input_width=self.input_width, num_layers=self.num_layers, hidden_width=self.hidden_width,
                          num_classes=self.num_classes, num_relations=num_relations, dropout=self.dropout).to(
            self.device)
        if self.verbose:
            print(f"Number of relations: {num_relations}")
            print(f"Model = {self.model}")

    def _criterion_initialization(self):
        if self.num_classes == 1:
            self.criterion = torch.nn.BCELoss()
        elif self.num_classes >= 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception("Width not supported")

    def _optimizer_initialization(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)  # , weight_decay=5e-4)


class RGCN(torch.nn.Module):
    def __init__(self, input_width: int, num_layers: int, hidden_width: int, num_classes: int, num_relations: int,
                 dropout: bool = True):
        # TODO: set normalize = False?, initialize weights to 1.0? set aggr = "mean"/"sum"?
        super().__init__()
        # torch.manual_seed(SEED)
        self.dropout = dropout
        self.num_classes = num_classes
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:  # First layer
                self.convs.append(
                    RGCNConv(input_width, hidden_width, num_relations=num_relations, root_weight=False,
                             bias=False))
            elif i == num_layers - 1:  # Last layer
                self.convs.append(
                    RGCNConv(hidden_width, num_classes, num_relations=num_relations, root_weight=False,
                             bias=False))
            else:  # Middle layers
                self.convs.append(
                    RGCNConv(hidden_width, hidden_width, num_relations=num_relations, root_weight=False,
                             bias=False))

    def forward(self, x, edge_index, edge_type):
        h = x
        i = 0
        for conv in self.convs:
            if self.dropout:
                h = F.dropout(h, p=0.5, training=self.training)
            h = conv(h, edge_index, edge_type)
            if i < len(self.convs) - 1:
                h = F.tanh(h)
            i += 1

        if self.num_classes == 1:
            return F.sigmoid(h), h
        else:
            return F.softmax(h, dim=1), h


class SybilRankGNN(SybilGNN):
    def __init__(self, graph: Graph = None,
                 honest_nodes: [int] = None,
                 sybil_nodes: [int] = None,
                 num_layers: int = None,
                 total_trust: float = 100.0,
                 pivot: float = 0.2,
                 name: str = None):
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         name=name if name is not None else "SybilRankGNN")

        self.uses_directed_graph = False

        self.total_trust = total_trust
        self.num_layers = num_layers
        self.pivot = pivot

    def _find_sybils(self) -> list[int]:
        if self.num_layers is None:
            self.num_layers = int(math.ceil(math.log2(self.pyg_graph.num_nodes)))

        # Set initial values
        honest_trust = self.total_trust / self.num_honest_nodes
        x_values = torch.full(size=(self.pyg_graph.num_nodes, 1), fill_value=0.0, dtype=torch.float32)
        for i in range(self.graph.num_nodes()):
            if self.train_labels[i] == -1:
                x_values[i] = honest_trust
        self.pyg_graph.x = x_values.to(self.device)

        # Construct neighbor index
        neighbor_index = [len(self.graph.neighbors(node)) for node in self.graph.nodes_list()]

        # Construct model
        self.model = SybilRankModel(num_layers=self.num_layers, neighbor_index=neighbor_index, device=self.device).to(
            self.device)
        print(self.model)

        # Get trust values
        trust = self.model(self.pyg_graph.x.to(self.device), self.pyg_graph.edge_index.to(self.device))
        trust = trust.detach().cpu().numpy().flatten()

        # Reorder trust so that it represents the node order [0,...n-1], since this is not necessarily the case
        ordered_trust = np.zeros_like(trust)
        j = 0
        for i in self.graph.nodes_list():
            ordered_trust[i] = trust[j]
            j += 1

        # Normalize trust
        normalized_trust = SybilRank.degree_normalize_trust(ordered_trust, self.graph)

        # print(f"normalized = {normalized_trust}")
        # print(f"... sum = {normalized_trust.sum()}")

        self.trust_values = normalized_trust
        self.has_trust_values = True

        # Rank trust
        ranked_trust = SybilRank.rank_trust(normalized_trust)

        default_pivot_idx = int(self.pivot * len(ranked_trust))

        return [node for node, trust in ranked_trust[:default_pivot_idx]]


class SybilRankModel(torch.nn.Module):
    def __init__(self, num_layers: int, neighbor_index: [int], device):
        super().__init__()
        # torch.manual_seed(SEED)
        self.neighbor_index = neighbor_index  # Number of neighbors for each node
        self.device = device
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = GCNConv(in_channels=1, out_channels=1, bias=False, aggr="sum", normalize=False)
            # TODO pass device from Algorithm
            fixed_weight_param = torch.nn.Parameter(torch.tensor([[1.0]], device=self.device), requires_grad=False).to(
                self.device)  # Weight fixed to 1.0
            layer.lin.register_parameter("weight", fixed_weight_param)
            self.layers.append(layer)

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            for i in range(h.shape[0]):
                h[i] /= self.neighbor_index[i]  # SybilRank-specific
            h = layer(h, edge_index)

        return h


class SybilFinderGAT(SybilGNN):
    def __init__(self, graph: Graph = None,
                 honest_nodes: [int] = None,
                 sybil_nodes: [int] = None,
                 threshold: float = 0.5,
                 pretrained_algorithm=None,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 train_model: bool = True,
                 fine_tune: bool = False,
                 num_epochs: int = 10,
                 patience: int = None,
                 input_width: int = 1,
                 num_layers: int = 2,
                 hidden_width: int = 4,
                 num_classes: int = 2,
                 num_heads: int = 4,
                 dropout: bool = True,
                 name: str = None):
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         threshold=threshold,
                         pretrained_algorithm=pretrained_algorithm,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         train_model=train_model,
                         fine_tune=fine_tune,
                         num_epochs=num_epochs,
                         patience=patience,
                         input_width=input_width,
                         num_layers=num_layers,
                         hidden_width=hidden_width,
                         num_classes=num_classes,
                         dropout=dropout,
                         name=name if name is not None else f"SybilGAT-E{num_epochs}")
        self.num_heads = num_heads

    def _model_initialization(self):
        if self.num_layers is None:
            self.num_layers = int(math.ceil(math.log2(self.pyg_graph.num_nodes)))
            if self.verbose:
                print(f"Number of layers automatically set to log_2(n) = {self.num_layers} (n = #nodes)")

        self.model = GAT(input_width=self.input_width,
                         num_layers=self.num_layers,
                         hidden_width=self.hidden_width,
                         num_classes=self.num_classes,
                         num_heads=self.num_heads,
                         dropout=self.dropout).to(self.device)
        if self.verbose:
            print(f"Model = {self.model}")

    def _criterion_initialization(self):
        if self.num_classes == 1:
            self.criterion = torch.nn.BCELoss()
        elif self.num_classes >= 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception("Width not supported")

    def _optimizer_initialization(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)


class GAT(torch.nn.Module):
    def __init__(self, input_width, num_layers, hidden_width, num_classes, num_heads, dropout: bool = True):
        super().__init__()
        # torch.manual_seed(SEED)
        self.dropout = dropout
        self.num_classes = num_classes
        self.convs = torch.nn.ModuleList()

        # self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

        for i in range(num_layers):
            if i == 0:  # First layer
                self.convs.append(GATConv(input_width, hidden_width, heads=num_heads))
            elif i == num_layers - 1:  # Last layer
                self.convs.append(GATConv(hidden_width * num_heads, num_classes, heads=1))
            else:  # Middle layers
                self.convs.append(GATConv(hidden_width * num_heads, hidden_width, heads=num_heads))

    def forward(self, x, edge_index):

        # x = self.conv1(x, edge_index)
        # # x = F.elu(x)
        # x = F.tanh(x)
        # x = self.conv2(x, edge_index)
        # h = x

        h = x
        i = 0
        for conv in self.convs:
            if self.dropout:
                h = F.dropout(h, p=0.5, training=self.training)
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.tanh(h)
                # h = F.elu(h)
            i += 1

        if self.num_classes == 1:
            return F.sigmoid(h), h
        else:
            return F.softmax(h, dim=1), h
