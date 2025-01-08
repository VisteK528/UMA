from collections import Counter
import numpy as np
from typing import Tuple

def entropy_func(class_count: int, num_samples: int) -> float:
    probability = class_count / num_samples
    entropy = - probability * np.log2(probability)
    return entropy


def split(data: np.ndarray, classes: np.ndarray, split_feature: int, split_val: float) -> Tuple[np.ndarray, np.ndarray]:
    dataset = np.c_[data, classes]
    feature_column = dataset[:, split_feature].astype(float)
    mask = feature_column >= split_val

    child_a = dataset[mask]
    child_b = dataset[~mask]
    return child_a, child_b


class Group:
    def __init__(self, group_classes):
        self.group_classes = group_classes
        self.entropy = self.group_entropy()

    def __len__(self) -> int:
        return len(self.group_classes)

    def group_entropy(self) -> float:
        entropy = 0
        class_counts = Counter(self.group_classes)
        num_samples = len(self)
        for group_class_count in class_counts.values():
            entropy += entropy_func(group_class_count, num_samples)
        return entropy


class Node:
    def __init__(self, split_feature=None, split_val=None, depth=None, child_node_a=None, child_node_b=None, val=None):
        self.split_feature = split_feature
        self.split_val = split_val
        self.depth = depth
        self.child_node_a = child_node_a
        self.child_node_b = child_node_b
        self.val = val

    def predict(self, data) -> int:
        if self.val is not None:
            return self.val
        elif data[self.split_feature] >= self.split_val:
            return self.child_node_a.predict(data)
        else:
            return self.child_node_b.predict(data)


class DecisionTreeClassifier(object):
    def __init__(self, max_depth, random_forest_version=False):
        self.depth = 0
        self.max_depth = max_depth
        self.tree = None
        self._random_forest_version = random_forest_version

    @staticmethod
    def get_split_entropy(group_a: Group, group_b: Group) -> float:
        split_entropy = 0
        parent_group_count = len(group_a) + len(group_b)
        child_groups = [group_a, group_b]
        for group in child_groups:
            split_entropy += (len(group) / parent_group_count) * group.group_entropy()
        return split_entropy

    def get_information_gain(self, parent_group: Group, child_group_a: Group, child_group_b: Group) -> float:
        information_gain = parent_group.group_entropy() - self.get_split_entropy(child_group_a, child_group_b)
        return information_gain

    def get_best_feature_split(self, feature_values: np.ndarray, classes: np.ndarray) -> Tuple[float, float]:
        parent = Group(classes)
        possible_thresholds = np.unique(feature_values)
        best_split_val = None
        best_gain = 0

        for threshold in possible_thresholds:
            child_a, child_b = split(feature_values, classes, 0, threshold)
            if child_a.shape[0] == 0 or child_b.shape[0] == 0:
                continue
            child_a = Group(child_a[:, -1])
            child_b = Group(child_b[:, -1])
            gain = self.get_information_gain(parent, child_a, child_b)

            if gain >= best_gain:
                best_gain = gain
                best_split_val = threshold
        return best_split_val, best_gain

    def get_best_split(self, data: np.ndarray, classes: np.ndarray) -> Tuple[int, float, float]:
        best_argument = None
        best_split = None
        best_gain = 0

        num_arguments = data.shape[1]
        chosen_arguments = [x for x in range(num_arguments)]

        if self._random_forest_version:
            chosen_arguments = np.random.choice(chosen_arguments, size=int(np.floor(np.sqrt(num_arguments))),
                                                replace=False)
        for argument in chosen_arguments:

            split_val, split_gain = self.get_best_feature_split(data[:, argument], classes)
            if split_val is None:
                continue

            child_a, child_b = split(data, classes, argument, split_val)
            child_a = Group(child_a[:, -1])
            child_b = Group(child_b[:, -1])
            gain = self.get_information_gain(Group(classes), child_a, child_b)

            if gain >= best_gain:
                best_gain = gain
                best_argument = argument
                best_split = split_val

        return best_argument, best_split, best_gain

    def build_tree(self, data: np.ndarray, classes: np.ndarray, depth=0) -> 'Node':
        if depth == self.max_depth or len(set(classes)) == 1:
            return Node(val=Counter(classes).most_common(1)[0][0])

        best_argument, best_split, best_gain = self.get_best_split(data, classes)

        if best_argument is None:
            return Node(val=Counter(classes).most_common(1)[0][0])

        child_a_data, child_b_data = split(data, classes, best_argument, best_split)
        child_a_classes = child_a_data[:, -1]
        child_b_classes = child_b_data[:, -1]

        child_a_node = self.build_tree(child_a_data[:, :-1], child_a_classes, depth + 1)
        child_b_node = self.build_tree(child_b_data[:, :-1], child_b_classes, depth + 1)

        return Node(split_feature=best_argument, split_val=best_split, depth=depth, child_node_a=child_a_node,
                    child_node_b=child_b_node)

    def fit(self, data: np.ndarray, classes: np.ndarray) -> None:
        self.tree = self.build_tree(data, classes)

    def predict(self, data: np.ndarray) -> int:
        if self.tree is not None:
            return self.tree.predict(data)

    def evaluate(self, data: np.ndarray, classes: np.ndarray, verbose=0) -> float:
        if self.tree is not None:
            samples = len(classes)
            positively_predicted = 0
            for i, subdataset in enumerate(zip(data, classes), 1):
                sample, actual_class = subdataset
                predicted_class = self.tree.predict(sample)
                if predicted_class == actual_class:
                    positively_predicted += 1
                if verbose == 1:
                    print(f"Predicting {i:>2}/{samples}\tPrediction: {predicted_class}\tActual class: {actual_class}")
            accuracy = positively_predicted / samples
            if verbose == 1:
                print()
            print(f"Accuracy after predicting {samples} samples: {accuracy * 100:.2f}%")
            return accuracy