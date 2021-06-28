import numpy as np


def load_dataset_splits(dataset_name):
    """Load class names of the indicated dataset

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        dataset_splits (dict): Here is a demo structure:
            datasets = {
                'train': ['n01532829', 'n01558993', 'n01704323', ...],
                'val': [...],
                'test': [...],
            }
    """

    # Load your dataset here ...
    dataset_splits = None

    return dataset_splits

def adaptive_sampling(dataset_name, split_name, num_ways, adaptive_sampling_mode, seed, global_vars):
    """Adaptive sampling

    Args:
        dataset_name (str): Name of the dataset.
        split_name (str): Name of the data splitting, including "train", "val" and "test".
        num_ways (int): Number of ways.
        adaptive_sampling_mode (int): Adaptive sampling mode, refer to the README.
        seed (int): Random seed.

    Returns:
        selected_classes (list[str]): Names of the selected classes.
    """

    dataset_splits = load_dataset_splits()
    choice_a = dataset_splits[split_name]

    rng = np.random.RandomState(seed)

    # No adaptive sampling
    if adaptive_sampling_mode == 0:
        selected_classes = rng.choice(a=choice_a, size=num_ways, replace=False)

    # Adaptive sampling w/o class correlation
    elif 1 < adaptive_sampling_mode < 100:
        choice_p = np.array([global_vars.TRAIN_CLASSES_WEIGHTS[c] for c in choice_a])
        choice_p /= choice_p.sum()
        selected_classes = rng.choice(a=choice_a, size=num_ways, replace=False, p=choice_p)

    # Adaptive sampling w/ class correlation
    elif 100 < adaptive_sampling_mode < 200:
        selected_classes = []

        # Select the first two classes
        idx_to_cls = global_vars.IDX_TO_CLS_CORR
        first_2cls_idx = rng.choice(
            a=np.arange(global_vars.NUM_TRAIN_CLASSES ** 2),
            p=global_vars.TRAIN_CLASSES_CORR.flatten() / global_vars.TRAIN_CLASSES_CORR.sum())
        first_2cls = idx_to_cls[first_2cls_idx]
        selected_classes.extend(first_2cls)

        # Select the rest classes
        for _ in range(num_ways - 2):
            now_weights = None
            for i, j in enumerate(selected_classes):
                if now_weights is None:
                    now_weights = global_vars.TRAIN_CLASSES_CORR[j].copy()
                else:
                    now_weights *= global_vars.TRAIN_CLASSES_CORR[j]
            new_cls = rng.choice(a=np.arange(global_vars.NUM_TRAIN_CLASSES), p=now_weights / now_weights.sum())
            selected_classes.append(new_cls)

        selected_classes = [global_vars.TRAIN_CLASSES[i] for i in selected_classes]

    else:
        raise NotImplementedError

    return selected_classes
