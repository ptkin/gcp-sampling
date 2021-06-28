import numpy as np


def init(dataset, num_ways):
    """Initialize some global settings

    Args:
        dataset (str): Name of the dataset, including "mini_imagenet_full_size", "cifar100_fs", etc.
        num_ways (int): Number of ways.
    """

    global SELECTED_CLASSES
    SELECTED_CLASSES = [''] * num_way

    # Mini-imageNet
    global TRAIN_CLASSES
    if dataset == 'mini_imagenet_full_size':
        TRAIN_CLASSES = [
            'n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081', 'n01843383', 'n01910747', 'n02074367',
            'n02089867', 'n02091831', 'n02101006', 'n02105505', 'n02108089', 'n02108551', 'n02108915', 'n02111277',
            'n02113712', 'n02120079', 'n02165456', 'n02457408', 'n02606052', 'n02687172', 'n02747177', 'n02795169',
            'n02823428', 'n02966193', 'n03017168', 'n03047690', 'n03062245', 'n03207743', 'n03220513', 'n03337140',
            'n03347037', 'n03400231', 'n03476684', 'n03527444', 'n03676483', 'n03838899', 'n03854065', 'n03888605',
            'n03908618', 'n03924679', 'n03998194', 'n04067472', 'n04243546', 'n04251144', 'n04258138', 'n04275548',
            'n04296562', 'n04389033', 'n04435653', 'n04443257', 'n04509417', 'n04515003', 'n04596742', 'n04604644',
            'n04612504', 'n06794110', 'n07584110', 'n07697537', 'n07747607', 'n09246464', 'n13054560', 'n13133613']

    # CIFAR-FS
    elif dataset == 'cifar100_fs':
        TRAIN_CLASSES = [
            'apple', 'aquarium_fish', 'bear', 'bee', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'can', 'castle',
            'caterpillar', 'chair', 'clock', 'cloud', 'cockroach', 'couch', 'cup', 'dinosaur', 'dolphin', 'elephant',
            'forest', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lawn_mower', 'lion', 'lizard', 'lobster',
            'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'palm_tree', 'pear', 'pine_tree', 'plate',
            'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'seal', 'shrew', 'skunk', 'skyscraper', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'tank', 'tiger', 'train', 'trout', 'tulip', 'turtle',
            'willow_tree', 'wolf'
        ]

    else:
        raise NotImplementedError

    # Number of training classes
    global NUM_TRAIN_CLASSES
    NUM_TRAIN_CLASSES = len(TRAIN_CLASSES)

    # Class weights dictionary
    global TRAIN_CLASSES_WEIGHTS
    TRAIN_CLASSES_WEIGHTS = {c: 1.0 for c in TRAIN_CLASSES}

    # Class correlation weights matrix
    global TRAIN_CLASSES_CORR
    TRAIN_CLASSES_CORR = np.ones([NUM_TRAIN_CLASSES, NUM_TRAIN_CLASSES]) - np.eye(NUM_TRAIN_CLASSES)

    # Map index to class-pair
    global IDX_TO_CLS_CORR
    IDX_TO_CLS_CORR = [[i, j]
                       for i in range(NUM_TRAIN_CLASSES)
                       for j in range(NUM_TRAIN_CLASSES)]
