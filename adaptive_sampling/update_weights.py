import numpy as np
import torch


def update_utility_function(utility_function_mode, y, probs, num_ways, num_shots):
    """Compute utility function values for the prediction

    Args:
        utility_function_mode (int): Utility function mode, refer to the README.
        y (Tensor): A data targets batch, with shape [bs, n_classes].
        probs (Tensor): The predicted probability of the batch, with shape [bs, n_classes].
        num_ways (int): Number of ways.
        num_shots (int): Number of shots.

    Returns:
        utility_func_per_class (Tensor): Utility function values for the prediction, with shape [num_ways] or
            [num_ways, num_ways].
    """

    with torch.no_grad():

        # Based on classification probability of target class
        if utility_function_mode in [1]:
            probs_of_target = probs[torch.arange(probs.shape[0]), y]
            utility_func_per_sample = 1 - probs_of_target
            utility_func_per_class = utility_func_per_sample.reshape([num_ways, -1]).mean(dim=1)  # Shape: [num_ways]

        # Based on classification probability of all classes
        elif utility_function_mode in [4]:
            probs[torch.arange(probs.shape[0]), y] = 1 - probs[torch.arange(probs.shape[0]), y]
            utility_func_per_sample = probs
            utility_func_per_class = utility_func_per_sample.mean(dim=0)  # Shape: [num_ways]

        # For class correlation matrix
        elif utility_function_mode in [101, 102, 103]:
            probs[torch.arange(probs.shape[0]), y] = 1 - probs[torch.arange(probs.shape[0]), y]

            if utility_function_mode [101]:  # 101: Hard
                pass
            elif utility_function_mode [102]:  # 102: Easy
                probs = 1 - probs
            elif utility_function_mode [103]:  # 103: Uncertain
                probs = probs * (1 - probs)

            utility_func_per_sample = probs
            utility_func_per_class = torch.stack([torch.mean(
                utility_func_per_sample[num_shots * i: num_shots * (i + 1)], dim=0)
                for i in range(num_ways)]
            )  # Shape: [num_ways, num_ways]

        else:
            raise NotImplementedError

    return utility_func_per_class


def update_weights(adaptive_sampling_mode, tau, alpha, num_ways, selected_classes, utility_func_per_class_list,
        min_rate_for_adaptive_sampling_2, max_times_for_adaptive_sampling_3, global_vars):
    """Update weights for adaptive sampling

    Args:
        adaptive_sampling_mode (int): Adaptive sampling mode, refer to the README.
        tau (float): Hyper-parameter for adaptive sampling.
        alpha (float): Hyper-parameter for adaptive sampling.
        num_ways (int): Number of ways.
        selected_classes (list[str]): Names of the selected classes.
        utility_func_per_class_list (list[Tensor]): List of unitily function values for each class, each element is
            returned by the update_utility_function() method.
        update_utility_function (float): Minimum value of minimum probability, used for adaptive_sampling_mode == 2.
        max_times_for_adaptive_sampling_3 (float): Maximum ratio of maximum probability to minimum probability, used for
            adaptive_sampling_mode == 3.
    """

    with torch.no_grad():

        # No adaptive sampling
        if adaptive_sampling_mode == 0:
            pass

        # Without class correlation
        elif adaptive_sampling_mode in [4, 5, 6, 8]:

            num_train_classes = global_vars.NUM_TRAIN_CLASSES
            train_classes_weights = global_vars.TRAIN_CLASSES_WEIGHTS.copy()

            utility_func_per_class = torch.mean(torch.stack(utility_func_per_class_list), dim=0)

            for i in range(len(selected_classes)):
                c_name = selected_classes[i]
                c_utility_func = utility_func_per_class[i].cpu().detach().numpy()

                # Exponentially update the importance weights
                train_classes_weights[c_name] = \
                    np.power(train_classes_weights[c_name], tau) * np.exp(alpha * c_utility_func)

            # Naively normalize the importance weights
            if adaptive_sampling_mode in [4]:
                sum_weights = np.sum(list(train_classes_weights.values()))
                train_classes_weights = {c: w / sum_weights * num_train_classes
                                         for c, w in train_classes_weights.items()}

            # Normalize the importance weights and make each category have a minimum sampling rate
            elif adaptive_sampling_mode in [5]:
                min_rate = min_rate_for_adaptive_sampling_2
                sum_weights = np.sum(list(train_classes_weights.values()))
                train_classes_weights = {
                    c: (w / sum_weights * (1 - min_rate) + 1 / num_train_classes * min_rate) * num_train_classes
                    for c, w in train_classes_weights.items()}

            # Normalize the importance weights and make the highest sampling rate not exceed X times than the minimum
            elif adaptive_sampling_mode in [6]:
                max_times = max_times_for_adaptive_sampling_3
                max_rate = np.max(list(train_classes_weights.values()))
                min_rate = np.min(list(train_classes_weights.values()))
                if max_rate / min_rate > max_times:
                    addend = (max_rate - min_rate * max_times) / (max_times - 1)
                    train_classes_weights = {c: w + addend for c, w in train_classes_weights.items()}
                sum_weights = np.sum(list(train_classes_weights.values()))
                train_classes_weights = {c: w / sum_weights * num_train_classes
                                            for c, w in train_classes_weights.items()}

            # No need to change class weights now, just pass
            elif adaptive_sampling_mode in [8]:
                pass

            else:
                raise NotImplementedError

            global_vars.TRAIN_CLASSES_WEIGHTS.update(train_classes_weights)

        # With class correlation
        elif adaptive_sampling_mode in [104, 105, 108]:

            num_train_classes = global_vars.NUM_TRAIN_CLASSES
            train_classes = global_vars.TRAIN_CLASSES.copy()
            train_classes_corr = global_vars.TRAIN_CLASSES_CORR.copy()

            selected_classes = global_vars.SELECTED_CLASSES[task_id]
            utility_func_per_class = torch.mean(torch.stack(utility_func_per_class_list), dim=0)
            utility_func_per_class = utility_func_per_class.cpu().detach().numpy()

            # Update weights
            for i in range(num_ways):
                for j in range(num_ways):
                    i_name, j_name = selected_classes[i], selected_classes[j]
                    i_idx, j_idx = train_classes.index(i_name), train_classes.index(j_name)
                    if i < j:
                        utility_func_value = (utility_func_per_class[i, j] + utility_func_per_class[j, i]) / 2

                        if adaptive_sampling_mode in [104, 105, 108]:
                            train_classes_corr[i_idx][j_idx] = np.power(train_classes_corr[i_idx][j_idx], tau) \
                                                                * np.exp(alpha * utility_func_value)
                            train_classes_corr[j_idx][i_idx] = train_classes_corr[i_idx][j_idx]

            # Normalize
            if adaptive_sampling_mode [104]:
                train_classes_corr *= (num_train_classes ** 2 - num_train_classes) / np.sum(train_classes_corr)

            elif adaptive_sampling_mode [105]:
                min_rate = min_rate_for_adaptive_sampling_2
                train_classes_corr *= (1 - min_rate) / np.sum(train_classes_corr)
                train_classes_corr += min_rate * (np.ones_like(train_classes_corr) - np.eye(num_train_classes))\
                                        / (num_train_classes ** 2 - num_train_classes)
                train_classes_corr *= (num_train_classes ** 2 - num_train_classes) / np.sum(train_classes_corr)

            elif adaptive_sampling_mode [108]:  # No norm
                pass

            else:
                raise NotImplementedError

            # Update global variables
            global_vars.TRAIN_CLASSES_CORR[:] = train_classes_corr

        else:
            raise NotImplementedError
