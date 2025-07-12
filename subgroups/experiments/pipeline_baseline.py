from ..utils.scoring import compute_margins
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from subgroups.utils.random import fork_rng
from subgroups.datastorage.experiment import Experiment
import os
import chz
import pandas as pd
from sklearn.utils import shuffle


def fit_single_classifier(split_rng, build_model_rngs, feature_label_dict, model_factory, train_size_per_class, test_size_per_class):

    rng = fork_rng(split_rng, 3)
    indices_class0 = rng[0].choice(len(feature_label_dict['labels_class0']), size=train_size_per_class+test_size_per_class, replace=False)
    indices_class0_train = indices_class0[:train_size_per_class]
    indices_class0_test = indices_class0[train_size_per_class:]
    indices_class1 = rng[1].choice(len(feature_label_dict['labels_class1']), size=train_size_per_class+test_size_per_class, replace=False)
    indices_class1_train = indices_class1[:train_size_per_class]
    indices_class1_test = indices_class1[train_size_per_class:]

    X_train = np.concatenate((feature_label_dict['features_class0'][indices_class0_train], feature_label_dict['features_class1'][indices_class1_train]))
    y_train = np.concatenate((feature_label_dict['labels_class0'][indices_class0_train], feature_label_dict['labels_class1'][indices_class1_train]))
    X_test = np.concatenate((feature_label_dict['features_class0'][indices_class0_test], feature_label_dict['features_class1'][indices_class1_test]))
    y_test = np.concatenate((feature_label_dict['labels_class0'][indices_class0_test], feature_label_dict['labels_class1'][indices_class1_test]))

    random_state = np.random.RandomState(rng[2].bit_generator)
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=random_state)

    model = model_factory.build_model(rng=np.random.default_rng(build_model_rngs.bit_generator))
    model.fit(X_train_shuffled, y_train_shuffled)
    pred_test = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, pred_test[:,1])
    margins = compute_margins(pred_test[:,1], y_test)
    return auc, np.mean(margins)

def run_baseline(n_models,experiment,batch_starter_seed,train_size=0.7,test_size=0.3):
    pc_indices = experiment.feature_selector.feature_indices(n_pcs=experiment.npcs)
    random_generator = experiment.tc_random_generator(batch_starter_seed=batch_starter_seed)
    nsamples = experiment.dataset.num_samples

    build_model_rngs_children = fork_rng(random_generator.model_build_rng, n_models)
    shuffle_rng = fork_rng(random_generator.train_data_shuffle_rng, n_models)

    if isinstance(train_size, int) and isinstance(test_size, int):
        train_size_per_class = int(train_size//2)
        test_size_per_class = int(test_size//2)
    elif (isinstance(train_size, float)) and (isinstance(test_size, float)):
        if train_size+test_size > 1:
            raise ValueError("train_size and test_size must be less than 1")
        train_size_per_class = int(train_size*nsamples//2)
        test_size_per_class = int(test_size*nsamples//2)
    else:
        raise ValueError("train_size and test_size must be either int or float")

    all_aucs = []
    all_margins = []

    features = experiment.dataset.features[:,pc_indices]
    labels = experiment.dataset.coarse_labels

    dict_features_labels = {
        'features_class0': features[labels == 0],
        'features_class1': features[labels == 1],
        'labels_class0': labels[labels == 0],
        'labels_class1': labels[labels == 1]
    }

    for i in tqdm(range(n_models)):
        auc, margins = fit_single_classifier(shuffle_rng[i],
                                             build_model_rngs_children[i],
                                            dict_features_labels,
                                            experiment.model_factory,
                                            train_size_per_class=train_size_per_class,
                                            test_size_per_class=test_size_per_class)

        all_aucs.append(auc)
        all_margins.append(margins)
    return pd.DataFrame({'aucs': all_aucs, 'mean_margins': all_margins})


def pipeline_baseline(experiment: Experiment, batch_starter_seed: int, n_models: int, train_size: int, test_size: int, in_memory: bool):

    results = run_baseline(n_models, experiment, batch_starter_seed, train_size, test_size)

    
    if in_memory:
        return results
    else:
        path = os.path.join(experiment.path, experiment.experiment_name, "clustering_outputs", f'baseline_results_batch_{batch_starter_seed}_train_{train_size}_test_{test_size}.csv')
        results.to_csv(path, index=False)
        print(f"Baseline results saved to {path}")
    
    

if __name__ == "__main__":
    chz.entrypoint(pipeline_baseline)



