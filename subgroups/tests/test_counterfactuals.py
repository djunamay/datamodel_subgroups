from ..datasets.test_data import RandomDataset
from ..counterfactuals.datahandling import CoarseSplits
from ..counterfactuals.counterfactuals import CounterfactualEvaluation
from ..datasets.test_data import RandomDataset
from ..models.classifier import XgbFactory
import numpy as np
from sklearn.utils import shuffle

def test_coarse_splits():
    data = RandomDataset()

    for i in np.unique(data.fine_labels):
        fine_labels = data.fine_labels==int(i)
        splits = CoarseSplits(features=data.features, labels=data.coarse_labels, fine_label_bool=fine_labels)

        # fine labels are a subset of coarse labels
        assert len(np.unique(data.coarse_labels[fine_labels]))==1

        # check the dimensions of the splits
        vals, counts = np.unique(fine_labels[data.coarse_labels], return_counts=True)
        vals2, counts2 = np.unique(fine_labels[np.invert(data.coarse_labels)], return_counts=True)

        if len(counts2)==1:
            assert splits.whole.X.shape[0] == counts2[0]
            assert splits.split_a.X.shape[0] == counts[~vals]
            assert splits.split_b.X.shape[0] == counts[vals]
        elif len(counts2)==2:
            assert splits.whole.X.shape[0] == counts[0]
            assert splits.split_a.X.shape[0] == counts2[~vals2]
            assert splits.split_b.X.shape[0] == counts2[vals2]


def test_counterfactual_evaluation_partitions():

    dataset = RandomDataset()
    eval = CounterfactualEvaluation(features=dataset.features, 
                            coarse_labels=dataset.coarse_labels, 
                            train_size=50, test_size=50, 
                            classifier=XgbFactory(max_depth=7))

    import numpy as np
    labs = np.zeros_like(dataset.coarse_labels)
    labs[dataset.coarse_labels] = np.random.randint(0, 2, size=np.sum(dataset.coarse_labels))

    outputs = eval._prepare_data(cluster_labels=labs,
                    sample_indices=None,
                    shuffle_rng=np.random.default_rng(0))

    # test for no test-train leakage
    out = []
    for key in outputs.keys():
        x_test = np.sum(outputs[key]['X_test'], axis=1)
        x_train = np.sum(outputs[key]['X_train'], axis=1)
        x = np.concatenate([x_test, x_train])
        out.append(x)
        assert len(np.unique(x))==len(x)
    
    # test that partitions are true partitions
    assert len(np.unique(np.concatenate(out)))==len(np.concatenate(out))

def test_counterfactual_evaluation_predictions():

    # Make predictions with class
    dataset = RandomDataset()
    eval = CounterfactualEvaluation(features=dataset.features, 
                            coarse_labels=dataset.coarse_labels, 
                            train_size=50, test_size=50, 
                            classifier=XgbFactory(max_depth=7))

    labs = np.zeros_like(dataset.coarse_labels)
    labs[dataset.coarse_labels] = np.random.randint(0, 2, size=np.sum(dataset.coarse_labels))


    out_probs = eval._counterfactual_evaluation(cluster_labels=labs,
                    sample_indices=None,
                    shuffle_rng=np.random.default_rng(0),
                    model_rng=np.random.default_rng(0))
    
    # Make predictions "manually"
    def make_predictions(labs):
        rng = np.random.default_rng(0)
        data = eval._prepare_data(cluster_labels=labs,
                        sample_indices=None,
                        shuffle_rng=rng)

        model = eval.classifier.build_model(rng=np.random.default_rng(0))
        X_tr = np.concatenate([data['A']['X_train'], data['B']['X_train']])
        y_tr = np.concatenate([data['A']['y_train'], data['B']['y_train']])
        X_tr, y_tr = shuffle(X_tr, y_tr, random_state=np.random.RandomState(rng.bit_generator))
        model.fit(X_tr, y_tr)

        X_te = np.concatenate([data['A']['X_test'], data['B']['X_test']])
        y_te = np.concatenate([data['A']['y_test'], data['B']['y_test']])
        pred_1_1 = model.predict_proba(X_te)[:,1]

        X_te2 = np.concatenate([data['A']['X_test'], data['C']['X_test']])
        y_te2 = np.concatenate([data['A']['y_test'], data['C']['y_test']])
        pred_1_2 = model.predict_proba(X_te2)[:,1]
        return pred_1_1, pred_1_2, y_te, y_te2

    pred_1_1, pred_1_2, y_te, y_te2 = make_predictions(labs)

    assert np.array_equal(y_te, out_probs['y_test'])
    assert np.array_equal(y_te2, out_probs['y_test'])
    assert np.array_equal(y_te, y_te2)
    assert np.array_equal(pred_1_2, out_probs['train1_test2']['probs'])
    assert np.array_equal(pred_1_1, out_probs['train1_test1']['probs'])


def test_counterfactual_evaluation_results():
    from sklearn.metrics import roc_auc_score
    from ..utils.random import fork_rng


    # Make predictions with class
    dataset = RandomDataset()
    eval = CounterfactualEvaluation(features=dataset.features, 
                            coarse_labels=dataset.coarse_labels, 
                            train_size=50, test_size=50, 
                            classifier=XgbFactory(max_depth=7))

    labs = np.zeros_like(dataset.coarse_labels)
    labs[dataset.coarse_labels] = np.random.randint(0, 2, size=np.sum(dataset.coarse_labels))

    res = eval.counterfactual_evaluation(partition=labs, sample_indices=None, model_rng=np.random.default_rng(0), shuffle_rng=np.random.default_rng(0), n_iter=1)


    def make_predictions(labs, model_rng, shuffle_rng):
        build_model_rngs_children = fork_rng(model_rng, 1)
        train_data_shuffle_rngs_children = fork_rng(shuffle_rng, 1)
        data = eval._prepare_data(cluster_labels=labs,
                        sample_indices=None,
                        shuffle_rng=train_data_shuffle_rngs_children[0])

        model = eval.classifier.build_model(rng=build_model_rngs_children[0])
        X_tr = np.concatenate([data['A']['X_train'], data['B']['X_train']])
        y_tr = np.concatenate([data['A']['y_train'], data['B']['y_train']])
        X_tr, y_tr = shuffle(X_tr, y_tr, random_state=np.random.RandomState(train_data_shuffle_rngs_children[0].bit_generator))
        model.fit(X_tr, y_tr)

        X_te = np.concatenate([data['A']['X_test'], data['B']['X_test']])
        y_te = np.concatenate([data['A']['y_test'], data['B']['y_test']])
        pred_1_1 = model.predict_proba(X_te)[:,1]

        X_te2 = np.concatenate([data['A']['X_test'], data['C']['X_test']])
        y_te2 = np.concatenate([data['A']['y_test'], data['C']['y_test']])
        pred_1_2 = model.predict_proba(X_te2)[:,1]
        return pred_1_1, pred_1_2, y_te, y_te2

    pred_1_1, pred_1_2, y_te, y_te2 = make_predictions(labs, np.random.default_rng(0), np.random.default_rng(0))

    rocauc_1_1 = roc_auc_score(y_te, pred_1_1)
    rocauc_1_2 = roc_auc_score(y_te, pred_1_2   )

    temp = res[res['split']=='split_a']
    assert rocauc_1_1==float(temp[temp['prob_type']=='evaluation_on_split']['auc'])
    
    temp = res[res['split']=='split_b']
    assert rocauc_1_2==float(temp[temp['prob_type']=='evaluation_outside_split']['auc'])

# TODO: after this just write tests for the gtex data

