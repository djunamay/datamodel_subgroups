from ..datasets.test_data import RandomDataset
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..datasamplers.random_generators import RandomGeneratorSNR, RandomGeneratorTC
from ..models.classifier import XgbFactory, XgbFactoryInitializer
from ..datasamplers.mask_generators import fixed_alpha_mask_factory_initializer
from ..experiments.stopping_condition import SNRPrecisionStopping
from ..experiments.pipeline_tc import pipeline_tc
from ..experiments.pipeline_dm import pipeline_dm
from ..datastorage.experiment import Experiment
from ..datamodels.datamodels_pipeline import DatamodelsPipelineBasic
from ..datamodels.regressor import LassoFactory
from ..datamodels.indices import SequentialIndices
import shutil
import numpy as np
import os

def test_pipeline_dm():

    random_dataset = RandomDataset()

    exp = Experiment(dataset=random_dataset, 
            mask_factory=fixed_alpha_mask_factory(alpha=0.1), 
            model_factory=XgbFactory(), 
            model_factory_initializer=XgbFactoryInitializer(), 
            mask_factory_initializer=fixed_alpha_mask_factory_initializer(),
            in_memory=False, 
            snr_n_models=5, 
            snr_n_passes=3,
            snr_random_generator=RandomGeneratorSNR,
            tc_random_generator=RandomGeneratorTC,
            stopping_condition=SNRPrecisionStopping(tolerance=0.05),
            path = './temp',
            experiment_name = 'test_experiment',
            
            datamodels_pipeline = DatamodelsPipelineBasic(path_to_inputs = './temp/test_experiment/classifier_outputs',
                                                        datamodel_factory = LassoFactory(n_lambdas=10, cv_splits=5),
                                                        path_to_outputs = './temp/test_experiment/datamodel_outputs'),
            indices_to_fit = SequentialIndices(batch_size=5),
            dm_n_train = 90,
            dm_n_test = 10)

    # create temporary training data
    batch_size = 50
    pipeline_tc(exp, batch_size=batch_size, batch_starter_seed=1)
    pipeline_tc(exp, batch_size=batch_size, batch_starter_seed=2)

    try:
        # check that both runs are recognized and concatenated in the datamodels pipeline
        assert exp.datamodels_pipeline._masks.shape[0]==2*batch_size

        # check that correct number of training samples are returned 
        n_train = 90
        n_test = 10

        x_train, y_train = exp.datamodels_pipeline._train_samples(exp.datamodels_pipeline._masks, exp.datamodels_pipeline._margins, n_train)
        
        assert x_train.shape[0]==n_train
        assert y_train.shape[0]==n_train

        # check that correct number of test samples are returned 
        x_test, y_test = exp.datamodels_pipeline._test_samples(exp.datamodels_pipeline._masks, exp.datamodels_pipeline._margins, n_train, n_test)
        assert x_test.shape[0]==n_test
        assert y_test.shape[0]==n_test

        # check that correct number of test samples are returned when n_test is None
        x_test, y_test = exp.datamodels_pipeline._test_samples(exp.datamodels_pipeline._masks, exp.datamodels_pipeline._margins, n_train, n_test=None)
        assert x_test.shape[0]==exp.datamodels_pipeline._masks.shape[0]-n_train
        assert y_test.shape[0]==exp.datamodels_pipeline._margins.shape[0]-n_train

        # check that all masks are unique and there is no train-test leakage
        assert np.unique(np.vstack([x_train, x_test]), axis=0).shape[0]==np.vstack([x_train, x_test]).shape[0]
        assert np.unique(np.vstack([y_train, y_test]), axis=0).shape[0]==np.vstack([y_train, y_test]).shape[0]

    finally:
        root = exp.path  # "./temp"
        if os.path.isdir(root):
            shutil.rmtree(root)