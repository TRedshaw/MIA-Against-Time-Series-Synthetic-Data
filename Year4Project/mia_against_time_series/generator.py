import numpy as np
import pandas as pd
# from ctgan import TVAE  # IDK WHAT THE FDIFFERENCE IS
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
from sdv.single_table import TVAESynthesizer as TVAE


def par_time_synthesiser(mem_set, mem_set_size, synthetic_size, signal_length, training_epochs):
    """Specific PAR Setup - training set configuration"""
    mem_set_vertical = mem_set.flatten().T
    ecg_id = np.repeat(np.arange(mem_set_size) + 1, signal_length)
    ecg_sequence = np.tile(np.arange(signal_length) + 1, mem_set_size)

    # TESTING PRINTS
    # print(mem_set_vertical.shape)
    # print(ecg_id.shape)
    # print(ecg_sequence.shape)
    # quit()

    # Setup correct table structure for PARsynthesiser
    mem_set_df = pd.DataFrame()
    metadata = SingleTableMetadata()
    mem_set_df.insert(0, "ecg_id", ecg_id)
    mem_set_df.insert(1, "ecg_seq", ecg_sequence)
    mem_set_df.insert(2, "ecg_val", mem_set_vertical)
    mem_set_df.columns = mem_set_df.columns.astype(str)

    # Get metadata
    metadata.detect_from_dataframe(mem_set_df)
    metadata.update_column(column_name='ecg_id', sdtype='id')
    metadata.set_sequence_key(column_name='ecg_id')
    metadata.set_sequence_index(column_name='ecg_seq')

    '''Generating synthetic data'''
    synthesiser = PARSynthesizer(
        metadata=metadata,
        epochs=training_epochs,
        verbose=True,
        )
    synthesiser.fit(mem_set_df)  # Training the synthesizer with the dataset
    synth_set = synthesiser.sample(num_sequences=synthetic_size)  # Generating the synthetic dataset

    '''Reshaping synthetic_data dataframes for KDE'''
    synth_set = np.reshape(synth_set.to_numpy()[:, 2], (synthetic_size, signal_length))

    return synth_set


def tvae_time_synthesiser(mem_set, synthetic_size, training_epochs, metadata):
    mem_set_df = pd.DataFrame(mem_set)  # Convert mem_set to dataframe for generations
    mem_set_df.columns = mem_set_df.columns.astype(str)

    synthesiser = TVAE(metadata=metadata, epochs=training_epochs)
    synthesiser.fit(mem_set_df)  # incl. number of epochs and metadata

    synth_set = synthesiser.sample(synthetic_size)
    synth_set = synth_set.to_numpy()

    return synth_set
