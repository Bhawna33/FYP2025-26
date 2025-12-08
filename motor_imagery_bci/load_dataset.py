import mne
import numpy as np
import os

def load_bci_iv_2b(data_dir, subjects=[1]):

    X = []
    y = []

    for sub in subjects:
        fname = os.path.join(data_dir, f"B0{sub}T.gdf")  
        raw = mne.io.read_raw_gdf(fname, preload=True)

        
        raw.pick_types(eeg=True)

        
        events, _ = mne.events_from_annotations(raw)

        
        event_id = {"left": 1, "right": 2}

      
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=2.0,
            tmax=6.0,
            baseline=None,
            preload=True
        )

        labels = epochs.events[:, -1]  
        data = epochs.get_data()       

        X.append(data)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

   
    y = (y - 1).astype(int)

    print("Loaded dataset:")
    print("X:", X.shape)
    print("y:", y.shape)

    return X, y
