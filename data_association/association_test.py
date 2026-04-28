import numpy as np
import data_association

if __name__ == "__main__":
    # 1. Initialize the associator
    associator = data_association.GNNDataAssociator(gate_threshold=10.0)

    # 2. Create Dummy Data
    # Track 0 is at (0,0), Track 1 is at (10,10)
    dummy_tracks = [{'id': 1, 'x': 0, 'y': 0}, {'id': 2, 'x': 10, 'y': 10}]

    # Meas 0 is close to Track 0. Meas 1 is close to Track 1. Meas 2 is clutter far away.
    dummy_meas = [{'x': 0.5, 'y': 0.5}, {'x': 10.2, 'y': 9.8}, {'x': 50, 'y': 50}]

    # 3. Run the association
    matches, un_tracks, un_meas = associator.associate(dummy_tracks, dummy_meas)

    # 4. Print Results
    print(f"Matches (Track Index, Meas Index): {matches}")
    print(f"Unmatched Tracks: {un_tracks}")
    print(f"Unmatched Measurements: {un_meas}")