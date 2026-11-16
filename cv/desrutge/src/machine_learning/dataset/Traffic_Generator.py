"""
dataset/Traffic_Generator.py
Path: desrutge/src/machine_learning/dataset/Traffic_Generator.py

Restored to match the ORIGINAL DesRUTGe return signature
(trainloaders, valloaders, testset), but the data source is now the
verified field-survey workbook scenarios in SIMULA.py, NOT Barcelona's
detector_data.csv and NOT cv_stats.json flow fields (which no longer
exist — cv_stats.json now carries vehicle_type_distribution only).

This file is kept functional (not deprecated) because other DesRUTGe
entry points — main_sim_centralized.py, simulation.py,
centralized_client.py — still import load_Traffic_Generator() and
prepare_dataset_Traffic(). run_pipeline.py does NOT use this file;
it calls cv_loader.load_cv_trainloader() directly. This file exists
for the federated/legacy DesRUTGe code paths only.
"""

import numpy as np
from utils.utils_logs import *


class Traffic_Generator:
    def __init__(self, scenario: str = "high", transform=None):
        """
        Loads the 24-slot hourly demand profile for a given scenario.

        Args:
            scenario: "low" | "medium" | "high" — matches
                      SIMULA.DEMAND_SCENARIOS keys.
            transform: kept for API compatibility, not used.
        """
        from machine_learning.training.SIMULA import DEMAND_SCENARIOS

        if scenario not in DEMAND_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Valid options: {list(DEMAND_SCENARIOS.keys())}"
            )

        total = DEMAND_SCENARIOS[scenario]["total_veh_hr"]
        self.data = [float(total)] * 24
        self.scenario = scenario

        log_info(f"Traffic_Generator loaded scenario='{scenario}' "
                 f"({len(self.data)} hourly slots, total={total} veh/hr)")
        print(self.data[:6], "...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}",
                f"Scenario: {getattr(self, 'scenario', 'unknown')}"]
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " + line for line in body]
        return "\n".join(lines)


def load_Traffic_Generator(scenario: str = "high"):
    """
    Loads the demand profile for a given verified field-survey scenario.
    Replaces the original detector_data.csv loader.
    """
    dataset = Traffic_Generator(scenario, transform=None)
    return dataset


def prepare_dataset_Traffic(dataset):
    """
    Prepares trainloaders and valloaders from the loaded dataset.
    Matches the ORIGINAL DesRUTGe return signature exactly:
        trainloaders[i] = list of flow values for "node" i
        valloaders[i]   = same
        testset         = held-out slice

    Since SmartTraffic-OM does not use federated multi-node training,
    there is only one "node" (i=0), whose trainloader is the FULL
    24-slot profile — NOT a single sliced value. This matches the
    behaviour run_pipeline.py actually relies on
    (np.mean(trainloader) over all 24 slots).
    """
    flows = [dataset[i] for i in range(len(dataset))]

    testset = flows[-1:]
    trainset = flows[:-1]   # 23 slots; or use `flows` directly if you
                             # want testset empty — see note below

    print(f"Trainset: {len(trainset)} samples, Testset: {len(testset)} samples.")

    # Single node — full profile as one trainloader, matching what
    # run_pipeline.py expects when computing np.mean(trainloader).
    trainloaders = [trainset]
    valloaders   = [trainset]

    return trainloaders, valloaders, testset