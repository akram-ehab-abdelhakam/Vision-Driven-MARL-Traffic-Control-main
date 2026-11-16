import subprocess
import time
from .FARM import farm
import xml.etree.ElementTree as ET
import os
import random

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))


def find_det(det):
    Ruta = det + "/detectors.add.xml"
    file_path = []
    tree = ET.parse(Ruta)
    root = tree.getroot()
    for induction_loop in root.findall('inductionLoop'):
        file = induction_loop.get('lane')
        if file:
            file_path.append(file[0:-2])
    return file_path


def crear_OD(cant_coches, det):
    file_name = det + "/od_file.od"
    content = f"""$OR;D2
0.00 1.00
1.0
taz taz {cant_coches}"""
    with open(file_name, "w") as file:
        file.write(content)


def crear_trips(det):
    taz_file = det + "/miniTAZ.xml"
    od_file = det + "/od_file.od"
    output_file = det + "/trips.xml"
    command = ["od2trips", "-n", taz_file, "-d", od_file, "-o", output_file]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error estándar:", e.stderr)


def validar_trips(det):
    input_file  = det + "/trips.xml"
    output_file = det + "/trips_via.xml"
    tree = ET.parse(input_file)
    root = tree.getroot()
    for trip in root.findall('trip'):
        trip.set("departLane", "random")
    tree.write(output_file)


def crear_rutas(det):
    network_file = det + "/osm.net.xml"
    trips_file = det + "/trips_via.xml"
    output_routes = det + "/routes.rou.xml"
    additional_file = det + "/miniTAZ.xml"
    command = [
        "duarouter", "-n", network_file, "-t", trips_file,
        "-o", output_routes, "--routing-algorithm", "astar",
        "--additional-files", additional_file,
        "--ignore-errors", "--no-warnings"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error estándar duaroute:", e.stderr)


def llamar_SUMO(det):
    sumocfg_file = det + "/sim.sumocfg"
    command = ["sumo", "-c", sumocfg_file, "--no-warnings"]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar la simulación: {e}")
    except FileNotFoundError:
        print("SUMO no está instalado o no se puede encontrar el ejecutable.")


_ENTRY_EDGES = ['E7',  'E4',  'E6',  'E24', 'E16', 'E21']
_EXIT_EDGES  = ['E19', 'E3',  'E5',  'E12', 'E17', 'E20']

DEMAND_SCENARIOS = {
    "low":    {"entry_weights": [440, 290, 150, 440, 170, 290],
               "exit_weights":  [490, 330, 160, 280, 200, 330],
               "total_veh_hr":  1780},
    "medium": {"entry_weights": [680, 450, 230, 680, 270, 450],
               "exit_weights":  [760, 510, 250, 430, 300, 510],
               "total_veh_hr":  2765},
    "high":   {"entry_weights": [970, 650, 320, 970, 390, 650],
               "exit_weights":  [1090, 720, 360, 620, 430, 720],
               "total_veh_hr":  3950},
}

ACTIVE_SCENARIO = "high"

# Set by run_pipeline.py before calling train_model()
OM_NETWORK_DIR = None


def build_trainloader(scenario: str = None) -> list:
    """
    Single source of truth for the PPO trainloader. Returns a flat
    24-slot list at the scenario's verified total_veh_hr.
    Used by both run_pipeline.py and dataset/Traffic_Generator.py.
    """
    s = scenario or ACTIVE_SCENARIO
    total = DEMAND_SCENARIOS[s]["total_veh_hr"]
    return [float(total)] * 24


def crear_trips_directo(cant_coches, det):
    scenario = DEMAND_SCENARIOS[ACTIVE_SCENARIO]
    entry_weights = scenario["entry_weights"]
    exit_weights  = scenario["exit_weights"]

    depart_times = sorted([random.uniform(0, 3000) for _ in range(cant_coches)])

    lines = ['<trips>']
    for i, depart in enumerate(depart_times):
        from_edge = random.choices(_ENTRY_EDGES, weights=entry_weights, k=1)[0]
        to_edge   = random.choices(_EXIT_EDGES,  weights=exit_weights,  k=1)[0]
        lines.append(
            f'    <trip id="{i}" depart="{depart:.2f}" '
            f'from="{from_edge}" to="{to_edge}" departLane="random"/>'
        )
    lines.append('</trips>')

    with open(det + "/trips.xml", 'w') as f:
        f.write('\n'.join(lines))
    random.seed(None)


def simulacion(cant_coches, det):
    if OM_NETWORK_DIR is not None:
        target_dir = OM_NETWORK_DIR
    else:
        target_dir = os.path.join(_TRAINING_DIR, det)

    crear_trips_directo(cant_coches, target_dir)
    validar_trips(target_dir)
    crear_rutas(target_dir)
    llamar_SUMO(target_dir)

    while True:
        res = farm(target_dir)
        if res is not None:
            break
    return res