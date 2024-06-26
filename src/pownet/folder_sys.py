import os


def get_pownet_dir() -> str:
    """Does not assume the user saves the folder under their home directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_output_dir() -> str:
    return os.path.join(get_pownet_dir(), "outputs")

# is this still needed?
def get_input_dir() -> str:
    return os.path.join(get_pownet_dir(), "user_inputs")

# is this still needed?
def get_temp_dir() -> str:
    return os.path.join(get_pownet_dir(), "temp")


def get_model_dir() -> str:
    return os.path.join(get_pownet_dir(), "model_library")


def get_home_dir() -> str:
    return os.path.expanduser("~")


def get_database_dir() -> str:
    return os.path.join(get_pownet_dir(), "database")

# is this needed?
def get_test_dir() -> str:
    return os.path.join(get_pownet_dir(), "src", "test_pownet")


def delete_all_gurobi_solutions() -> None:
    """Remove all Gurobi solution files from the output folder.
    Use this function at the beginning of the simulation when warmstart is on.
    """
    solution_files = os.listdir(get_output_dir())
    for s_file in solution_files:
        file_extension = os.path.splitext(s_file)[1]
        if ".sol" in file_extension:
            os.remove(os.path.join(get_output_dir(), s_file))


def count_mps_files(instance_folder: str) -> int:
    """Count the number of files ending with .mps"""
    num_instances = 0
    for file in os.listdir(instance_folder):
        if file.endswith(".mps"):
            num_instances += 1
    return num_instances
