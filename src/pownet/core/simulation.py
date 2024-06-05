import pickle
import os

import highspy

from pownet.core.builder import ModelBuilder
from pownet.core.input import SystemInput
from pownet.core.record import SystemRecord
from pownet.processing.functions import (
    create_init_condition,
    get_current_time,
)
from pownet.folder_sys import get_output_dir
import pownet.config as config

class Simulator:
    def __init__(
        self,
        system_input: SystemInput,
        write_model: bool = False,
        use_gurobi: bool = True,
    ) -> None:

        self.model = None
        self.use_gurobi = use_gurobi

        self.system_input = system_input
        self.T = self.system_input.T

        self.model_name = system_input.model_name
        self.write_model = write_model

    def run(
        self, steps: int,init_conds, simulated_day,mip_gap: float = None, timelimit: float = None,
    ) -> SystemRecord:    # adding init_conds and simulated_day as inputs to function (run)
        # Initialize objects
        system_record = SystemRecord(self.system_input)
        builder = ModelBuilder(self.system_input)



        # The indexing of 'k' starts at zero because we use this to
        # index the parameters of future simulation periods (t + self.k*self.T)
        # Need to ensure that steps is a multiple of T

        STEP_BY_STEP=config.get_stepbystep()
        ONE_STEP=config.get_onestep()

        if STEP_BY_STEP or ONE_STEP:
            steps_to_run=1
        else:
            steps_to_run = min(steps, 365 * 24 // self.T)


        for k in range(0, steps_to_run):
            # Create a gurobipy model for each simulation period
            if STEP_BY_STEP or ONE_STEP:
                simulated_step=simulated_day
            else:
                simulated_step=k

            print("\n\n\n============")
            print(f"PowNet: Simulate step {simulated_step+1}\n\n")


            # self.model = builder.build(
            #     k=simulated_step,
            #     init_conds = init_conds,
            #     mip_gap=mip_gap,
            #     timelimit=timelimit,
            # )

            if k == 0:
                self.model = builder.build(
                    k=simulated_step,
                    init_conds=init_conds,
                    mip_gap=mip_gap,
                    timelimit=timelimit,
                )
            else:
                self.model = builder.update(
                    k=simulated_step,
                    init_conds=init_conds,
                    mip_gap=mip_gap,
                    timelimit=timelimit,
                )

            # We can write the model as .MPS and use non-Gurobi solvers
            if self.write_model:
                # Save the model
                dirname = os.path.join(
                    get_output_dir(), f"{self.model_name}_{self.T}_instances"
                )
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                self.model.write(os.path.join(dirname, f"{self.model_name}_{simulated_step}.mps"))

            # Solve the model with either Gurobi or HiGHs
            if self.use_gurobi:
                self.model.optimize()

            else:
                # Export the instance to MPS and solve with HiGHs
                mps_file = "temp_instance_for_HiGHs.mps"
                self.model.write(mps_file)

                self.model = highspy.Highs()
                self.model.readModel(mps_file)
                self.model.run()

                # Delete the MPS file
                os.remove(mps_file)

            # In case when the model is infeasible, we generate an output file
            # to troubleshoot the problem. The model should always be feasible.
            if self.use_gurobi:
                if self.model.status == 3:
                    print(f"PowNet: Iteration: {simulated_step} is infeasible.")
                    self.model.computeIIS()
                    c_time = get_current_time()
                    ilp_file = os.path.join(
                        get_output_dir(),
                        f"infeasible_{self.model_name}_{self.T}_{simulated_step}_{c_time}.ilp",
                    )
                    self.model.write(ilp_file)

                    mps_file = os.path.join(
                        get_output_dir(),
                        f"infeasible_{self.model_name}_{self.T}_{simulated_step}_{c_time}.mps",
                    )
                    self.model.write(mps_file)

                    # Need to learn about the initial conditions as well
                    with open(
                        os.path.join(
                            get_output_dir(),
                            f"infeasible_{self.model_name}_{self.T}_{simulated_step}_{c_time}.pkl",
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(system_record, f)
                    break

            # Need k to increment the hours field and also init_conds for next time step
            system_record.keep(self.model, simulated_step)
            init_conds = system_record.get_init_conds()

        return system_record
