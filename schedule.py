from ConstantEnums import *
from copy import deepcopy

class Schedule:
    def __init__(self):
        self.phases = [Phase.AWAKENING, Phase.GUIDED_LEARNING, Phase.REINFORCEMENT_LEARNING]
        self.teaching_plan = {
            Task.AWAKENING_TASKS: [],
            Task.PRE_TRAINING_TASKS: [
                # Task.LIMB_CONTROL_FL_BEND.value, Task.LIMB_CONTROL_FR_BEND.value, 
                # Task.LIMB_CONTROL_BL_BEND.value, Task.LIMB_CONTROL_BR_BEND.value,
                # Task.LIMB_CONTROL_FL_STRAIGHTEN.value, Task.LIMB_CONTROL_FR_STRAIGHTEN.value,
                # Task.LIMB_CONTROL_BL_STRAIGHTEN.value, Task.LIMB_CONTROL_BR_STRAIGHTEN.value
            ],
            Task.ADVANCED_TASKS: [Task.SIT.value, Task.STAND.value, Task.LIE.value, Task.RUN.value],
        }
        
        self.command_sequences = {
            Task.LIMB_CONTROL_FL_BEND.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [1, -1, -1, -1], CommandAttr.LSTIM.value: [1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 0, 0, 0]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 0, 0, 0]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_FL_STRAIGHTEN.value
            },
            Task.LIMB_CONTROL_FR_BEND.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, 1, -1, -1], CommandAttr.LSTIM.value: [-1, 1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 1, 0, 0]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, 1, -1, -1], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 1, 0, 0]]},
                },

                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_FR_STRAIGHTEN.value
            },
            Task.LIMB_CONTROL_BL_BEND.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 1, -1], CommandAttr.LSTIM.value: [-1, -1, 1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 1, 0]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 1, -1], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 1, 0]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_BL_STRAIGHTEN.value
            },
            Task.LIMB_CONTROL_BR_BEND.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, -1, 1], CommandAttr.LSTIM.value: [-1, -1, -1, 1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 1]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, -1, 1], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 1]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_BR_STRAIGHTEN.value
            },
            Task.LIMB_CONTROL_FL_STRAIGHTEN.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [2, -1, -1, -1], CommandAttr.LSTIM.value: [2, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 0, 0, 0]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [2, -1, -1, -1], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 0, 0, 0]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_FL_BEND.value
            },
            Task.LIMB_CONTROL_FR_STRAIGHTEN.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, 2, -1, -1], CommandAttr.LSTIM.value: [-1, 2, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 2, 0, 0]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, 2, -1, -1], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 2, 0, 0]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_FR_BEND.value
            },
            Task.LIMB_CONTROL_BL_STRAIGHTEN.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 2, -1], CommandAttr.LSTIM.value: [-1, -1, 2, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 2, 0]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 2, -1], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 2, 0]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_BL_BEND.value
            },
            Task.LIMB_CONTROL_BR_STRAIGHTEN.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, -1, 2], CommandAttr.LSTIM.value: [-1, -1, -1, 2], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 2]]},
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, -1, 2], CommandAttr.LSTIM.value:  [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 2]]},
                },
                Task.CONJUGATION_TASK.value: Task.LIMB_CONTROL_BR_BEND.value
            },

            Task.SIT.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [2, 2, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 0, 0]]},
                    1: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 1, 1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 1, 1]]},
                    2: {CommandAttr.A.value: "s", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 1, 1]]},
                    3: {CommandAttr.A.value: "i", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 1, 1]]},
                    4: {CommandAttr.A.value: "t", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 1, 1]]}
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: "s", CommandAttr.V.value:  [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    1: {CommandAttr.A.value: "i", CommandAttr.V.value:  [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    2: {CommandAttr.A.value: "t", CommandAttr.V.value:  [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 1, 1]]}
                },
                Task.CONJUGATION_TASK.value: None
            },
            Task.STAND.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [2, 2, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 0, 0]]},
                    1: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 2, 2], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]},
                    2: {CommandAttr.A.value: "s", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]},
                    3: {CommandAttr.A.value: "t", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]},
                    4: {CommandAttr.A.value: "a", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]},
                    5: {CommandAttr.A.value: "n", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]},
                    6: {CommandAttr.A.value: "d", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]}
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: "s", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    1: {CommandAttr.A.value: "t", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    2: {CommandAttr.A.value: "a", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    3: {CommandAttr.A.value: "n", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    4: {CommandAttr.A.value: "d", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 2, 2, 2]]}
                },
                Task.CONJUGATION_TASK.value: Task.LIE.value
            },

            Task.LIE.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [-1, -1, 1, 1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 1, 1]]},
                    1: {CommandAttr.A.value: None, CommandAttr.V.value: [1, 1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 1, 1, 1]]},
                    2: {CommandAttr.A.value: "l", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 1, 1, 1]]},
                    3: {CommandAttr.A.value: "i", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 1, 1, 1]]},
                    4: {CommandAttr.A.value: "e", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 1, 1, 1]]}
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: "l", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    1: {CommandAttr.A.value: "i", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    2: {CommandAttr.A.value: "e", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 1, 1, 1]]}
                },
                Task.CONJUGATION_TASK.value: Task.STAND.value
            },
            Task.RUN.value: {
                Phase.GUIDED_LEARNING.value: {
                    0: {CommandAttr.A.value: None, CommandAttr.V.value: [2, 1, 1, 2], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 1, 1, 2]]},
                    1: {CommandAttr.A.value: None, CommandAttr.V.value: [1, 2, 2, 1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 2, 2, 1]]},
                    2: {CommandAttr.A.value: None, CommandAttr.V.value: [2, 1, 1, 2], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[2, 1, 1, 2]]},
                    3: {CommandAttr.A.value: None, CommandAttr.V.value: [1, 2, 2, 1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[1, 2, 2, 1]]},
                    4: {CommandAttr.A.value: "r", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    5: {CommandAttr.A.value: "u", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    6: {CommandAttr.A.value: "n", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 1, CommandAttr.LIMB_STATE.value: [[2, 1, 1, 2], [1, 2, 2, 1],[2, 1, 1, 2], [1, 2, 2, 1]]}
                },
                Phase.REINFORCEMENT_LEARNING.value: {
                    0: {CommandAttr.A.value: "r", CommandAttr.V.value:  [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    1: {CommandAttr.A.value: "u", CommandAttr.V.value:  [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 0, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                    2: {CommandAttr.A.value: "n", CommandAttr.V.value:  [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 1, CommandAttr.LIMB_STATE.value: [[2, 1, 1, 2], [1, 2, 2, 1],[2, 1, 1, 2], [1, 2, 2, 1]]}
                },
                Task.CONJUGATION_TASK.value: None
            }

        }




        self.response_sequences = {
            Response.GOOD.value: {
                0: {CommandAttr.A.value: "g", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 0]]},
                1: {CommandAttr.A.value: "o", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 0]]},
                2: {CommandAttr.A.value: "o", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 0]]},
                3: {CommandAttr.A.value: "d", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[0, 0, 0, 0]]},
            },
            Response.NO.value: {
                0: {CommandAttr.A.value: "n", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
                1: {CommandAttr.A.value: "o", CommandAttr.V.value: [-1, -1, -1, -1], CommandAttr.LSTIM.value: [-1, -1, -1, -1], CommandAttr.CHECK.value: 1, CommandAttr.LOOP.value: 0, CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]]},
            },
        }

        self.reward_parameters = {
            Params.VARIATION_COEFFICIENT.value: {
                Params.REWARD_DECAY.value: 0.9,
                Params.DELAY_EXTENSION_COEFFICIENT.value: 0.1,
                Params.BREAKTHROUGH_BONUS_COEFFICIENT.value: 0.5,
                Params.ATTENUATION_COEFFICIENT.value: 0.8
            },

            Task.LIMB_CONTROL_FL_BEND.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_FR_BEND.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_BL_BEND.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_BR_BEND.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_FL_STRAIGHTEN.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_FR_STRAIGHTEN.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_BL_STRAIGHTEN.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIMB_CONTROL_BR_STRAIGHTEN.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},

            Task.SIT.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.STAND.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.LIE.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 3},
            Task.RUN.value: {Params.REWARD_DELAY.value: 0, Params.HOLD_DURATION.value: 5}
        }  # Reward structures for each phase/task

        self.input_timing_parameters = {
            Params.VARIATION_COEFFICIENT.value: {
                Params.NOISE_INCREMENT_COEFFICIENT.value: 0.1,
                Params.UPPER_LIMIT.value: 0.5
            },

            Task.LIMB_CONTROL_FL_BEND.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_FR_BEND.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_BL_BEND.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_BR_BEND.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_FL_STRAIGHTEN.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_FR_STRAIGHTEN.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_BL_STRAIGHTEN.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},
            Task.LIMB_CONTROL_BR_STRAIGHTEN.value: {Params.STIMULI_TIMING_DURATION.value: 3, Params.LETTER_TIMING_DURATION.value: -1},

            Task.SIT.value: {Params.STIMULI_TIMING_DURATION.value: -1, Params.LETTER_TIMING_DURATION.value: 3},
            Task.STAND.value: {Params.STIMULI_TIMING_DURATION.value: -1, Params.LETTER_TIMING_DURATION.value: 3},
            Task.LIE.value: {Params.STIMULI_TIMING_DURATION.value: -1, Params.LETTER_TIMING_DURATION.value: 3},
            Task.RUN.value: {Params.STIMULI_TIMING_DURATION.value: -1, Params.LETTER_TIMING_DURATION.value: 3},

            Response.GOOD.value: {Params.STIMULI_TIMING_DURATION.value: -1, Params.LETTER_TIMING_DURATION.value: 3},
            Response.NO.value: {Params.STIMULI_TIMING_DURATION.value: -1, Params.LETTER_TIMING_DURATION.value: 3}
        }  # Input variability settings

    def get_tasks(self, phase):
        """Retrieve tasks for a given phase."""
        return self.teaching_plan.get(phase, [])

    def get_task_details(self, task, phase):
        """Retrieve details for a specific task in a given phase."""
        return {
            Params.COMMAND_SEQUENCES.value: deepcopy(self.command_sequences.get(task, {}).get(phase, {})),
            Params.REWARD_PARAMETERS.value: {
                Params.VARIATION_COEFFICIENT.value: self.reward_parameters.get(Params.VARIATION_COEFFICIENT.value),
                task: self.reward_parameters.get(task)
            },
            Params.INPUT_TIMING_PARAMETERS.value: {
                Params.VARIATION_COEFFICIENT.value: self.input_timing_parameters.get(Params.VARIATION_COEFFICIENT.value),
                task: self.input_timing_parameters.get(task)
            },
        }


if __name__ == "__main__":
    # Initialize the Schedule
    schedule = Schedule()

    # Test phases retrieval
    print("Testing phase retrieval:")
    for phase in schedule.phases:
        print(f"Tasks for phase {phase.value}: {schedule.get_tasks(phase.value)}")

    # Test teaching plan integrity
    print("\nTesting teaching plan:")
    for task, details in schedule.teaching_plan.items():
        print(f"Task: {task.value}, Details: {details}")

    # Test command sequences retrieval
    print("\nTesting command sequences:")
    for task in schedule.command_sequences.keys():
        for phase in schedule.phases:
            commands = schedule.command_sequences[task].get(phase.value, None)
            if commands:
                print(f"Commands for task {task} in phase {phase.value}: {commands}")

    # Test reward parameters
    print("\nTesting reward parameters:")
    for task, params in schedule.reward_parameters.items():
        print(f"Reward parameters for {task}: {params}")

    # Test input timing parameters
    print("\nTesting input timing parameters:")
    for task, params in schedule.input_timing_parameters.items():
        print(f"Input timing parameters for {task}: {params}")

    # Test task details retrieval
    print("\nTesting task details retrieval:")
    for task in schedule.command_sequences.keys():
        for phase in schedule.phases:
            details = schedule.get_task_details(task, phase.value)
            print(f"Details for task {task} in phase {phase.value}: {details}")

    print("\nAll tests executed.")
