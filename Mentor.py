from schedule import Schedule
from awarenessCradle import Engine
from ConstantEnums import *
import random
class NoiseIncreaseProgress:
    """ Shows how noise gradually increase, and provides records for progress rollbacks. """
    def __init__(self, noise_increment_coefficient = 0.1, upper_limit = 0.5, basic_noise = 0):
        self.noise_increment_coefficient = noise_increment_coefficient
        self.upper_limit = upper_limit
        self.max_noise_tolerance = 0
        self.noise = basic_noise
        pass
    def tick(self):
        # finish the test under a certain noise, progress
        self.max_noise_tolerance = max(self.max_noise_tolerance,self.noise)
        self.noise = min(self.noise + self.noise_increment_coefficient, self.upper_limit)
        

    def rollback(self):
        self.noise = max(self.noise - self.noise_increment_coefficient, 0)


        

class RewardReductionProgress:
    """ Shows how reward is gradually reduced and delayed, and provides records for progress rollbacks. """
    def __init__(self, reward_probability = 1.4, delay_extension_coefficient = 0.1, breakthrough_bouns_coefficient = 0.5, attenuation_coefficient = 0.1,min_reward_probability = 0.1, delay_up_limit = 50, min_reward_size = 0.1):
        self.reward_probability = reward_probability 
        self.delay_extension_coefficient = delay_extension_coefficient
        self.breakthrough_bouns_coefficient = breakthrough_bouns_coefficient
        self.attenuation_coefficient = attenuation_coefficient
        self.min_reward_probability = min_reward_probability
        self.delay_up_limit = delay_up_limit
        self.min_reward_size = min_reward_size
        
        self.reward_delay = 0
        self.reward_size = 1

        pass

    def tick(self, breakthrough = False):
        if breakthrough == True:
            self.reward_size *= (1 + self.breakthrough_bouns_coefficient)
            self.reward_delay *= (1 - min(self.breakthrough_bouns_coefficient,1))
            self.reward_probability *= (1 + self.breakthrough_bouns_coefficient)
        else:
            self.reward_size = max(self.reward_size * (1 - max(self.attenuation_coefficient,0)), self.min_reward_size)
            self.reward_delay = min(self.reward_delay * (1 + self.delay_extension_coefficient), self.delay_up_limit)
            self.reward_probability = max(self.reward_probability * (1 - max(self.attenuation_coefficient,0)), self.min_reward_probability)

    def rollback(self):
        """ Reward doesn't support rollback"""
        
        pass

class SignalTiming:
    """The basic time step length of input signals, modified by noise and reward params"""
    def __init__(self, V_signal_timing = 30, A_signal_timing = 30, Lstim_signal_timing = 10, reward_signal_timing = 30):
        self.V_signal_timing = V_signal_timing
        self.A_signal_timing = A_signal_timing
        self.Lstim_signal_timing = Lstim_signal_timing
        self.reward_signal_timing = reward_signal_timing

    def tick(self):
        pass

class TaskTrainingProgress:
    """ The training stage and mastery level reached for a task, with rollback provided"""
    def __init__(self, task, conjugation_task = None, noise_params = {}, reward_params = {},  continuous_failure_tolerance = 5, proficiency_decay_rate = 0.7, proficiency_dampening_factor  = 20):
        self.task = task
        self.noise_increase_progress = NoiseIncreaseProgress(**noise_params)
        self.reward_reduction_progress = RewardReductionProgress(**reward_params)
        self.task_timing = SignalTiming()
        self.continuous_failure_tolerance = continuous_failure_tolerance
        self.phase = 0 # 0 guided learning, 1 reinforcement learning


        self.record = [] # (phase, noise_increase_progress, reward_reduction_progress)
        self.continuous_failure_times = 0
        self.guided_learning_failure_times = 0
        self.reinforcement_learning_failure_times = 0
        self.guided_learning_success_times = 0
        self.reinforcement_learning_success_times = 0
        
        self.proficiency_dampening_factor = proficiency_dampening_factor
        self.proficiency_decay_rate = proficiency_decay_rate
        self.guided_learning_proficiency = 0 # 
        self.reinforcement_learning_proficiency = 0 # 

        self.conjugation_task = conjugation_task

        self.best_completion_time = float('inf')  # Use infinity for better clarity
        self.best_action_time = float('inf')  # Use infinity for better clarity
        
    
        
    def tick(self, success=False, action_time=float('inf'), completion_time=float('inf')):

        
        if(success == False):
            self.continuous_failure_times+=1
            if (self.continuous_failure_times>=self.continuous_failure_tolerance):
                self.rollback()
        else:
            breakthrough = False

            if completion_time + action_time < self.best_completion_time + self.best_action_time:
                breakthrough = True
                self.best_completion_time = completion_time
                self.best_action_time = action_time

            if (breakthrough):
                self.record.append({"phase":self.phase,"noise":self.noise_increase_progress, "reward": self.reward_reduction_progress})
            self.noise_increase_progress.tick()
            self.reward_reduction_progress.tick(breakthrough=breakthrough)
        
        # update proficiency

        if self.phase == 0:
            # Weighted impact calculation
            impact = self.proficiency_decay_rate * self.guided_learning_proficiency + (1 - self.proficiency_decay_rate) * success

            # Normalize by total counts
            total_counts = self.guided_learning_success_times + self.guided_learning_failure_times
            self.guided_learning_proficiency = (impact * self.guided_learning_success_times) / (total_counts + self.proficiency_dampening_factor)
        elif self.phase == 1:
            # Weighted impact calculation
            impact = self.proficiency_decay_rate * self.reinforcement_learning_proficiency + (1 - self.proficiency_decay_rate) * success

            # Normalize by total counts
            total_counts = self.reinforcement_learning_success_times + self.reinforcement_learning_failure_times
            self.reinforcement_learning_proficiency = (impact * self.reinforcement_learning_success_times) / (total_counts + self.proficiency_dampening_factor)

        pass
        

    def rollback(self):
        if (self.record.__len__()>0):
            self.continuous_failure_times = 0
            archive = self.record.pop()
            self.phase = archive["phase"]
            self.continuous_failure_times = 0
            self.noise_increase_progress = archive["noise"]
            # self.reward_reduction_progress = archive["reward"] # don't rollback to earlier, higher reward
            
        pass

    def failure_rate(self):
        if self.phase == 0:
            return self.guided_learning_failure_times/(self.guided_learning_failure_times+self.guided_learning_success_times) if (self.guided_learning_failure_times+self.guided_learning_success_times)!=0 else 1
        else:
            return self.reinforcement_learning_failure_times/(self.reinforcement_learning_failure_times+self.reinforcement_learning_success_times) if (self.reinforcement_learning_failure_times+self.reinforcement_learning_success_times)!=0 else 1
        
    def get_proficiency(self):
        """Proficiency: A comprehensive score based on the number of completions and failure rate"""
        if self.phase == 0:
            return self.guided_learning_proficiency
        elif self.phase == 1:
            return self.reinforcement_learning_proficiency
        else:
            return 0 
        


            
        

class TaskPlanner:
    """When a task is completed, decide the next task based on the completion status of the task"""
    def __init__(self, tasks = {
            0:[
                (Task.LIMB_CONTROL_FL_BEND.value,Task.LIMB_CONTROL_FL_STRAIGHTEN.value), 
                (Task.LIMB_CONTROL_FR_BEND.value,Task.LIMB_CONTROL_FR_STRAIGHTEN.value), 
                (Task.LIMB_CONTROL_BL_BEND.value,Task.LIMB_CONTROL_BL_STRAIGHTEN.value),
                (Task.LIMB_CONTROL_BR_BEND.value,Task.LIMB_CONTROL_BR_STRAIGHTEN.value),
                (Task.LIMB_CONTROL_FL_STRAIGHTEN.value,Task.LIMB_CONTROL_FL_BEND.value), 
                (Task.LIMB_CONTROL_FR_STRAIGHTEN.value,Task.LIMB_CONTROL_FR_BEND.value),
                (Task.LIMB_CONTROL_BL_STRAIGHTEN.value,Task.LIMB_CONTROL_BL_BEND.value), 
                (Task.LIMB_CONTROL_BR_STRAIGHTEN.value,Task.LIMB_CONTROL_BR_BEND.value)
            ],
            1: [(Task.SIT.value,None), (Task.STAND.value,Task.LIE.value), (Task.LIE.value,Task.STAND.value)],
            2: [(Task.RUN.value,None)]},
            proficiency_threshold = 0.5

        ):
        """Regist all tasks"""

        if tasks.__len__() == 0:
            raise Exception("No tasks!")

        self.task_training_progress_manager = {
            stage: {task: TaskTrainingProgress(task = task, conjugation_task=conjugation_task) for (task,conjugation_task) in task_list}
            for stage, task_list in tasks.items()
        }

        # self.proficiency_manager = {
        #     stage: {task: progress.get_proficiency()  for progress in progress_list}
        #     for stage, progress_list in tasks.items()
        # }
        # self.phase_manager = {
        #     stage: {task: progress.phase  for progress in progress_list}
        #     for stage, progress_list in tasks.items()
        # }

        # Task management
        self.planned_queue = []  # Stores tasks to be conducted (time from 0 to n )
        self.history_queues = {}  # Stores completed tasks by task name (time from 0 to n )

        self.proficiency_threshold = proficiency_threshold
        self.current_stage = min(self.task_training_progress_manager.keys())
        self.current_task = None
        self.current_task_training_progress:TaskTrainingProgress = None
        for task, progress in self.task_training_progress_manager[self.current_stage].items():
            self.current_task = task # any element in value: task_list 
            self.current_task_training_progress = progress # the reference to the corresponding object in "self.task_training_progress_manager"
            break
        pass

    def update_stage(self):
        # from the lowest stage to the highest, if any proficiency lower than self.proficiency_threshold, train tasks at this stage.
        for stage, progress_list in self.task_training_progress_manager.items():
            if any(progress.get_proficiency() < self.proficiency_threshold for (task, progress) in progress_list):
                self.current_stage = stage
                break
        pass 
        
    def get_task_training_progress(self, __task):
        for stage, progress_list in self.task_training_progress_manager.items():
            for (task, progress) in progress_list:
                if task == __task:
                    return progress
        
    
    def choose_task_based_on_weight(task_weights):
        """
        Choose a task from a dictionary of task-weight pairs based on the weights.

        Args:
            task_weights (dict): A dictionary where keys are tasks and values are their corresponding weights.

        Returns:
            The chosen task (key from the dictionary).
        """
        if not task_weights:
            raise ValueError("The task_weights dictionary is empty.")

        tasks, weights = zip(*task_weights.items())
        
        # Ensure weights are non-negative
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        # Normalize weights to create a probability distribution
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("The sum of weights must be greater than 0.")

        probabilities = [w / total_weight for w in weights]

        # Use random.choices to select based on weights
        chosen_task = random.choices(tasks, probabilities)[0]

        return chosen_task



    def tick(self, reset = False, success = False, action_time=float('inf'), completion_time=float('inf')):
        """give feedback on the current task, and the task planner will provide new suitable task policies"""
        # if you want to reset the task, you means you didn't train for the task because there are conflicts in the objective state and start state, the planner will choose another different task and the conjugation task is preferred.


        if self.current_task == None:
            self.update_stage()
            for task, progress in self.task_training_progress_manager[self.current_stage].items():
                self.current_task = task # any element in value: task_list 
                self.current_task_training_progress = progress # the reference to the corresponding object in "self.task_training_progress_manager"
                break
        pass
        # Priority 1
        if (reset == True): 
            self.update_stage()
            if self.current_task_training_progress.conjugation_task != None and self.get_task_training_progress(self.current_task_training_progress.conjugation_task).get_proficiency() < self.proficiency_threshold:
                
                self.current_task = self.current_task_training_progress.conjugation_task
                self.current_task_training_progress = self.get_task_training_progress(self.current_task)
                return

        # 2

        else:
            self.current_task_training_progress.tick(success=success, action_time=action_time, completion_time=completion_time)
            self.update_stage()

            # if failed, repeat
            if success == False:
                return # continue with current task
            
            

            ### if succeeded
            # if task or conjungation task is in guided training phase, adopt conjungation task
            if self.current_task_training_progress.conjugation_task != None:
                conjugation_task_progress:TaskTrainingProgress = self.get_task_training_progress(self.current_task_training_progress.conjugation_task)
                if (self.current_task_training_progress.phase == 0) or conjugation_task_progress.phase == 0:
                    self.current_task = self.current_task_training_progress.conjugation_task
                    self.current_task_training_progress = conjugation_task_progress

            
            # if any task progress (except current task) is in guided learning period and the training process has been launched (has at least one execution), choose it.
            for (task, progress) in self.task_training_progress_manager[self.current_stage]:
                if task == self.current_task:
                    continue
                if progress.phase == 0 and (progress.guided_learning_failure_times + progress.guided_learning_success_times) > 0:
                    self.current_task = task
                    self.current_task_training_progress = progress
                    return 
                
            # if all task progresses in process are in reinforcement learning period, and include at least one task proficiency less than theshold, choose one within them to train until all tasks in process pass (low proficiency, higher stage tasks are preferred)
            exist_less_proficient_task = False
            reinforcement_learning_task_weights = {} # (task, weights) pairs
            guided_learning_task = []
            for (stage, progress_list) in self.task_training_progress_manager.items():
                if stage > self.current_stage and len(guided_learning_task)>0:
                    break
                for (task, progress) in progress_list:
                    if progress.phase == 1:
                        exist_less_proficient_task += int(progress.get_proficiency() < self.proficiency_threshold)
                        weight = ((1.01-progress.get_proficiency()) + stage) * ( 1 + int(progress.get_proficiency() < self.proficiency_threshold))
                        reinforcement_learning_task_weights[task] = weight
                    else:
                        guided_learning_task.append(task)


            if exist_less_proficient_task:
                self.current_task = self.choose_task_based_on_weight(reinforcement_learning_task_weights)
                self.current_task_training_progress = self.get_task_training_progress(self.current_task)
                return
            else:
                if len(guided_learning_task) > 0:
                    # start a new task training progress
                    self.current_task = guided_learning_task.pop()
                    self.current_task_training_progress = self.get_task_training_progress(self.current_task)
                else:
                    # all tasks are proficient
                    self.current_task = None
                    self.current_task_training_progress = None

            
            pass


class Mentor:
    def __init__(self, schedule:Schedule, engine_reference:Engine):
        self.schedule = schedule  # Reference to the Schedule
        self.current_phase = Phase.GUIDED_LEARNING  # Start with the pre-training phase
        self.current_task = Task.LIMB_CONTROL_BL_BEND  # Active task
        self.time = 0
        self.baby_limb_state_queue = []
        # self.last_check_baby_limb_state = [-1,-1,-1,-1] # the last limb state of last check cmd
        
        
        self.task_planner = TaskPlanner()
        self.training_status = None
        self.training_result = None

        # # Task management
        # self.planned_queue = []  # Stores tasks to be conducted (time from 0 to n )
        # self.history_queues = {}  # Stores completed tasks by task name (time from 0 to n )

        # command sequence
        self.command_queue:list[dict] = [] # store commands of one single task (time from 0 to n )
        # self.response_queue = [] # store reward of one single task (time from 0 to n )
        self.last_time_step_command = {}
        self.actions_between_checks = 0 
        self.single_step_command_waiting = 0
        self.single_step_command_waiting_up_limit = 0 #(3 * actions between two checks)

        # # command and feedback forward to the Engine at self.time
        # self.current_command = {} 
        # self.current_reward = {}

        # self.task_state = {
        #     "transition_state": None,  # Tracks transition states between actions
        #     "start_time": None,  # Start time for the current task
        #     "action_time": None,  # Tracks time taken for each action
        # }  # Progress tracking within tasks
        # self.start_time = float('inf')
        self.action_time = float('inf')
        self.completion_time = float('inf')

        # self.analysis = {}  # Baby performance metrics
        self.engine_reference = engine_reference  # Reference to the Engine

        # Time-dependent state management
        self.time_step = 0  # Current time step within a task
        self.time_span = 10  # Default time span for input signals
        self.reward_delay = 0  # Delay for issuing rewards
        self.noise_level = 0.0  # Current noise affecting signal timing


        

        # Metrics for task analysis
        self.metrics = {
            "completion_status": "in progress",  # Task completion status
            "completion_time": None,  # Time taken to complete the task
            "action_time": [],  # List of times taken for each action
            "retry_count": 0  # Number of retries for the task
        }



    def reset_task(self, task, baby_limb_state):
        """ Judge if a task reset is required"""
        
        if (baby_limb_state == [-1,-1,-1,-1]):
            return False
        task_details = self.schedule.get_task_details(task=task, phase=1)
        command_sequence = task_details[Params.COMMAND_SEQUENCES.value]
        _,final_state = command_sequence.items()[-1]
        if len(final_state) != 1:
            return False
        else:
            if [baby_limb_state] ==  final_state:
                return True
            else:
                return False




    def generate_response_sequence(self, phase, succeed, A_signal_timing, reward_probability, reward_delay, reward_size, reward_signal_timing, noise):
        noise = noise/2
        if succeed == False:
            command_sequence = self.schedule.response_sequences.get(Response.NO.value)
            num_cmd = len(command_sequence.items())
            input_sequence = []
            if num_cmd == 0:
                raise Exception("Invalid command sequence!")
            
            for step, details in command_sequence.items():
                
                max_signal_timing = 0
                A_signal_timing_noise = 0
                
                if (details[CommandAttr.A.value]!=None):
                    A_signal_timing_noise = A_signal_timing * (1 + random.uniform(-noise, noise))
                
                max_signal_timing = int(max(A_signal_timing_noise,1))
                for i in range(0,max_signal_timing):
                    time_adjusted_signal = {
                        CommandAttr.A.value: details.get(CommandAttr.A.value) ,
                        CommandAttr.V.value: [-1,-1,-1,-1],
                        CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                        CommandAttr.CHECK.value: 0,
                        CommandAttr.LOOP.value: 0,
                        CommandAttr.LIMB_STATE.value: [[-1,-1,-1,-1]],
                        CommandAttr.REWARD_SIZE.value: 0 
                    }
                    input_sequence.append(time_adjusted_signal)
        else:
            with_reward = random.uniform(0, 1) < reward_probability
            command_sequence = self.schedule.response_sequences.get(Response.GOOD.value)
            num_cmd = len(command_sequence.items())
            input_sequence = []
            if num_cmd == 0:
                raise Exception("Invalid command sequence!")
            if phase == 0:
                
                for step, details in command_sequence.items():
                    
                    max_signal_timing = 0
                    A_signal_timing_noise = 0
                    
                    if (details[CommandAttr.A.value]!=None):
                        A_signal_timing_noise = A_signal_timing * (1 + random.uniform(-noise, noise))
                    
                    max_signal_timing = int(max(A_signal_timing_noise,1))
                    for i in range(0,max_signal_timing):
                        time_adjusted_signal = {
                            CommandAttr.A.value: details.get(CommandAttr.A.value) ,
                            CommandAttr.V.value: [-1,-1,-1,-1],
                            CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                            CommandAttr.CHECK.value: 0,
                            CommandAttr.LOOP.value: 0,
                            CommandAttr.LIMB_STATE.value: [[-1,-1,-1,-1]],
                            CommandAttr.REWARD_SIZE.value: reward_size * with_reward
                        }
                        input_sequence.append(time_adjusted_signal)
            else:
                for step, details in command_sequence.items():
                    
                    max_signal_timing = 0
                    A_signal_timing_noise = 0
                    
                    if (details[CommandAttr.A.value]!=None):
                        A_signal_timing_noise = A_signal_timing * (1 + random.uniform(-noise, noise))
                    
                    max_signal_timing = int(max(A_signal_timing_noise,1))
                    for i in range(0,max_signal_timing):
                        time_adjusted_signal = {
                            CommandAttr.A.value: details.get(CommandAttr.A.value) ,
                            CommandAttr.V.value: [-1,-1,-1,-1],
                            CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                            CommandAttr.CHECK.value: 0,
                            CommandAttr.LOOP.value: 0,
                            CommandAttr.LIMB_STATE.value: [[-1,-1,-1,-1]],
                            CommandAttr.REWARD_SIZE.value: 0
                        }
                        input_sequence.append(time_adjusted_signal)

                    for i in range(0,reward_delay):
                        time_adjusted_signal = {
                            CommandAttr.A.value: details.get(CommandAttr.A.value) ,
                            CommandAttr.V.value: [-1,-1,-1,-1],
                            CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                            CommandAttr.CHECK.value: 0,
                            CommandAttr.LOOP.value: 0,
                            CommandAttr.LIMB_STATE.value: [[-1,-1,-1,-1]],
                            CommandAttr.REWARD_SIZE.value: 0
                        }
                        input_sequence.append(time_adjusted_signal)

                    if with_reward:
                        for i in range(0,num_cmd*A_signal_timing):
                            time_adjusted_signal = {
                                CommandAttr.A.value: details.get(CommandAttr.A.value) ,
                                CommandAttr.V.value: [-1,-1,-1,-1],
                                CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                                CommandAttr.CHECK.value: 0,
                                CommandAttr.LOOP.value: 0,
                                CommandAttr.LIMB_STATE.value: [[-1,-1,-1,-1]],
                                CommandAttr.REWARD_SIZE.value: reward_size
                            }
                            input_sequence.append(time_adjusted_signal)

        return input_sequence
    def generate_input_sequence(self, task, phase, A_signal_timing, V_signal_timing, Lstim_signal_timing, reward_size, reward_signal_timing, reward_delay, noise ):
        """Logic: Generate time-dependent input stimuli for the current task using task details and strategies. 
        The continuous reward strength of guided learning is the ratio of the reward strength (reward_size) to the signal timing"""
        if task == None:
            raise Exception("Task None!")

        task_details = self.schedule.get_task_details(task=task, phase=phase)
        command_sequence = task_details[Params.COMMAND_SEQUENCES.value]
        input_sequence = []

        num_cmd = len(command_sequence.items())
        if num_cmd == 0:
            raise Exception("Invalid command sequence!")
        
        for step, details in command_sequence.items():
            
            max_signal_timing = 0
            A_signal_timing_noise = 0
            V_signal_timing_noise = 0
            Lstim_signal_timing_noise = 0
            if (details[CommandAttr.A.value]!=None):
                A_signal_timing_noise = A_signal_timing * (1 + random.uniform(-noise, noise))
            if(details[CommandAttr.V.value]!=[-1, -1, -1, -1]):
                V_signal_timing_noise = V_signal_timing * (1 + random.uniform(-noise, noise))
            if(details[CommandAttr.LSTIM.value]!=[-1, -1, -1, -1]):
                Lstim_signal_timing_noise = Lstim_signal_timing * (1 + random.uniform(-noise, noise))
            
            max_signal_timing = int(max(A_signal_timing_noise,V_signal_timing_noise,Lstim_signal_timing_noise,1))
            for i in range(0,max_signal_timing):
                time_adjusted_signal = {
                    CommandAttr.A.value: details.get(CommandAttr.A.value) if i< A_signal_timing_noise else None,
                    CommandAttr.V.value: details.get(CommandAttr.V.value) if i < V_signal_timing_noise else [-1,-1,-1,-1],
                    CommandAttr.LSTIM.value: details.get(CommandAttr.LSTIM.value) if i < Lstim_signal_timing_noise else [-1,-1,-1,-1],
                    CommandAttr.CHECK.value: details.get(CommandAttr.CHECK.value) if i == (max_signal_timing-1) else 0,
                    CommandAttr.LOOP.value: details.get(CommandAttr.LOOP.value) if i == (max_signal_timing-1) else 0,
                    CommandAttr.LIMB_STATE.value: details.get(CommandAttr.LIMB_STATE.value),
                    CommandAttr.REWARD_SIZE.value: reward_size/(num_cmd* max_signal_timing) if phase == 0 else 0 
                }
                input_sequence.append(time_adjusted_signal)
        last_cmd = command_sequence.items()[-1]
        for j in range(0, int(reward_delay)):
            # if the last cmd does not loop and check is required, holds the status before final response and reward.
            time_adjusted_signal = {
                CommandAttr.A.value: None,
                CommandAttr.V.value: [-1,-1,-1,-1],
                CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                CommandAttr.CHECK.value: last_cmd.get(CommandAttr.CHECK.value) if (last_cmd.get(CommandAttr.LOOP.value) == 0) else 0,
                CommandAttr.LOOP.value: last_cmd.get(CommandAttr.LOOP.value) if i == (max_signal_timing-1) else 0,
                CommandAttr.LIMB_STATE.value: [[0, 0, 0, 0]],
                CommandAttr.REWARD_SIZE.value: 0 

            }
            input_sequence.append(time_adjusted_signal)
        return input_sequence

    def get_task_params(self):
        if self.task_planner == None:
            raise Exception("TaskPlanner None!")
        return {
            "succeed": True if self.training_result == TrainingResult.SUCCESS.value else False,
            "task":self.task_planner.current_task,
            "phase":self.task_planner.current_task_training_progress.phase,
            "A_signal_timing":self.task_planner.current_task_training_progress.task_timing.A_signal_timing,
            "V_signal_timing":self.task_planner.current_task_training_progress.task_timing.V_signal_timing,
            "Lstim_signal_timing":self.task_planner.current_task_training_progress.task_timing.Lstim_signal_timing,
            "reward_size": self.task_planner.current_task_training_progress.reward_reduction_progress.reward_size,
            "reward_signal_timing":self.task_planner.current_task_training_progress.task_timing.reward_signal_timing,
            "reward_delay" : self.task_planner.current_task_training_progress.reward_reduction_progress.reward_delay,
            "noise":self.task_planner.current_task_training_progress.noise_increase_progress.noise,

        }
    def limb_state_consistency(ls_1:list, ls_2:list, ls_expect:list):
        """
        Monitoring the consistency of limb states that need to remain constant. Even if "check" is false in the command
        ls_1: start state
        ls_2: end state
        ls_expect: expected end state
        """
        for i in range(0,4):
            if ls_expect[i] == -1 or ls_1[i] == -1 or ls_2[i] == -1:
                continue
            if ls_expect[i] == 0 and ls_1[i]!=ls_2[i]:
                return False
            
        return True

    def limb_state_gradient_consistency(ls_1:list, ls_2:list, ls_expect:list):
        """
        Checking the changing trends consistency of limb states. Used when "check" is true in the command
        ls_1: start state
        ls_2: end state
        ls_expect: expected end state
        """
        for i in range(0,4):
            if ls_expect[i] == -1 or ls_1[i] == -1 or ls_2[i] == -1:
                continue
            if ls_expect[i] == 0 and ls_1[i]!=ls_2[i]:
                return False
            if ls_expect[i]!=0 and abs(ls_2[i]-ls_1[i]) > abs(ls_expect[i]-ls_1[i]):
                return False
        return True

    def limb_state_equal(ls,ls_expect):
        """Check if two limb states the same, based on ls_expect"""
        for i in range(0,4):

            if ls_expect[i]==ls[i] or ls_expect[i] == -1 or ls_expect == 0:
                continue
            if ls[i] == -1 or ls[i]== 0:
                return False
            
        return True
            
            

    def tick(self, time, baby_limb_state):
        self.time += 1
        if self.time != time:
            raise Exception("Mentor sync failed!")
        
        # initial task
        if self.task_planner.current_task == None:
            
            # initialize

            self.task_planner.tick()
            while self.reset_task(task=self.task_planner.current_task,baby_limb_state=baby_limb_state):
                self.task_planner.tick(reset=True)

            self.command_queue.clear()
            # self.response_queue.clear()
            self.baby_limb_state_queue.clear()
            self.training_result = None
            self.actions_between_checks = 0 
            self.single_step_command_waiting = 0
            self.single_step_command_waiting_up_limit = 0


            task_params = self.get_task_params()
            cmd_sequence = self.generate_input_sequence(task_params)
            self.command_queue.extend(cmd_sequence)
            self.training_status = TrainingStatus.IN_PROGRESS.value
            self.baby_limb_state_queue.append(baby_limb_state)
            return
        else:
            if (self.training_status == TrainingStatus.IN_PROGRESS.value): 
                if self.command_queue.__len__() == 0:
                    raise Exception("Invalid Command Queue")
                # Normally, if a check is not needed, just pops up the first command
                if (self.command_queue[0].get(CommandAttr.CHECK, False) == False):
                    if self.limb_state_consistency(self.baby_limb_state_queue[-1], baby_limb_state, self.command_queue[0].get(CommandAttr.LIMB_STATE.value,[-1,-1,-1,-1]) ):
                        self.actions_between_checks+=1
                        self.last_time_step_command = self.command_queue.pop(0)
                        self.baby_limb_state_queue.append(baby_limb_state)
                        # self.action_time+=1
                        # self.completion_time+=1
                        return
                    else:
                        # Training failure, reply NO
                        self.last_time_step_command = self.command_queue.pop(0)
                        self.training_status = TrainingStatus.IN_RESPONSE.value
                        self.training_result = TrainingResult.FAILURE.value
                        response_params = self.get_task_params()
                        self.command_queue = self.generate_response_sequence(**response_params)
                        self.baby_limb_state_queue=[baby_limb_state]
                        self.action_time = float('inf')
                        self.single_step_command_waiting = 0


                else:
                #if a check is needed, wait until the action done or time out
                    if (self.last_time_step_command.get(CommandAttr.CHECK.value, False) == False):
                        self.single_step_command_waiting_up_limit = self.actions_between_checks* 3 * len(self.command_queue[0].get(CommandAttr.LIMB_STATE.value, []))
                        self.action_time = 0
                        self.baby_limb_state_queue = [baby_limb_state]
                        self.last_time_step_command = self.command_queue[0]
                        # prepare all limb state needed to be achieved sequentially
                        self.expected_limb_state_queue:list = self.command_queue[0].get(CommandAttr.LIMB_STATE.value) #[state1, state2,...]
                        return
                    else:
                        # Wait until the action done:
                        self.action_time+=1
                        self.single_step_command_waiting +=1
                        if (self.single_step_command_waiting > self.single_step_command_waiting_up_limit):
                            
                            # Training failure, reply NO
                            self.last_time_step_command = self.command_queue.pop(0)
                            self.training_status = TrainingStatus.IN_RESPONSE.value
                            self.training_result = TrainingResult.FAILURE.value
                            response_params = self.get_task_params()
                            self.command_queue = self.generate_response_sequence(**response_params)
                            self.baby_limb_state_queue=[baby_limb_state]
                            self.action_time = float('inf')
                            self.single_step_command_waiting = 0
                        else:
                            if (self.limb_state_gradient_consistency(self.baby_limb_state_queue[-1],baby_limb_state,self.expected_limb_state_queue[0])):
                                if self.limb_state_equal(baby_limb_state,self.expected_limb_state_queue[0]):
                                    # pop
                                    self.last_time_step_command = self.command_queue.pop(0)
                                    self.baby_limb_state_queue.append(baby_limb_state)
                                    self.single_step_command_waiting = 0
                                    self.single_step_command_waiting_up_limit = 0
                                    if self.expected_limb_state_queue.__len__ == 0:
                                        # reply GOOD
                                        # self.last_time_step_command = self.command_queue.pop(0)
                                        self.training_status = TrainingStatus.IN_RESPONSE.value
                                        self.training_result = TrainingResult.SUCCESS.value
                                        response_params = self.get_task_params()
                                        self.command_queue = self.generate_response_sequence(**response_params)
                                        self.baby_limb_state_queue=[baby_limb_state]
                                        self.single_step_command_waiting = 0
                                        return
                                    else:
                                        return

                                else:
                                    self.single_step_command_waiting += 1
                        
                            else:
                                # Training failure, reply NO
                                self.last_time_step_command = self.command_queue.pop(0)
                                self.training_status = TrainingStatus.IN_RESPONSE.value
                                self.training_result = TrainingResult.FAILURE.value
                                response_params = self.get_task_params()
                                self.command_queue = self.generate_response_sequence(**response_params)
                                self.baby_limb_state_queue=[baby_limb_state]
                                self.action_time = float('inf')
                                self.single_step_command_waiting = 0


            elif (self.training_status == TrainingStatus.IN_RESPONSE.value):
                self.last_time_step_command = self.command_queue.pop(0)
                self.baby_limb_state_queue = [baby_limb_state]
                if self.command_queue.__len__() == 0:
                    
                    self.task_planner.tick(success=(self.training_result == TrainingResult.SUCCESS.value),action_time=self.action_time,completion_time=self.completion_time)
                    
                    while self.reset_task(task=self.task_planner.current_task,baby_limb_state=baby_limb_state):
                        self.task_planner.tick(reset=True)

                    # self.baby_limb_state_queue.clear()

                    task_params = self.get_task_params()

                    cmd_sequence = self.generate_input_sequence(task_params)
                    self.command_queue.extend(cmd_sequence)
                    self.training_status = TrainingStatus.IN_PROGRESS.value
                    self.training_result = None
                    return
                else:
                    return
    def get_latest_command(self):
        
        return self.time,self.command_queue[-1] if self.time != 0 else {
                CommandAttr.A.value: None,
                CommandAttr.V.value: [-1,-1,-1,-1],
                CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                CommandAttr.CHECK.value: 0,
                CommandAttr.LOOP.value: 0,
                CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]],
                CommandAttr.REWARD_SIZE.value: 0 

            }


    # def monitor_baby_state(self):
    #     """Logic: Monitor the baby's state over a time span. Collect outputs at each time step and return them as a list."""
    #     monitored_states = map()
    #     for _ in range(self.time_span):
    #         baby_output = self.engine_reference.get_baby_output()
    #         monitored_states.append(baby_output)
    #     return monitored_states

    # def analyze_performance(self, monitored_states):
    #     """Logic: Analyze performance based on monitored states. Calculate response time, check task completion, and identify transition states."""
    #     action_started = False
    #     action_start_time = None
    #     transition_states = []

    #     for time_step, state in enumerate(monitored_states):
    #         if state["action_status"] == "transition":
    #             transition_states.append(state)
    #         elif state["action_status"] == "completed":
    #             action_started = True
    #             if action_start_time is None:
    #                 action_start_time = time_step

    #     response_time = action_start_time if action_started else None
    #     task_completion = all(state["task_status"] == "completed" for state in monitored_states)

    #     self.metrics = {
    #         "completion_status": "completed" if task_completion else "in progress",
    #         "completion_time": response_time if task_completion else None,
    #         "action_time": [state.get("action_time") for state in transition_states],
    #         "retry_count": self.metrics["retry_count"] + (1 if not task_completion else 0)
    #     }
    #     return self.metrics

    # def manage_tasks(self):
    #     """Logic: Manage task progression. If the queue is empty, initialize tasks. On completion, move tasks to history and update/reset; on failure, retry with adjustments."""
    #     if not self.planned_queue:
    #         primary_tasks = self.schedule.get_phase_tasks(self.current_phase)
    #         intermediate_tasks = self.generate_intermediate_tasks()
    #         self.planned_queue.extend(primary_tasks + intermediate_tasks)

    #     if self.metrics["completion_status"] == "completed":
    #         completed_task = self.planned_queue.pop(0)
    #         task_name = completed_task["name"]
    #         phase = completed_task["phase"]

    #         if task_name not in self.history_queues:
    #             self.history_queues[task_name] = []
    #         self.history_queues[task_name].append(completed_task)

    #         self.update_best_score(task_name, completed_task)

    #         # Avoid adding reset tasks immediately after completing a reset task
    #         if "reset" not in task_name:
    #             self.planned_queue.extend(self.generate_intermediate_tasks())
    #     else:
    #         current_task = self.planned_queue[0]
    #         task_name = current_task["name"]
    #         if task_name in self.history_queues and self.history_queues[task_name]:
    #             retry_task = self.history_queues[task_name][-1]
    #             self.planned_queue.insert(1, retry_task)

    # def generate_intermediate_tasks(self):
    #     """Logic: Generate tasks to reset limb states or review skills. Return a list of task dictionaries with default configurations."""
    #     return [
    #         {"name": "reset_limb_state", "phase": self.current_phase, "coefficients": {"noise": 0.1}},
    #         {"name": "review_previous_task", "phase": self.current_phase, "coefficients": {"noise": 0.05}}
    #     ]

    # def update_best_score(self, task_name, completed_task):
    #     """Logic: Update the best performance metrics for the given task, such as noise or reward delay tolerance."""
    #     pass

    # def get_next_phase(self):
    #     """Logic: Determine the next phase of training. Return the next phase or default to the last phase."""
    #     phases = self.schedule.phases
    #     current_index = phases.index(self.current_phase)
    #     if current_index + 1 < len(phases):
    #         return phases[current_index + 1]
    #     return "Reinforcement Learning"

    # def strategy_management(self):
    #     """Logic: Adjust cultivation strategies based on task progress. Update noise and reward parameters dynamically."""
    #     if self.metrics["completion_status"] == "completed":
    #         self.noise_strategy_set["add_noise_in_command"] *= 0.9
    #         self.reward_strategy_set["delayed_feedback"] = max(1, self.reward_strategy_set["delayed_feedback"] - 1)
    #     else:
    #         self.noise_strategy_set["add_noise_in_command"] *= 1.1
    #         self.reward_strategy_set["delayed_feedback"] += 1

    

    # def issue_stimuli(self):
    #     """Logic: Send generated stimuli to the engine to forward them to the baby."""
    #     stimuli = self.generate_input_sequence()
    #     if stimuli:
    #         for stimulus in stimuli:
    #             self.engine_reference.forward_input_to_baby(stimulus)

    # def manage_training(self):
    #     """Logic: Monitor baby state, analyze performance, adjust strategies, and manage tasks dynamically."""
    #     monitored_states = self.monitor_baby_state()
    #     performance_report = self.analyze_performance(monitored_states)
    #     self.strategy_management()
    #     self.manage_tasks()
    #     return performance_report
