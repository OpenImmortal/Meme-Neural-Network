from schedule import Schedule
from ConstantEnums import *
import random
# ==================================================================================================
# AwarenessCradle - Neurodevelopmental Training Environment
#
# A structured learning system that implements progressive task training with dynamic difficulty
# adjustment through noise/reward mechanisms. Key components:
#
# 1. Noise/Reward Progression - Gradually increases environmental challenges while reducing rewards
# 2. Task Phasing - Implements guided learning -> reinforcement learning transitions
# 3. Proficiency Tracking - Maintains metrics for skill mastery assessment
# 4. Adaptive Planning - Dynamically selects tasks based on performance metrics
#
# Architecture Components:
# - Mentor: Orchestrates training process, manages task sequencing and reward signals
# - Baby: The learning neural network agent that adapts to environmental stimuli
# - Engine: Manages environment state, limb control, and system feedback loops
# - TaskPlanner: Dynamic task selection based on proficiency metrics
# - Noise/Reward Controllers: Manage challenge/reward progression curves
#
# Key Design Patterns:
# - Progressive Difficulty: Challenge escalation with safety rollback mechanisms
# - Delayed Gratification: Gradual transition from immediate to delayed rewards
# - Skill Transfer: Conjugated task training for generalized skill development
# - Adaptive Feedback: Real-time performance analysis driving task selection
# ==================================================================================================


class NoiseIncreaseProgress:
    """Manages progressive noise introduction with rollback capabilities for neurodevelopmental training.
    
    Implements a safety-checked noise progression system that:
    - Gradually increases environmental challenges
    - Tracks maximum successful noise tolerance
    - Allows controlled regression when failures occur
    
    Attributes:
        noise_increment_coefficient (float): Percentage increase per successful iteration (range 0-1)
        upper_limit (float): Absolute maximum noise allowed in system (range 0-1)
        max_noise_tolerance (float): Highest verified working noise level
        noise (float): Current active noise level (range 0-upper_limit)

    Methods:
        tick(): Progressively increase noise after successful iterations
        rollback(): Safely reduce noise level on consecutive failures

    Usage Example:
        noise_progress = NoiseIncreaseProgress(noise_increment_coefficient=0.1, upper_limit=0.5)
        for _ in range(5):
            noise_progress.tick()  # Incrementally increases noise
        noise_progress.rollback()  # Reduce noise after failures

    Relationship to System:
        - Integrated with TaskTrainingProgress for challenge scaling
        - Coordinated with RewardReductionProgress for difficulty balance
    """
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
    """Manages dynamic reward shaping for neurodevelopmental training.
    
    Implements a performance-driven reward system that:
    - Gradually reduces reward frequency and magnitude as proficiency increases
    - Introduces delayed gratification for reinforcement learning phases
    - Provides breakthrough bonuses for exceptional performance
    - Maintains minimum reward thresholds to prevent extinction
    - Implements exponential decay for reward parameters
    
    Attributes:
        reward_probability (float): Initial probability of reward delivery (0-1)
        delay_extension_coefficient (float): Rate of delay increase per success (0-1)
        breakthrough_bonus_coefficient (float): Reward multiplier for breakthroughs (0-1)
        attenuation_coefficient (float): Rate of reward parameter decay (0-1)
        min_reward_probability (float): Minimum reward probability floor (0-1)
        delay_up_limit (int): Maximum allowable delay steps
        min_reward_size (float): Minimum reward value floor

    Methods:
        tick(breakthrough): Update reward parameters based on performance
        rollback(): Not supported - rewards only move forward

    Usage Example:
        reward_progress = RewardReductionProgress(
            reward_probability=1.0,
            breakthrough_bonus_coefficient=0.2,
            attenuation_coefficient=0.1
        )
        reward_progress.tick(breakthrough=True)  # Boost rewards for breakthrough
        reward_progress.tick()  # Normal attenuation

    Relationship to System:
        - Works inversely with NoiseIncreaseProgress
        - Directly influences Mentor's reward generation
        - Integrated with TaskTrainingProgress for skill mastery tracking
    """
    def __init__(self, reward_probability = 1.4, delay_extension_coefficient = 0.1, breakthrough_bonus_coefficient = 0.5, attenuation_coefficient = 0.1,min_reward_probability = 0.1, delay_up_limit = 50, min_reward_size = 0.1):
        self.reward_probability = reward_probability 
        self.delay_extension_coefficient = delay_extension_coefficient
        self.breakthrough_bonus_coefficient = breakthrough_bonus_coefficient
        self.attenuation_coefficient = attenuation_coefficient
        self.min_reward_probability = min_reward_probability
        self.delay_up_limit = delay_up_limit
        self.min_reward_size = min_reward_size
        
        self.reward_delay = 0
        self.reward_size = 1

        pass

    def tick(self, breakthrough = False):
        if breakthrough == True:
            self.reward_size *= (1 + self.breakthrough_bonus_coefficient)
            self.reward_delay *= (1 - min(self.breakthrough_bonus_coefficient,1))
            self.reward_probability *= (1 + self.breakthrough_bonus_coefficient)
        else:
            self.reward_size = max(self.reward_size * (1 - max(self.attenuation_coefficient,0)), self.min_reward_size)
            self.reward_delay = min(self.reward_delay * (1 + self.delay_extension_coefficient), self.delay_up_limit)
            self.reward_probability = max(self.reward_probability * (1 - max(self.attenuation_coefficient,0)), self.min_reward_probability)

    def rollback(self):
        """ Reward doesn't support rollback"""
        
        pass

class SignalTiming:
    """The basic time step length of input signals, modified by noise and reward params"""
    def __init__(self, V_signal_timing = 10, A_signal_timing = 10, Lstim_signal_timing = 10, reward_signal_timing = 10):
        self.V_signal_timing = V_signal_timing
        self.A_signal_timing = A_signal_timing
        self.Lstim_signal_timing = Lstim_signal_timing
        self.reward_signal_timing = reward_signal_timing

    def tick(self):
        pass

class TaskTrainingProgress:
    """Tracks training progress and proficiency metrics for individual tasks.
    
    Attributes:
        task (Enum): The specific task being trained
        noise_increase_progress (NoiseIncreaseProgress): Noise progression tracker
        reward_reduction_progress (RewardReductionProgress): Reward progression tracker
        task_timing (SignalTiming): Temporal parameters for task signals
        continuous_failure_tolerance (int): Allowed consecutive failures before regression
        phase (int): Current training phase (0=guided, 1=reinforcement)
        record (list): History of successful progression states
        proficiency_dampening_factor (float): Controls proficiency score sensitivity
        proficiency_decay_rate (float): Rate of proficiency score decay
        conjugation_task (Enum): Related task for skill transfer
        
    Methods:
        tick(success, action_time, completion_time): Update progress metrics
        rollback(): Revert to last successful training state
        failure_rate(): Calculate current failure probability
        get_proficiency(): Calculate composite skill mastery score
    """
    def __init__(self, task, conjugation_task = None, noise_params = {}, reward_params = {},  continuous_failure_tolerance = 5, proficiency_decay_rate = 0.7, proficiency_dampening_factor  = 5, phase_transiation_proficiency_threshold = 0.5):
        self.task = task
        self.noise_increase_progress = NoiseIncreaseProgress(**noise_params)
        self.reward_reduction_progress = RewardReductionProgress(**reward_params)
        self.task_timing = SignalTiming()
        self.continuous_failure_tolerance = continuous_failure_tolerance
        self.phase_transiation_proficiency_threshold = phase_transiation_proficiency_threshold
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
        
    
        
    def tick(self, success=None, action_time=float('inf'), completion_time=float('inf')):
        if success == None:
            # The task is still in progress
            return

        # success = True # TODO: Comment out this line when deploying

        recoreded = False
        if(success == False):
            self.continuous_failure_times+=1
            if self.phase == Phase.GUIDED_LEARNING.value:
                self.guided_learning_failure_times+=1
            elif self.phase == Phase.REINFORCEMENT_LEARNING.value:
                self.reinforcement_learning_failure_times+=1
            if (self.continuous_failure_times>=self.continuous_failure_tolerance):
                self.rollback()
        else:
            self.continuous_failure_times = 0
            if self.phase == Phase.GUIDED_LEARNING.value:
                self.guided_learning_success_times+=1
            elif self.phase == Phase.REINFORCEMENT_LEARNING.value:
                self.reinforcement_learning_success_times+=1


            breakthrough = False

            # Temporarily disable the completion time comparison, comment out the following 2 lines to enable it
            completion_time = 0
            self.best_completion_time = completion_time

            if completion_time + action_time < self.best_completion_time + self.best_action_time:
                breakthrough = True
                self.best_completion_time = completion_time
                self.best_action_time = action_time

            if (breakthrough):
                self.record.append({"phase":self.phase,"noise":self.noise_increase_progress, "reward": self.reward_reduction_progress})
                recoreded = True
            self.noise_increase_progress.tick()
            self.reward_reduction_progress.tick(breakthrough=breakthrough)

            
        
        # update proficiency

        if self.phase == Phase.GUIDED_LEARNING.value:
            # Weighted impact calculation
            impact = self.proficiency_decay_rate * self.guided_learning_proficiency + (1 - self.proficiency_decay_rate) * success

            # Normalize by total counts
            total_counts = self.guided_learning_success_times + self.guided_learning_failure_times
            self.guided_learning_proficiency = (impact * self.guided_learning_success_times) / (total_counts + self.proficiency_dampening_factor)

            # Transfer to reinforcement learning phase
            if success and self.get_proficiency() > self.phase_transiation_proficiency_threshold:
                if recoreded == False: # don't duplicate record
                    self.record.append({"phase":self.phase,"noise":self.noise_increase_progress, "reward": self.reward_reduction_progress})
                self.guided_learning_proficiency *= self.proficiency_decay_rate
                self.phase = Phase.REINFORCEMENT_LEARNING.value

        elif self.phase == Phase.REINFORCEMENT_LEARNING.value:
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
    """Dynamic task selection system based on performance metrics and skill progression.
    
    Implements a hierarchical training strategy:
    1. Guided Learning Phase: Step-by-step instruction with immediate feedback
    2. Reinforcement Phase: Independent practice with delayed rewards
    3. Skill Integration: Combined task execution and transfer learning
    
    Attributes:
        task_training_progress_manager (dict): Tracks progress per task/stage
        proficiency_threshold (float): Minimum score for skill mastery
        current_stage (int): Active training complexity level
        current_task (Enum): Currently selected task
        current_task_training_progress (TaskTrainingProgress): Active task's metrics
        
    Methods:
        update_stage(): Adjust training complexity based on proficiency
        get_task_training_progress(task): Retrieve progress metrics for specific task
        choose_task_based_on_weight(task_weights): Probabilistic task selection
        tick(reset, success, action_time, completion_time): Main progression logic
    """
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
            proficiency_threshold = 0.2

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
            if any(progress.get_proficiency() < self.proficiency_threshold for (task, progress) in progress_list.items()):
                self.current_stage = stage
                break
        pass 
        
    def get_task_training_progress(self, __task):
        for stage, progress_list in self.task_training_progress_manager.items():
            for (task, progress) in progress_list.items():
                if task == __task:
                    return progress
        
    
    @staticmethod
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


    def report_status(self):
        print("####Task Planner Status:####")
        print("Current Stage:", self.current_stage)
        print("Current Task:", self.current_task)
        print("Current Task Training Progress:", self.current_task_training_progress.get_proficiency())
        print("Current Task Training Phase:", self.current_task_training_progress.phase)
        print("Training Count:", self.current_task_training_progress.guided_learning_failure_times + self.current_task_training_progress.guided_learning_success_times + self.current_task_training_progress.reinforcement_learning_failure_times + self.current_task_training_progress.reinforcement_learning_success_times)
        print("Guided LearningAccuracy:", self.current_task_training_progress.guided_learning_success_times/(self.current_task_training_progress.guided_learning_failure_times+self.current_task_training_progress.guided_learning_success_times) if (self.current_task_training_progress.guided_learning_failure_times+self.current_task_training_progress.guided_learning_success_times)!=0 else 0)
        print("Reinforcement Learning Accuracy:", self.current_task_training_progress.reinforcement_learning_success_times/(self.current_task_training_progress.reinforcement_learning_failure_times+self.current_task_training_progress.reinforcement_learning_success_times) if (self.current_task_training_progress.reinforcement_learning_failure_times+self.current_task_training_progress.reinforcement_learning_success_times)!=0 else 0)

        print("All Task Training Progresses:")
        for stage, progress_list in self.task_training_progress_manager.items():
            print(f"Stage {stage}")
            print(f"{'Task':<40} {'Proficiency':<20} {'Phase':<10} {'Guided Learning Accuracy':<25} {'Reinforcement Learning Accuracy':<30}")
            print("-" * 100)
            for (task, progress) in progress_list.items():
                guided_accuracy = progress.guided_learning_success_times / (progress.guided_learning_failure_times + progress.guided_learning_success_times) if (progress.guided_learning_failure_times + progress.guided_learning_success_times) != 0 else 0
                reinforcement_accuracy = progress.reinforcement_learning_success_times / (progress.reinforcement_learning_failure_times + progress.reinforcement_learning_success_times) if (progress.reinforcement_learning_failure_times + progress.reinforcement_learning_success_times) != 0 else 0
                total_guided_times = progress.guided_learning_success_times + progress.guided_learning_failure_times
                total_reinforcement_times = progress.reinforcement_learning_success_times + progress.reinforcement_learning_failure_times
                print(f"{task:<40} {progress.get_proficiency():<20} {progress.phase:<10} {(str(guided_accuracy)+ str('/') + str(total_guided_times)):<25} {str(reinforcement_accuracy) + str('/')+ str(total_reinforcement_times):<30}")
                print()  # Fixed indentation and added closing parenthesis

        


    def tick(self, reset = False, success = None, action_time=float('inf'), completion_time=float('inf')):
        """give feedback on the current task, and the task planner will provide new suitable task policies"""
        # if you want to reset the task, you means you didn't train for the task because there are conflicts in the objective state and start state, the planner will choose another different task and the conjugation task is preferred.

        self.report_status()

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
                    return

            
            # if any task progress (except current task) is in guided learning period and the training process has been launched (has at least one execution), choose it.
            for (task, progress) in self.task_training_progress_manager[self.current_stage].items(): 
                # progress: "TaskTrainingProgress"
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
                for (task, progress) in progress_list.items():
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
    def __init__(self, schedule:Schedule, engine_reference:'Engine'):
        self.schedule = schedule  # Reference to the Schedule
        self.current_phase = Phase.GUIDED_LEARNING  # Start with the guided learning phase
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
        task_details = self.schedule.get_task_details(task=task, phase=Phase.GUIDED_LEARNING.value)
        command_sequence = task_details[Params.COMMAND_SEQUENCES.value]
        _,last_command = list(command_sequence.items())[-1]
        final_state = last_command.get(CommandAttr.LIMB_STATE.value)
        if len(final_state) != 1:
            return False
        else:
            if [baby_limb_state] ==  final_state:
                return True
            else:
                return False




    def generate_response_sequence(self, phase, succeed, A_signal_timing, reward_probability, reward_delay, reward_size, reward_signal_timing, noise, **kwargs):
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

                    for i in range(0,int(reward_delay)):
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
    def generate_input_sequence(self, task, phase, A_signal_timing, V_signal_timing, Lstim_signal_timing, reward_size, reward_signal_timing, reward_delay, noise, **kwargs):
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
        last_cmd = list(command_sequence.items())[-1]
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
        
        if self.task_planner == None or self.task_planner.current_task == None:
            raise Exception("TaskPlanner None! Maybe the task planner is not initialized yet or all tasks have been completed. (Please refer to All Task Training Progresses)")
        return {
            "succeed": True if self.training_result == TrainingResult.SUCCESS.value else False,
            "task":self.task_planner.current_task,
            "phase":self.task_planner.current_task_training_progress.phase,
            "A_signal_timing":self.task_planner.current_task_training_progress.task_timing.A_signal_timing,
            "V_signal_timing":self.task_planner.current_task_training_progress.task_timing.V_signal_timing,
            "Lstim_signal_timing":self.task_planner.current_task_training_progress.task_timing.Lstim_signal_timing,
            "reward_size": self.task_planner.current_task_training_progress.reward_reduction_progress.reward_size,
            "reward_probability":self.task_planner.current_task_training_progress.reward_reduction_progress.reward_probability,
            "reward_signal_timing":self.task_planner.current_task_training_progress.task_timing.reward_signal_timing,
            "reward_delay" : self.task_planner.current_task_training_progress.reward_reduction_progress.reward_delay,
            "noise":self.task_planner.current_task_training_progress.noise_increase_progress.noise,

        }
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def limb_state_equal(ls,ls_expect):
        """Check if two limb states the same, based on ls_expect"""
        for i in range(0,4):

            if ls_expect[i]==ls[i] or ls_expect[i] == -1 or ls_expect[i] == 0:
                continue
            if ls[i] != ls_expect[i]:
                return False
            if ls[i] == -1 or ls[i]== 0:
                return False
            
            
        return True
            
            

    def tick(self, time, baby_limb_state):
        self.time += 1
        if self.time != time:
            raise Exception("Mentor sync failed!")
        
        # initial task
        if self.task_planner.current_task == None or self.training_status == None:
            
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
            cmd_sequence = self.generate_input_sequence(**task_params)
            self.command_queue.extend(cmd_sequence)
            self.training_status = TrainingStatus.IN_PROGRESS.value
            self.last_time_step_command = self.command_queue.pop(0)
            self.baby_limb_state_queue.append(baby_limb_state)
            return
        else:
            if (self.training_status == TrainingStatus.IN_PROGRESS.value): 
                if self.command_queue.__len__() == 0:
                    raise Exception("Invalid Command Queue")
                # Normally, if a check is not needed, just pops up the first command
                if (self.command_queue[0].get(CommandAttr.CHECK.value, False) == False):
                    if self.limb_state_consistency(self.baby_limb_state_queue[-1], baby_limb_state, self.command_queue[0].get(CommandAttr.LIMB_STATE.value,[[-1,-1,-1,-1]])[0] ):
                        self.actions_between_checks+=1
                        self.last_time_step_command = self.command_queue.pop(0)
                        self.baby_limb_state_queue.append(baby_limb_state)
                        # self.action_time+=1
                        # self.completion_time+=1
                    else:
                        # Training failure, reply NO
                        self.last_time_step_command = self.command_queue.pop(0)
                        self.training_status = TrainingStatus.IN_RESPONSE.value
                        self.training_result = TrainingResult.FAILURE.value
                        response_params = self.get_task_params()
                        self.command_queue = self.generate_response_sequence(**response_params)
                        cmd_loss = self.last_time_step_command
                        cmd_loss[CommandAttr.LIMB_STATE.value] = [self.baby_limb_state_queue[-1]]
                        cmd_loss[RewardType.item.value] = RewardType.NEGATIVE.value
                        self.baby_limb_state_queue=[baby_limb_state]
                        self.action_time = float('inf')
                        self.single_step_command_waiting = 0

                    if self.command_queue.__len__() == 0:
                        # if the popped command is the last one and the check is not needed, training success
                        self.training_status = TrainingStatus.IN_RESPONSE.value
                        self.training_result = TrainingResult.SUCCESS.value
                        response_params = self.get_task_params()
                        self.command_queue = self.generate_response_sequence(**response_params)
                        self.baby_limb_state_queue=[baby_limb_state]
                        self.single_step_command_waiting = 0
                        self.actions_between_checks = 0


                else:
                #if a check is needed, wait until the action done or time out
                    if (self.last_time_step_command.get(CommandAttr.CHECK.value, False) == False):
                        self.single_step_command_waiting_up_limit = self.actions_between_checks * len(self.command_queue[0].get(CommandAttr.LIMB_STATE.value, []))
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

                            cmd_loss = self.last_time_step_command
                            cmd_loss[CommandAttr.LIMB_STATE.value] = [self.expected_limb_state_queue[0]]
                            cmd_loss[RewardType.item.value] = RewardType.NEGATIVE.value

                            self.baby_limb_state_queue=[baby_limb_state]
                            self.action_time = float('inf')
                            self.single_step_command_waiting = 0
                            self.actions_between_checks = 0
                        else:
                            if (self.limb_state_gradient_consistency(self.baby_limb_state_queue[-1],baby_limb_state,self.expected_limb_state_queue[0])):
                                if self.limb_state_equal(baby_limb_state,self.expected_limb_state_queue[0]):
                                    # pop 
                                    goal_state = self.expected_limb_state_queue.pop(0)
                                    self.baby_limb_state_queue.append(baby_limb_state)
                                    self.single_step_command_waiting = 0
                                    if self.expected_limb_state_queue.__len__() == 0:
                                        # If all limb states in a command are achieved, pop this command
                                        self.last_time_step_command = self.command_queue.pop(0)
                                        self.single_step_command_waiting_up_limit = 0
                                        self.actions_between_checks = 0

                                        cmd_loss = self.last_time_step_command
                                        cmd_loss[CommandAttr.LIMB_STATE.value] = [goal_state]
                                        cmd_loss[RewardType.item.value] = RewardType.POSITIVE.value

                                        if self.command_queue.__len__() == 0:
                                            # if all commands are done, reply GOOD
                                            # self.last_time_step_command = self.command_queue.pop(0)
                                            self.training_status = TrainingStatus.IN_RESPONSE.value
                                            self.training_result = TrainingResult.SUCCESS.value
                                            response_params = self.get_task_params()
                                            self.command_queue = self.generate_response_sequence(**response_params)
                                            self.baby_limb_state_queue=[baby_limb_state]
                                            self.single_step_command_waiting = 0
                                            self.actions_between_checks = 0
                                        return
                                    else:
                                        # one action done, but not all limb states are achieved, continue
                                        cmd_loss = self.last_time_step_command
                                        cmd_loss[CommandAttr.LIMB_STATE.value] = [goal_state]
                                        cmd_loss[RewardType.item.value] = RewardType.POSITIVE.value
                                        return

                                else:
                                    self.single_step_command_waiting += 1
                        
                            else:
                                # Training failure, reply NO
                                self.last_time_step_command = self.command_queue.pop(0)

                                cmd_loss = self.last_time_step_command
                                cmd_loss[CommandAttr.LIMB_STATE.value] = [self.expected_limb_state_queue[0]]
                                cmd_loss[RewardType.item.value] = RewardType.POSITIVE.value
                                
                                self.training_status = TrainingStatus.IN_RESPONSE.value
                                self.training_result = TrainingResult.FAILURE.value
                                response_params = self.get_task_params()
                                self.command_queue = self.generate_response_sequence(**response_params)
                                self.baby_limb_state_queue=[baby_limb_state]
                                self.action_time = float('inf')
                                self.single_step_command_waiting = 0
                                self.actions_between_checks = 0



            elif (self.training_status == TrainingStatus.IN_RESPONSE.value):
                self.last_time_step_command = self.command_queue.pop(0)
                self.baby_limb_state_queue = [baby_limb_state]
                if self.command_queue.__len__() == 0:
                    
                    self.task_planner.tick(success=(self.training_result == TrainingResult.SUCCESS.value),action_time=self.action_time,completion_time=self.completion_time)
                    
                    while self.reset_task(task=self.task_planner.current_task,baby_limb_state=baby_limb_state):
                        self.task_planner.tick(reset=True)

                    # self.baby_limb_state_queue.clear()

                    task_params = self.get_task_params()

                    cmd_sequence = self.generate_input_sequence(**task_params)
                    self.command_queue.extend(cmd_sequence)
                    self.training_status = TrainingStatus.IN_PROGRESS.value
                    self.training_result = None
                    return
                else:
                    return
            else:
                raise Exception("Invalid Training Status")
    def get_latest_command(self):
        
        return self.time,self.last_time_step_command if self.time != 0 else {
                CommandAttr.A.value: None,
                CommandAttr.V.value: [-1,-1,-1,-1],
                CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                CommandAttr.CHECK.value: 0,
                CommandAttr.LOOP.value: 0,
                CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]], # 此LIMB_STATE为期望的LIMB_STATE，并非baby的实际limb state， 当reward type不为None, or pending时，应当根据此limb state进行反向传播
                CommandAttr.REWARD_SIZE.value: 0, 
                RewardType.item.value: None
            }


class Limb:
    def __init__(self, dead_zone = 0, gathering_decline_coefficient = 2,  pain_zone = 100, basic_pain = 5, pain_growth_coefficient = 1, pain_decline_coefficient = 5, initial_state = 2):
        """The Bionic Limb: dead_zone is the continuous"""

        self.dead_zone = dead_zone
        self.gathering_decline_coefficient = gathering_decline_coefficient
        self.pain_zone = pain_zone
        self.basic_pain = basic_pain
        self.pain_growth_coefficient = pain_growth_coefficient
        self.pain_decline_coefficient = pain_decline_coefficient
        self.state = initial_state
        self.pain_scale = [0,0] # muscles for bend, and straighten
        self.gathering_scale = 0

    def tick(self, command):
        # Bend
        if command == 1:
            if command != self.state:
                self.gathering_scale+=1 # Gathering momentum for dead_zone
                self.pain_scale[0] = max(self.pain_scale[0]-self.pain_decline_coefficient,0) # Pain decline
                if self.gathering_scale >=self.dead_zone: # Trigger state transition
                    self.state = command
                    self.gathering_scale = 0
                else:
                    self.pain_scale[0] = max(self.pain_scale[0]-self.pain_decline_coefficient,0) # Pain decline (Bonus)
            else:
                self.pain_scale[0] += 1 # Pain increase
                self.gathering_scale = max(self.gathering_scale - self.gathering_decline_coefficient * 2, 0) # Gathering momentum decline


        # Straighten
        elif command == 2:
            if command != self.state:
                self.gathering_scale+=1 # Gathering momentum for dead_zone
                self.pain_scale[1] = max(self.pain_scale[1]-self.pain_decline_coefficient,0) # Pain decline
                if self.gathering_scale >=self.dead_zone: # Trigger state transition
                    self.state = command
                    self.gathering_scale = 0
                else:
                    self.pain_scale[1] = max(self.pain_scale[1]-self.pain_decline_coefficient,0) # Pain decline (Bonus)
            else:
                self.pain_scale[1] += 1 # Pain increase
                self.gathering_scale = max(self.gathering_scale - self.gathering_decline_coefficient * 2, 0) # Gathering momentum decline (with bonus)

        # No action
        elif command == 0:
            self.pain_scale = [max(self.pain_scale[i]-self.pain_decline_coefficient,0) for i in range(2)] # Pain decline
            self.gathering_scale = max(self.gathering_scale - self.gathering_decline_coefficient, 0) # Gathering momentum decline
            ## 这里是额外增加的部分，当command为0时，回复0状态
            if self.gathering_scale <= self.dead_zone//2:
                self.state = 0
                self.gathering_scale = 0

        else:
            # command > 2, the muscles for bending and straightening are both activated
            self.pain_scale = [self.pain_scale[i] + 1 for i in range(2)] # Pain increase



from NeuralModels import Baby

        
class Engine:
    def __init__(self, mentor_reference:Mentor, baby_reference: Baby):
        self.time = 0  # Global timer
        self.state = None  # Environment state (to be defined as a class/structure)
        self.baby_control_commands:map = {}  # Current limb commands U(t) (from baby)
        self.baby_limb_state = {}  # Current limb states O(t)
        self.mentor_control_commands = {}  # Mentor commands (from mentor)
        self.internal_feedback = {0:[0,0,0,0,0,0,0,0]}  # Internal feedback signals (punishments of pain from 4 limbs * 2 for each limb)
        self.external_feedback = {0:[0,0,0,0,0,0,0,0]}  # External feedback signals (rewards/ punishments from food，8 bits)
        self.mentor_reference = mentor_reference  # Reference to the Mentor
        self.baby_reference = baby_reference # Reference to the Baby



        self.limb_fl = Limb(initial_state = 0)
        self.limb_fr = Limb(initial_state = 0)
        self.limb_bl = Limb(initial_state = 0)
        self.limb_br = Limb(initial_state = 0)

        


    def tick(self):
        """Advance the global timer and update the environment state."""
        self.update_environment_state()

        self._present_state()

        self.time += 1

        self.mentor_reference.tick(self.time, baby_limb_state=self.baby_limb_state.get(self.time - 1, []))

        # TODO: Comment out the following if-else block when deploying
        # if self.mentor_control_commands.get(self.time - 1, {}).get('limb_state', [[-1,-1,-1,-1]]).__len__() != 0:
        #     self.mentor_reference.tick(self.time, baby_limb_state=self.mentor_control_commands.get(self.time - 1, {}).get('limb_state', [[-1,-1,-1,-1]])[0])
        # else:
        #     self.mentor_reference.tick(self.time, baby_limb_state=self.baby_limb_state.get(self.time - 1, [-1,-1,-1,-1]))
         

        self.baby_reference.tick(
            self.time,
            baby_limb_state=self.baby_limb_state.get(self.time - 1, []),
            mentor_control_command=self.mentor_control_commands.get(self.time - 1, {}),
            internal_feedback=self.internal_feedback.get(self.time - 1, []),
            external_feedback=self.external_feedback.get(self.time - 1, []),
        )
    def _present_state(self):
        print(f"******State at Time: {self.time}******")
        print(f"Baby Control Commands: {self.baby_control_commands.get(self.time, [-1, -1, -1, -1])}")
        print(f"Mentor Control Commands: {self.mentor_control_commands.get(self.time, [-1, -1, -1, -1])}")
        print(f"Baby Limb State: {self.baby_limb_state.get(self.time, [-1, -1, -1, -1])}")
        print(f"Limb State: {self.limb_fl.state}, {self.limb_fr.state}, {self.limb_bl.state}, {self.limb_br.state}")
        
        print(f"Internal Feedback: {self.internal_feedback.get(self.time, [-1, -1, -1, -1, -1, -1, -1, -1])}")
        print(f"External Feedback: {self.external_feedback.get(self.time, [-1, -1, -1, -1,-1, -1, -1, -1])}")
        
        

    def update_environment_state(self):
        """Perform the environment state transfer function."""
        self.update_baby_control_commands()
        self.update_mentor_control_commands()
        self.update_baby_limb_state()
        self.update_feedback()

    def update_baby_control_commands(self):
        try:
            (time_stamp, baby_control_command) = self.baby_reference.get_latest_command()
            if time_stamp == self.time:
                self.baby_control_commands[self.time] = baby_control_command
            else:
                raise ValueError(f"Time sync error: expected {self.time}, but got {time_stamp}")
        except Exception as e:
            print(f"Error in update_baby_control_commands: {e}")
            import traceback
            print(traceback.format_exc())

    def update_mentor_control_commands(self):
        try:
            (time_stamp, mentor_control_command) = self.mentor_reference.get_latest_command()
            if time_stamp == self.time:
                self.mentor_control_commands[self.time] = mentor_control_command
            else:
                raise ValueError(f"Time sync error: expected {self.time}, but got {time_stamp}")
        except Exception as e:
            print(f"Error in update_mentor_control_commands: {e}")
            import traceback
            print(traceback.format_exc())

    def update_baby_limb_state(self):
        limb_control_command = [-1, -1, -1, -1]
        baby_limb_command = self.baby_control_commands.get(self.time, [0, 0, 0, 0])
        limb_control_command = baby_limb_command

        # # NOTE: 调试Mentor代码，令
        # mentor_limb_command = self.mentor_control_commands.get(self.time, {}).get(CommandAttr.LSTIM.value, [-1, -1, -1, -1])
        # limb_control_command = mentor_limb_command



        self.limb_fl.tick(limb_control_command[0])
        self.limb_fr.tick(limb_control_command[1])
        self.limb_bl.tick(limb_control_command[2])
        self.limb_br.tick(limb_control_command[3])

        limb_state = [
            self.limb_fl.state,
            self.limb_fr.state,
            self.limb_bl.state,
            self.limb_br.state
        ]
        self.baby_limb_state[self.time] = limb_state

    def update_feedback(self):
        _internal_feedback = [
            self.limb_fl.pain_scale,
            self.limb_fr.pain_scale,
            self.limb_bl.pain_scale,
            self.limb_br.pain_scale
        ]
        internal_feedback = [bit for pair in _internal_feedback for bit in pair]
        
        self.internal_feedback[self.time] = internal_feedback

        try:
            (time_stamp, command) = self.mentor_reference.get_latest_command()
            reward = command.get(CommandAttr.REWARD_SIZE.value, 0)
            if time_stamp == self.time:
                self.external_feedback[self.time] = [reward] * 8
            else:
                raise ValueError(f"Time sync error in external feedback: expected {self.time}, but got {time_stamp}")
        except Exception as e:
            print(f"Error in update_feedback: {e}")

    def performance_report(self):
        print("Performance Report:")
        print(f"Total steps: {self.time}")
        print(f"Internal feedback: {self.internal_feedback}")
        print(f"External feedback: {self.external_feedback}")
        print(f"Limb states: {self.baby_limb_state}")

    def run(self, steps):
        """Main loop to simulate the environment over a number of steps."""
        for _ in range(steps):
            self.tick()
        self.performance_report()

# Example instantiation and usage
if __name__ == "__main__":
    # Placeholder for Schedule and Mentor setup
    schedule = Schedule()  # Replace with actual Schedule implementation
    mentor = Mentor(schedule, None)  # Replace None with Engine reference if needed later

    command_words = {'sit', 'stand', 'lie', 'run', 'good', 'no'}
    command_chars = {char for word in command_words for char in word}
    A_cmd_char_list=sorted(list(command_chars))
    A_cmd_char_index = {char: idx for idx, char in enumerate(A_cmd_char_list)}
    
    baby_params = {
        # "A_OrganCellsNum": 26,
        # "A_bufferCellNum": 26,  # buffer cells are the cells that receive or process specific stimuli from the certain organ before sending it to/after receiving it from the brain (bunch of normal cells)
        # "A_batchSize": int(0.5 * 26),

        "V_OrganCellsNum": 8,
        "V_bufferCellNum": 0,
        "V_batchSize":  int(0.5 * 8),


        "LAct_OrganCellsNum": 4*2, # the Actuator(s)/signal output interface(s) of the baby
        "LAct_bufferCellNum": 0,    
        "LAct_batchSize": int(0.5 * 4*2),

        "LState_OrganCellsNum": 4*2,
        "LState_bufferCellNum": 0,
        "LState_batchSize": int(0.5 * 4*2),

        "LPain_OrganCellsNum": 4*2 * 0,
        "LPain_bufferCellNum": 4*2* 2 * 0,
        "LPain_batchSize": int(0.5 * 4*2)*0,

        "Reward_OrganCellsNum": 8*0,
        "Reward_bufferCellNum": 8* 2*0,
        "Reward_batchSize": int(0.5 * 8)*0,

        "centralCellsNum": 0,
        "centralCell_batchSize": int(0.5 * 16),

        "loopNum": 5,

    }

    baby = Baby(A_OrganCellsNum= A_cmd_char_index.__len__(),
                A_cmd_char_index=A_cmd_char_index,
                A_bufferCellNum= A_cmd_char_index.__len__(),
                A_batchSize= int(0.2 * A_cmd_char_index.__len__()),
                # **baby_params
                )
    # Create the engine and link it with the Mentor
    engine = Engine(mentor_reference=mentor, baby_reference=baby)
    mentor.engine_reference = engine

    # Run the simulation for a fixed number of steps      
    engine.run(steps=3000)
