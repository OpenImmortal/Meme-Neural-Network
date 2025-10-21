

from enum import Enum, member

class Phase(Enum):
    AWAKENING = 0
    GUIDED_LEARNING = 0
    REINFORCEMENT_LEARNING = 1

class Task(Enum):
    AWAKENING_TASKS = "Awakening Tasks"
    PRE_TRAINING_TASKS = "Pre-Training Tasks"
    ADVANCED_TASKS = "Advanced Tasks"
    CONJUGATION_TASK = "conjugation_task"
    
    LIMB_CONTROL_FL_BEND = "Limb Control#FL-Bend"
    LIMB_CONTROL_FR_BEND = "Limb Control#FR-Bend"
    LIMB_CONTROL_BL_BEND = "Limb Control#BL-Bend"
    LIMB_CONTROL_BR_BEND = "Limb Control#BR-Bend"
    LIMB_CONTROL_FL_STRAIGHTEN = "Limb Control#FL-Straighten"
    LIMB_CONTROL_FR_STRAIGHTEN = "Limb Control#FR-Straighten"
    LIMB_CONTROL_BL_STRAIGHTEN = "Limb Control#BL-Straighten"
    LIMB_CONTROL_BR_STRAIGHTEN = "Limb Control#BR-Straighten"

    SIT = "Sit"
    STAND = "Stand"
    LIE = "Lie"
    RUN = "Run"

class Response(Enum):
    GOOD = "Good"
    NO = "No"

class CommandAttr(Enum):
    A = "A"
    V = "V"
    LSTIM = "Lstim"
    CHECK = "check"
    LOOP = "loop"
    LIMB_STATE = "limb_state"
    REWARD_SIZE = "reward_size"

class Params(Enum):
    VARIATION_COEFFICIENT = "variation_coefficient"
    REWARD_DECAY = "reward_decay"
    DELAY_EXTENSION_COEFFICIENT = "delay_extension_coefficient"
    BREAKTHROUGH_BONUS_COEFFICIENT = "breakthrough_bonus_coefficient"
    ATTENUATION_COEFFICIENT = "attenuation_coefficient"
    REWARD_DELAY = "reward_delay"
    HOLD_DURATION = "hold_duration"
    NOISE_INCREMENT_COEFFICIENT = "noise_increment_coefficient"
    UPPER_LIMIT = "upper_limit"
    STIMULI_TIMING_DURATION = "stimuli_timing_duration"
    LETTER_TIMING_DURATION = "letter_timing_duration"
    COMMAND_SEQUENCES = "command_sequences"
    REWARD_PARAMETERS = "reward_parameters"
    INPUT_TIMING_PARAMETERS = "input_timing_parameters"

class TrainingStatus(Enum):
    IN_PROGRESS = "in progress"
    COMMAND_FINISHED = "command finished" # waiting for response
    IN_RESPONSE = "in response"
    RESPONSE_FINISHED = "response finished" # command and response are finished 

class TrainingResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"



class TaskType(Enum):
    WAKE_WHEN_REWARD = 1
    WAKE_WHEN_IN_GATE_CHANGE = 2
    WAKE = 3

    CANCEL = -1

class RoleType(Enum):
    BABY = "baby"
    CELL = "cell"
    CONNECTION = "connection"


class ConnectionDirection(Enum):
    DEFAULT = 0
    INPUT = 1
    OUTPUT = 2
    RANDOM = 0


class GrowthType(Enum):
    LINEAR = "linear"
    PROPORTIONAL = "proportional"

class RewardType(Enum):
    item = "reward_type"
    POSITIVE = "positive" # 好结果
    NEGATIVE = "negative" # 坏结果
    PENDING = "pending" # 结果目前未知
    MIXED = "mixed" # 结果有好有坏






class Transition(Enum):
    """
    The transition of a phase
    - timestamp_start：当前过程开始的时间戳
    - timestamp_end：当前过程结束的时间戳
    - progress：当前状态的进度/完成度（如开启比例）
    - rate：当前状态持续的时长
    - target：切换到下一状态所需的条件，可为进度阈值或持续时间，取决于状态类型。
    - cycles：完成当前状态的触发次数
    """
    # default transition logic

    PROGRESS = "progress"
    RATE = "rate"
    TARGET = "target"
    CYCLES = "cycles"

class LinearM(Enum):
    """
    The linear model of a pulse
    """
    K = "k" # growth rate
    B = "b" # initial value
    RATIO = "ratio" # k = ratio * b
    X = "x" # current time
    Y = "y" # current value / progress (y = k * x + b)
    ...
class LinearEvent(Enum):


    X0 = "x0" # the start time of the event
    X1 = "x1" # the end time of the event
    Y0 = "y0"
    Y1 = "y1"


class ConnectionStatus(Enum):
    """
    The status of the connection
    """
    ID = "conn_id"
    ENABLED = "is_true_connection" # the connection is enabled
    IDLE = "idle" # the connection is enabled but idle
    DRIVE = "drive" # signal type driving the connection
    # OPEN_THRESHOLD = "open_threshold" # the threshold of the connection to open
    # CLOSE_THRESHOLD = "close_threshold" # the threshold of the connection to close


    GATE_STATUS = "gate_status" # the gate of the connection
    

    GATE_OPENING_TRANSITION = "gate_opening_transition" # the transition of the gate from closed_ready to opened
    GATE_OPENED_TRANSITION = "gate_opened_transition" # the transition of the gate from opened to closed
    GATE_CLOSED_CHARGING_TRANSITION = "gate_closed_charging_transition" # the transition of the gate from closed to closed_ready (Happens at the up end)

    UPSTREAM_DELTA_SIGNAL = "upstream_delta" # the delta of the upstream cell from connections
    UPSTREAM_OVERRIDE_SIGNAL = "upstream_override" # the overridden values of the upstream cell from connections

    DOWNSTREAM_DELTA_SIGNAL = "downstream_delta" # the delta of the downstream cell from connections
    DOWNSTREAM_OVERRIDE_SIGNAL = "downstream_override" # the overridden values of the downstream cell from connections

    UP_END_SIGNAL = "up_end_signal" # the status of the up end of the connection (not cell)
    DOWN_END_SIGNAL = "down_end_signal" # the status of the down end of the connection

class GateStatus(Enum):
    """门的状态枚举
    值说明：
    - OPENING     : 门正在打开过程中
    - OPENED      : 门已完全打开
    - CLOSED_CHARGING : 门已关闭且处于充能期（不可操作）
    - CLOSED_READY   : 门已关闭且充能完毕（等待开启命令）
    """
    OPENING = "opening"
    OPENED = "opened"
    CLOSED_CHARGING = "closed_charging"
    CLOSED_READY = "closed_ready"
    OTHER = "other"




        
class CellStatus(Enum):
    """
    The status of the cell
    """
    ID = "id" # the id of the cell
    ORGAN = "organ" # the cell is an organ cell

class ParamWindow(Enum):
    WEIGHTS = "weights"
    BIAS = "bias"
    MODULATION = "modulation"
    P = "p"
    Q = "q"
    FINGERPRINT = "fingerprint"

class Topology(Enum):
    item = "topology"
    WEIGHT = "weight"
    INPUTS = "inputs"
    OUTPUTS = "outputs"


class Event(Enum):
    ID= "event_id"
    CONN_ID = "conn_id"
    IS_RECEPTOR_INPUT_EVENT = "is_receptor_trigger_event"
    IS_ACTUATOR_OUTPUT_EVENT = "is_actuator_output_event" # 两种外部事件锁定progress 2.
    SIGNAL = "signal" # the signal name of y value


    X0 = "X0" # the start time of the event
    X1 = "X1" # the end time of the event
    START_TIMESTAMP = "X0"
    END_TIMESTAMP = "X1"
    PROGRESS = "progress"
    FINISHED = "finished"

    LINK = "link"


    # Sub-Events for Triggering
    @member
    class Gate_Opening_Transition(Enum): # Linear
        item = "gate_opening_transition" # the transition of the gate from closed_ready to opened
        PROGRESS = 0
        X0 = "x0" # the start time of the event
        X1 = "x1" # the end time of the event
        Y0 = "y0"
        Y1 = "y1"


        
    @member
    class Gate_Opened_Transition(Enum): # Linear
        item = "gate_opened_transition" # the transition of the gate from opened to closed
        PROGRESS = 1
        
        X0 = "x0" # the start time of the event
        X1 = "x1" # the end time of the event
        Y0 = "y0"
        Y1 = "y1"
    @member
    class Down_End_Attenuation(Enum): # Linear
        item = "down_end_attenuation"
        PROGRESS = 2
        X0 = "x0" # the start time of the event
        X1 = "x1" # the end time of the event
        Y0 = "y0"
        Y1 = "y1"

    
    


    

    

class Link(Enum):
    """
    The link of the chain of events based on the connection, which is stored in the connection object by the downstream cell.
    """

    EVENT = "event" # the event object
    
    @member
    class Condition(Enum):
        item = "condition"
        Y0 = "y0"
        ACTIVATION_STRENGTHS = "activation_strengths" # the dict of weighted trigger signal strength. (conn_id - w*signal)
        FATIGUE_SELF = "fatigue_self" # 自身Conn的疲劳
        FATIGUE_OTHERS = "fatigue_others" # 除自身外，同一节点其他Conn的疲劳（均值）
        BRANCH = "trg_branch" # 此链接的ParamGroup的branch的引用（对应此link）


    
    @member
    class PathRole(Enum):
        """
        The path role of the connection when it is triggered
        - START: the connection is the first connection in the path, triggered by no other connection, for example, triggered by the organ cells
        - END: the connection is triggered by other connections, and it is the last connection in the path
        - MID: the connection is triggered by other connections, and it is not the first or last connection in the path
        """
        item = "pathrole"
    
        START = "start"
        END = "end"
        MID = "mid"
    @member
    class Trigger(Enum):
        """
        How the connection is triggered
        - EVENTS: the connection is triggered by these events
        """
        item = "trigger"
        EVENTS = "trigger_events"  # [Event]


    @member
    class Opposite(Enum):
        """
        The opposite event that blocks the sequence happening
        """
        item = "opposite"
        EVENTS = "opposite_events"

    @member
    class Sequence(Enum):
        """
        How the sequence of the connection is triggered by this connection
        - EVENTS: the trigger of this connection directly triggers the sequence events
        """
        item = "sequence"
        EVENTS = "sequence_events" 


class Request(Enum):
    # 发起一个事件的请求
    class RequestType(Enum):
        item = "type"
        CHAIN_EVENT = "chain_event"
        PROPAGATION = "propagation"
        BREAK_CONN = "break_conn"
        

    


    # common args
    INITIATOR_CONN_ID = "init_conn_id"

    # chain events and propagation args
    INITIATOR_EVENT_ID = "init_event_id"
    TEMPORAL_CONTEXTUAL_VIEW = "temporal_contextual_view"

    # Chain events args
    TRIGGER_SIGNALS = "trig_signals" # the signals types that trigger the INITIATOR event
    SEQUENCE_SIGNALS = "seq_signals"
    TRIGGER_EVENTS = "trig_events" # [events]


    # Propagation args
    REWARD_TYPE = "reward_type"
    



class Signal_E(Enum): 
    '''
    E is the basic signal driving the connection and reflecting the activity of the cell （连接传递）
    '''
    E = "e"
    @member
    class Trace(Enum):
        '''
        Influence of using E
        '''
        WASTE = "e_waste"
        FATIGUE = "e_fatigue"
    @member
    class Constraint(Enum):
        '''
        Constraint of using E
        '''
        FIRE_COST = "e_fire_cost"
        STOCK = "e_stock" # stock of e
        MAX_STOCK = "e_max_stock" # max stock of e
        ARREAR = "e_arrear" # arrear of e
        MAX_ARREAR = "e_max_arrear" # max arrear of e

class Signal_G(Enum):
    '''
    G调节信号，促进生成新的链接 （细胞间传递 + 连接传递 + 全局）
    '''
    G = "g"
    

   
        

    
class Signal_F(Enum):
    '''
    F is the signal countering the effect of E, can be regarded as negative E （连接传递）
    '''
    F = "f"
    @member
    class Trace(Enum):
        '''
        Influence of using F
        '''
        WASTE = "f_waste"
        FATIGUE = "f_fatigue"
    @member
    class Constraint(Enum):
        '''
        Constraint of using F
        '''
        FIRE_COST = "f_fire_cost"
        stock = "f_stock" # stock of f
        MAX_STOCK = "f_max_stock" # max stock of f
        ARREAR = "f_arrear" # arrear of f
        MAX_ARREAR = "f_max_arrear" # max arrear of f    

class Signal_R(Enum):
    '''
    R是调节信号，代表奖励，（过量）修复疲劳，健壮连接，补充消耗的E、F。但细胞会优先补足自己应得的奖励，再传递给周围细胞。（细胞间传递 + 链接传递 + 事件链传递），与P竞争，会破坏竞争处的链接
    '''
    R = "r"
    @member
    class Trace(Enum):
        '''
        Influence of using R
        '''
        WASTE = "r_waste"

class Signal_P(Enum):
    '''
    P是调节信号，代表惩罚，会将疲劳与浪费带来的损伤固化。但每个细胞都会趋利避害，只承担自己应得的惩罚。（细胞间传递 + 连接传递 + 事件链传递）与R竞争，会破坏竞争处的链接
    '''
    P = "p"
