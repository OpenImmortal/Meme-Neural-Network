from __future__ import annotations
from typing import Dict, List, Any, Tuple, TYPE_CHECKING
from enum import Enum
import random

# from ConstantEnums import RoleType, TaskType, TrainingResult, TrainingStatus, Params, CommandAttr, Response, ConnectionDirection, CellStatus, ConnectionStatus
from ConstantEnums import *
from Message import Message
from EventTable import EventTable

from BiMultiMap import BiMultiMap

from ParamGroup import *


"""
ConnectionPath.py

Defines two classes:

1) ConnectionPathSnapshot:
   - A data container for path parameters at a specific timestamp.
   - Tracks path length, time since the path began receiving signals, 
     and time since both upstream and downstream gates are open.

2) ConnectionPath:
   - Maintains two snapshots (double-buffering) for read/write via timestamps.
   - Expects p, q, and other state to come from the upstream (and downstream) Cell snapshot.
   - Stores timeElapsedUpstreamOpen, timeElapsedBothOpen, which can be incremented
     if certain gating conditions are met.

Usage (Example Pseudocode):
    from ConnectionPath import ConnectionPath

    # Suppose we have a path from cellA to cellB
    path = ConnectionPath(connection_path_id=123, path_length=5.0)

    # At tick t=1, prepare the path to store new data
    path.prepareForTick(newTimestamp=1)

    # (Some code obtains upstreamCellSnapshot, gates, etc.)
    # path.transformSignal(upstreamCellSnapshot, gateUpstreamSnapshot, gateDownstreamSnapshot, dt=1.0)

    # Then fill the snapshot
    path.fillSnapshot(timestamp=1)

    # If we want to read last tick's data:
    old_path_snap = path.getSnapshot(timestamp=0)  # might raise error if not set up
"""


"""
Connection.py

Defines two classes:

1) ConnectionSnapshot:
   - A data container for storing the operational/regulatory flows 
     and signals at a specific timestamp, such as:
       - p_prop_flow_internal, p_intg_flow_internal
       - q_prop_flow_internal, q_intg_flow_internal
       - p_prop_flow_external, p_intg_flow_external
       - q_prop_flow_external, q_intg_flow_external
     (Extend as needed.)

2) Connection:
   - Maintains references to:
       - Upstream cell (ID)
       - Downstream cell (ID)
       - ConnectionGate objects (upstreamGate, downstreamGate)
       - ConnectionPath object
   - Uses double-buffer snapshots to record computed flows each timestamp.
   - Provides methods:
       prepareForTick(t)
       fillSnapshot(t)
       getSnapshot(t)
       computeFlowsAndSignals(t, dt, upstreamCellSnapshot, downstreamCellSnapshot, gateUpSnap, gateDownSnap, pathSnap)
         - calculates p/q flows, etc.
   - The tick(...) method orchestrates the gate updates, path updates, 
     then calls computeFlowsAndSignals(...) and finalizes the snapshot.

Integration Notes:
 - This class does not handle domain logic (max 20 neighbors, etc.), 
   which is managed by Baby.py.
 - The `is_true_connection` boolean can differentiate active/“false” connections.
 - Placeholder code remains for partial or advanced usage (like gating partial flows).
"""

import random
from typing import Optional, List, Dict, Set

# from __future__ import annotations
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     # 2) These imports only happen at "type-check time", not at runtime
#     from ConstantEnums import TaskType, RoleType
#     from Message import Message

#     # In a real codebase, you'd do:
#     from Baby import Baby
#     from Cell import Cell, CellSnapshot
#     from ConnectionGate import ConnectionGate, ConnectionGateSnapshot
#     from ConnectionPath import ConnectionPath, ConnectionPathSnapshot

class ConnectionSnapshot:
    """
    A container for storing connection-level computed flows at a specific timestamp.
    """

    def __init__(self):
        self.connectionID = 0  # or self.connectionID: int

        self.status: dict = {}

        self.constraints: dict = {}


    def clear(self):
        """
        Reset all fields to default values for the new tick snapshot.
        """
        self.connectionID = 0

        self.status: dict = {}

        self.constraints: dict = {}

    def __repr__(self):
        return (
            "ConnectionSnapshot("
            f"connectionID={self.connectionID}, "
            f"status={self.status}, "
            f"constraints={self.constraints}"
            ")"
        )



class Connection:
    """
    Represents a link between two cells, with:
    
    """

    _connection_id_counter = 1
    def __init__(
        self,
        is_true_connection : bool = True,
        upstream_cell_id: int = None,
        downstream_cell_id: int = None,
        initial_status : dict = {
            ConnectionStatus.GATE_STATUS.value:GateStatus.CLOSED_READY.value,

        },
        initial_constraints : dict = {
            'signal_type':Signal_E.E.value,
            'ts': 5,
            'k': 1,
            'b':3,

        },
        logic_rules: callable = None,
        parent_baby: "Baby" = None,
        trainable:bool = True,
        isReceptor:bool = False,
        isActuator:bool = False,

    ):
        """
        Args:
            connection_id (str): Unique ID for this connection.
            upstream_cell_id (str): The ID of the upstream cell. If the connection is not False, this only indicates the cell is at the random side of the connection (You need to certify the direction when enabling the connection).
            downstream_cell_id (str): The ID of the downstream cell.
        """
        Connection._connection_id_counter+=1
        self._connection_id = Connection._connection_id_counter

        self.baby = parent_baby

        # 
        self.upstream_cell_id = upstream_cell_id
        self.downstream_cell_id = downstream_cell_id


        self.status = copy.deepcopy(initial_status)
        self.trainable = trainable
        self.isReceptor = isReceptor # 允许无输入链接
        self.isActuator = isActuator # 允许无输出链接
        self.status["is_true_connection"] = is_true_connection
        self.status[ConnectionStatus.ID.value] = self._connection_id
        if self.isReceptor or self.isActuator:
            self.status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OTHER.value
        

        
        self.constraints = initial_constraints
        from LogicRules import connection_logic_rules
        self.logic_rules = logic_rules if logic_rules else connection_logic_rules




        


        



        # Double-buffered snapshots
        self._snapshots = [ConnectionSnapshot(), ConnectionSnapshot()]
        self._validTimestamps: Set[int] = set()
        self._currentTimestamp = 0

        # Indicate if the connection is idle (no flow)
        self.isIdle = 0
        self.idleLock = False # Lock idle state

        self.records: "EventTable" = EventTable()


        self.propagation_manager: PropagationManager= PropagationManager(baby=self.baby, parent= self, conn_id= self._connection_id)

        self.inbox:List[Message] = []       # List[Message]
        self.outbox:List[Message] = []      # List[Message]
        self.task_queue:List[Message] = []  # List[TaskType] or more complex items

    def random_idle(self):
        

    #     # unlock the idle state if both gates are open (One valid inter-cell communication)
    #     if self.gate_upstream.gate_state == 1 and self.gate_downstream.gate_state == 1:
    #         self.idleLock = False
    #     else:
    #         if self.idleLock == False:
    #             # Ramdomly choose a number between 0 and 1, if the number is bigger than alpha, the connection is idle
    #             self.isIdle = random.random() > self.alpha
    #             self.idleLock = True # Lock the idle state until it is waken up
    #             if self.isIdle:
    #                 self.add_outbox_message(recipient=self.upstream_cell_id, task_type=TaskType.WAKE_WHEN_REWARD, content=None)
    #                 self.add_outbox_message(recipient=self.downstream_cell_id, task_type=TaskType.WAKE_WHEN_REWARD, content=None)
        pass

    def process_inbox(self):
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.task_type == TaskType.WAKE.value:
                self.isIdle = max(self.isIdle - 1, 0)
            elif msg.task_type == TaskType.CANCEL.value:
                # Cancel the task published by the sender
                for idx, task in enumerate(self.task_queue):
                    if task.sender_type == msg.sender_type and task.sender == msg.sender:
                        self.task_queue.pop(idx)
            else:
                self.task_queue.append(msg)


    def process_tasks(self):
        """
        Go through 'task_queue' and respond accordingly.
        Example triggers for demonstration:
         - WAKE_WHEN_REWARD: if content says we got a reward, set self.awake=True.
         - WAKE: a direct command to wake up.
        """
        for msg in self.task_queue:
            if msg.task_type == TaskType.WAKE_WHEN_REWARD.value:
                raise NotImplementedError("WAKE_WHEN_REWARD not implemented for Connection.")
               

            elif msg.task_type == TaskType.WAKE_WHEN_IN_GATE_CHANGE.value:
                # if (self.gate_upstream.gate_state != self.gate_upstream.getSnapshot(self._currentTimestamp-1).gate_state ==0 ):
                    self.add_reply(msg, task_type=TaskType.WAKE, content=None)
                    msg.repeat -= 1
            elif msg.task_type == TaskType.WAKE.value:
                self.isIdle = 0
                msg.repeat -= 1
       # Remove any tasks that have been completed (repeat<0)
        self.task_queue = [msg for msg in self.task_queue if msg.repeat >= 0]

    def add_outbox_message(self, recipient, task_type,repeat = 0, content=None):
        """
        Create a new Message and add it to our outbox.
        """
        msg = Message(sender=self._connection_id,sender_type= RoleType.CONNECTION.value, recipient=recipient,recipient_type=RoleType.CELL.value, task_type=task_type,repeat=repeat, content=content)
        self.outbox.append(msg)


    def add_reply(self, original_message:Message, task_type, content=None):
        """
        Create a new Message as a reply to the original_message.
        """
        msg = original_message.reply(task_type=task_type, content=content)
        self.outbox.append(msg)




    def get_conn_id(self) -> int:
        """Read-only property for the connection's unique ID."""
        return self._connection_id

    def prepareForTick(self, newTimestamp: int):
        """
        Prepare a snapshot slot for the newTimestamp by clearing it.
        Called by the orchestrator (Baby or similar).
        """
        idx = newTimestamp % 2
        self._snapshots[idx].clear()
        self._validTimestamps.add(newTimestamp)
        self._currentTimestamp = newTimestamp

    def fillSnapshot(self, timestamp: int):
        """
        Finalize the snapshot for 'timestamp' by copying any "live" fields
        into the snapshot if desired.

        Raise error if the snapshot wasn't prepared.
        """
        if timestamp not in self._validTimestamps:
            self._validTimestamps = {x for x in self._validTimestamps if x < timestamp - 1}
            self._validTimestamps.add(timestamp)
        idx = timestamp % 2
        snap = self._snapshots[idx]

        snap.connectionID = self._connection_id


        # Copy "live" flow stats to snapshot

        snap.status = copy.deepcopy(self.status)
        snap.constraints = copy.deepcopy(self.constraints)      

    def getSnapshot(self, timestamp: int) -> ConnectionSnapshot:
        """
        Retrieve the snapshot for the given timestamp, or raise an error
        if not found. Enforces barrier approach.
        """

        return self._snapshots[timestamp % 2] if timestamp in self._validTimestamps else None
    def isWorking(self):
        """
        Returns True if the connection is active and working.
        """
        if self.is_true_connection == False or self.isIdle == True:
            return True
        else:
            return False

    def check_robustness(self,ts = 0, k = 0.01, b = 0.01) -> bool:
        """
        检查链接的健壮性。如果其任意约束触碰底线，则不健壮。需要被断开释放资源
        """
        if self.constraints["k"] <=k or self.constraints["ts"]<=ts or self.constraints["b"] <= b:
            return False
        else:
            return True



    def tick(
        self,
        timestamp: int,
        babyObj: "Baby",
        dt: float = 1,
        links:list[dict] = [], # trigger links
        **kwargs
    ):
        """
        Called once per tick by the orchestrator. 
        Steps:
         1) If not prepared, do prepareForTick().
         2) gate_upstream.tick(timestamp, self, babyObj)
         3) gate_downstream.tick(timestamp, self, babyObj)
         4) path.tick(timestamp, gateUpSnap, gateDownSnap, dt)
         5) read cell snapshots from babyObj
         6) computeFlowsAndSignals(...)
         7) fillSnapshot
        """
        if self.status["is_true_connection"] == False:
            return
        
        if timestamp not in self._validTimestamps:
            self.prepareForTick(timestamp)

        # 5) read cell snapshots from babyObj
        upCell = babyObj.cells[self.upstream_cell_id] if not self.isReceptor else None # "Cell"
        downCell = babyObj.cells[self.downstream_cell_id] if not self.isActuator else None
        if upCell:
            last_upCell_status = getattr(upCell.getSnapshot(timestamp - 1),"status",None)
            if not last_upCell_status:
                last_upCell_status = upCell.status
        else:
            last_upCell_status = None

        if downCell:
            last_downCell_status = getattr(downCell.getSnapshot(timestamp - 1),"status",None)
            if not last_downCell_status:
                last_downCell_status = downCell.status
        else:
            last_downCell_status = None
        last_status = self.getSnapshot(timestamp - 1).status if self.getSnapshot(timestamp - 1) else None


        # 6) apply logic rules
        logic_rule_kwargs = {**kwargs}
        if self.isReceptor:
            if downCell.receptor_input_signal:
                logic_rule_kwargs["receptor_input_signal"] = copy.copy(downCell.receptor_input_signal)
            else:
                return None

        if self.isActuator:
            if upCell.actuator_output_signal:
                logic_rule_kwargs["actuator_output_signal"] = copy.copy(upCell.actuator_output_signal)
            else:
                return None
            
        resdict:dict = self.logic_rules(timestamp = timestamp, conn_id= self.get_conn_id(), propagation_manager = self.propagation_manager,status = self.status, links = links, constraints = self.constraints, last_status = last_status, last_upCell_status = last_upCell_status, last_downCell_status = last_downCell_status,eventTable= self.records,baby = babyObj, **logic_rule_kwargs)

        # 7) check messages, process inbox, process tasks, and random_idle using current state and former state
        self.process_inbox()
        self.process_tasks()
        self.random_idle()

        # 8) fill the snapshot with current state
        self.fillSnapshot(timestamp)

        return resdict

    def __repr__(self):
        return (
            f"Connection(id={self.get_conn_id()}, "
            f"upstream_cell={self.upstream_cell_id}, "
            f"downstream_cell={self.downstream_cell_id}, "
            f"currentTimestamp={self._currentTimestamp}, "
            f"is_true_connection={self.is_true_connection})"
        )


"""
Cell.py

Updated to work seamlessly with the improved partial domain logic in Baby.py.
No direct domain-saturating logic is inside Cell anymore; Baby handles
which connections (true or false) are assigned at creation and ensures
we don't exceed the partial domain fraction. This Cell code focuses on:

  - Double-buffer snapshots (CellSnapshot).
  - Funds (p, q, external vs. internal).
  - The ledger of debts owed and expected returns.
  - Probability-based logic for division, apoptosis, connection creation,
    and disconnection, with requests sent to Baby.
  - A tick() function that handles per-tick updates.

In short, Cell doesn't worry about how many connections it can form initially
(beyond basic reference to connectionsIn / connectionsOut). 'Baby' manages
partial domain assignment or expansions.
"""

import random
from typing import List, Tuple, Dict, Any
class CellSnapshot:
    """
    A data container representing the cell's state at a specific time.
    Includes:
      - p (float): working capital
      - q (float): working debt
      - e (float): operational potential (electrons)
      - f (float): operational potential (positrons)
      - alpha (float): creditworthiness

      - q_external (float): portion of q from external sources (like externalDebt).
      - internalDebt (float): portion of q generated internally.
      - p_external (float): portion of p from external environment (like externalFunds).
      - internalFunds (float): portion of p generated internally.

      - ledger (dict): record of debts this cell owes to others: { otherCellID: debtAmount }
      - expectedReturns (dict): amounts other cells owe this cell: { otherCellID: returnAmount }
    """

    def __init__(self):

        self.cellID = 0
        self.status = None
        self.constraints = None


    def clear(self):
        """Reset fields to defaults each time we prepare a new snapshot."""
        self.cellID = 0
        self.status = None
        self.constraints = None

    def __repr__(self):
        return (
            f"CellSnapshot(cellID={self.cellID}, "
            f"status={self.status}, "
            f"constraints={self.constraints}, "
            f")"
        )


import LogicRules
class Cell:
    """
    Represents a neuron-like cell in the network.

    Responsibilities:
      - Maintains double-buffered snapshots (CellSnapshot) for barrier-based updates.
      - Tracks p, q, e, alpha, external vs. internal funds/debt, and ledgers for
        mutual debts or expected returns from other cells.
      - Probability-based logic for self-division, apoptosis, connection creation/disconnection.
      - A tick() method for per-step updates, receiving connection snapshots from Baby.

    'Baby' handles partial domain assignment and actual connection creation logic,
    so we don't saturate domain at cell creation. The cell itself only references
    connections via `connectionsIn` and `connectionsOut`.
    """
    cellID_counter = 1
    def __init__(
        self,
        parentBaby: Baby = None,
        canDivide: bool = True,
        canDie: bool = True,
        isOrganCell: bool = False,
        isReceptor: bool = False,
        isActuator: bool = False,
        initial_status: dict = {},
        initial_constraints: dict = {
            'signal_type':Signal_E.E.value,
        },
        max_conn:int = 10,
        logic_rules: callable = LogicRules.cell_logic_rules,
        chain_events: callable = LogicRules.chain_events
        
    ):
        """
        Args:
            cellID (int): Unique identifier for this cell.
            canDivide (bool): Whether the cell can spontaneously divide.
            canDie (bool): Whether the cell can undergo apoptosis.
            isOrganCell (bool): If True, indicates special organ cell (cannot die or divide).
            parentBaby (Baby): Reference to the Baby orchestrator for structural changes.
        """


        self.cellID = Cell.cellID_counter
        Cell.cellID_counter += 1
        self.canDivide = canDivide
        self.canDie = canDie
        if isOrganCell:
            self.canDivide = False
            self.canDie = False
        self.isOrganCell = isOrganCell
        self.isReceptor = isReceptor
        self.isActuator = isActuator

        self.parentBaby:"Baby" = parentBaby

        self.status = copy.copy(initial_status)
        self.constraints = copy.copy(initial_constraints)
        self.logic_rules = logic_rules
        self.chain_events = chain_events
        self.status[CellStatus.ID.value] = self.cellID
        self.status[CellStatus.ORGAN.value] = self.isOrganCell

        
        # Connection references
        self.connectionsIn: List[int] = []
        self.connectionsOut: List[int] = []

        self.inbox:List[Message] = []       # List[Message]
        self.outbox:List[Message] = []      # List[Message]
        self.task_queue:List[Message] = []  # List[TaskType] or more complex items


        # Indicate if the connection is idle (no flow)
        self.isIdle = 0
        self.idleLock = False # Lock idle state

        # Double-buffer snapshots
        self._snapshots = [CellSnapshot(), CellSnapshot()]
        self._validTimestamps = set()
        self._currentTimestamp = 0

        self.records = {}
        self.max_conn = max_conn
    
    def get_cell_id(self):
        return self.cellID
    

    def add_connection(self, conn:Connection):
        """
        根据connection的上下游cell，更新对应cell中的输入输出conn列表。此函数不会对Conn进行修改
        """
        if conn.upstream_cell_id == self.cellID and conn.get_conn_id() not in self.connectionsOut:
            self.connectionsOut.append(conn.get_conn_id())

        elif conn.downstream_cell_id == self.cellID and conn.get_conn_id() not in self.connectionsIn:
            self.connectionsIn.append(conn.get_conn_id())

    def remove_connection(self, conn:Connection):
        """
        根据connection的上下游cell，更新对应cell中的输入输出conn列表。此函数不会对Conn进行修改
        """
        if conn.upstream_cell_id == self.cellID:
            for i in range(len(self.connectionsOut)-1, -1, -1):
                if self.connectionsOut[i] == conn.get_conn_id():
                    del self.connectionsOut[i]

        elif conn.downstream_cell_id == self.cellID:
            for i in range(len(self.connectionsIn)-1, -1, -1):
                if self.connectionsIn[i] == conn.get_conn_id():
                    del self.connectionsIn[i]



    def has_in_neighbor(self,up_cell_id):
        return any(self.parentBaby.connections[conn_id].upstream_cell_id == up_cell_id for conn_id in self.connectionsIn)
    
    def has_out_neighbor(self, down_cell_id):
        return any(self.parentBaby.connections[conn_id].upstream_cell_id == down_cell_id for conn_id in self.connectionsOut)

    def has_out_neighbor(self,cell_id):
        return cell_id

    def getConnectionNumber(self):
        return len(self.connectionsIn) + len(self.connectionsOut) + self.isOrganCell # organ cell interacts with outter world, so it has one more connection
    
    def get_in_degree(self):
        return self.connectionsIn.__len__()
    
    def get_out_degree(self):
        return self.connectionsOut.__len__()
    
    def get_degree(self):
        return self.getConnectionNumber()

    def get_max_degree(self):
        return self.max_conn

    def random_idle(self):
        

        # unlock the idle state if the cell is not in isolation
        if self.inInsolation() == False:
            self.idleLock = False
        else:
            if self.idleLock == False:
                # Ramdomly choose a number between 0 and 1, if the number is bigger than alpha, the connection is idle
                self.isIdle = (random.random() > self.alpha) * 2 # if the cell is idle, it will ignore the next external stimuli, but will still react to the 2rd stimuli
                self.idleLock = True # Lock the idle state until it is waken up
                if self.isIdle:
                    # Send a message to all the Connections that connecting the upstream cell
                    for conn in self.connectionsIn:
                        self.add_outbox_message(recipient=conn.get_conn_id(), task_type=TaskType.WAKE_WHEN_IN_GATE_CHANGE,repeat=1, content=None)
                    
                    


    def process_inbox(self):
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.task_type == TaskType.WAKE.value:
                self.isIdle = max(self.isIdle - 1, 0)
            elif msg.task_type == TaskType.CANCEL.value:
                # Cancel the task published by the sender
                for idx, task in enumerate(self.task_queue):
                    if task.sender_type == msg.sender_type and task.sender == msg.sender:
                        self.task_queue.pop(idx)
            else:
                self.task_queue.append(msg)


    def process_tasks(self):
        """
        Go through 'task_queue' and respond accordingly.
        Example triggers for demonstration:
         - WAKE_WHEN_REWARD: if content says we got a reward, set self.awake=True.
         - WAKE: a direct command to wake up.
        """
        for msg in self.task_queue:
            if msg.task_type == TaskType.WAKE_WHEN_REWARD.value:
                # if self.p_prop_working_external > 0:
                    msg.reply(task_type=TaskType.WAKE, content=None)
                    msg.repeat -= 1

            elif msg.task_type == TaskType.WAKE_WHEN_IN_GATE_CHANGE.value:
                raise NotImplementedError("WAKE_WHEN_IN_GATE_CHANGE not implemented for Cell.")
                # if (self.gate_upstream.gate_state != self.gate_upstream.getSnapshot(self._currentTimestamp-1).gate_state ==0 ):
                #     self.add_reply(msg, task_type=TaskType.WAKE, content=None)
                #     msg.repeat -= 1
            elif msg.task_type == TaskType.WAKE.value:
                self.isIdle -= 1
                msg.repeat -= 1

        self.isIdle = max(self.isIdle, 0)
        # Remove any tasks that have been completed (repeat<0)
        self.task_queue = [msg for msg in self.task_queue if msg.repeat >= 0]

    def add_outbox_message(self, recipient, task_type,repeat = 0, content=None):
        """
        Create a new Message and add it to our outbox. (from cell to connection)
        """
        msg = Message(sender=self.cellID,sender_type= RoleType.CELL.value, recipient=recipient,recipient_type=RoleType.CONNECTION.value, task_type=task_type,repeat=repeat, content=content)
        self.outbox.append(msg)


    def add_reply(self, original_message:Message, task_type,repeat = 0, content=None):
        """
        Create a new Message as a reply to the original_message.
        """
        msg = original_message.reply(task_type=task_type, repeat = repeat, content=content)
        self.outbox.append(msg)


    def receptor_input(self, timestamp:int,value:float,signal_type:str = "E"):
        """
        感受器信号，激活一个内置的不可训练conn，用以激活后续可训练conn。当且仅当该细胞是感受器时有效,value必须为正
        """
        if self.isOrganCell and self.isReceptor:
            self.receptor_input_signal = {"timestamp":timestamp,"value":value,"signal_type":signal_type}

    def actuator_output(self, timestamp:int,validation_value:Optional[int],signal_type:str = "E", reward_type = None)->bool:
        """
        执行器输出信号，激活一个内置的不可训练conn，用以承接输入conn的信号，只进行布尔值输出，loss永远为0。当且仅当该细胞是执行器时有效. validation_value只与实际输出信号进行布尔值比较（必须具有相同符号。0为负号）
        @return 验证结果。为true说明结果一致，positive
        """
        if self.isOrganCell and self.isActuator:
            self.actuator_output_signal = {"timestamp":timestamp,"validation_value":validation_value,"signal_type":signal_type, "reward_type":reward_type}


    def _get_recent_links(self, max_len = 4)-> list[dict]:
        """
        获取最近触发的一定数目的outConn对应的event。按照触发时间由近（小索引）到远（大索引）排列
        """
        links:list[dict] = []
        for conn_id in self.connectionsOut:
            connObj = self.parentBaby.getConnection(connID=conn_id)
            events = connObj.records.get_recent_events(n=max_len)
            links.extend([event[Event.LINK.value] for event in events])

        links.sort(key=lambda x: x[Link.EVENT.value][Event.START_TIMESTAMP.value],reverse=True)
        return links[:max_len]


    # --------------------------------------------------------
    # The main TICK method
    # --------------------------------------------------------
    def tick(
        self,
        timestamp: int,
        # dt: float,
        # connectionSnapshots: List[Tuple[int, Any]],  # e.g. (connID, connSnap)
        **kwargs
    ):
        """
        Orchestrates this cell's per-tick update:
          1) Prepare a new snapshot if not already done.
          2) Update funds/debts using the provided connectionSnapshots.
          3) Apply cell logic (division, apoptosis, etc.).
          4) Fill the snapshot with final results.

        Args:
            timestamp (int): Current tick index.
            dt (float): Time delta for partial updates if needed.
            connectionSnapshots (list): A list of (connectionID, connectionSnapshot)
                                        relevant to this cell.
        """

        cellStepper:"StepperIterator" = self.parentBaby.cellStepper

        # Step 1: Prepare snapshot if needed
        if timestamp not in self._validTimestamps:
            self.prepareForTick(timestamp)


        requests = []

        # Step 2: tick out connections
        trigger_events = [] # 只采用当前timestamp上一时间戳的links作为判断依据
        for connID in self.connectionsIn:
            conn = self.parentBaby.getConnection(connID)
            active_events:list[dict] = conn.records.search_events(start_max= timestamp-1, progress_min=Event.Down_End_Attenuation.value.PROGRESS.value,progress_max=Event.Down_End_Attenuation.value.PROGRESS.value+1)
            ending_actives = conn.records.search_events(end_min=timestamp,end_max=timestamp+1) # 当前timestamp结束的事件在上一时刻也是活跃的 
            active_events.extend(ending_actives)

            if len(active_events)>1:
                raise ValueError("Signal overlapping!")
            elif len(active_events)==1:
                last_event = active_events[-1]
                trigger_events.append(last_event)


        if trigger_events:
            links = self._get_recent_links(max_len = 4)
            link_0 = {}
            link_0[Link.Trigger.value.item.value] = {}
            link_0[Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value] = trigger_events
            links[:0] = [link_0]
        else:
            links = []


        # 为了保证一致性，除了感受器可以同时tick输入conn和输出conn外，其他细胞只能tick 输出 conn。
        if self.isOrganCell and self.isReceptor:
            for connID in self.connectionsIn:
                conn:Connection = self.parentBaby.getConnection(connID)
                resdict = conn.tick(timestamp = timestamp, babyObj = self.parentBaby, links= links)
                
                if resdict:
                    requests.extend(resdict["requests"])

            for connID in self.connectionsOut:
                conn:Connection = self.parentBaby.getConnection(connID)
                if conn.check_robustness():
                    resdict = conn.tick(timestamp = timestamp, babyObj = self.parentBaby, links= links)
                    
                    if resdict:
                        requests.extend(resdict["requests"])
                else:
                    requests.append({
                        Request.RequestType.value.item.value:Request.RequestType.value.BREAK_CONN.value,
                        Request.INITIATOR_CONN_ID.value:conn.get_conn_id()

                    })

        elif self.isOrganCell and self.isActuator:
            for connID in self.connectionsOut:
                conn:Connection = self.parentBaby.getConnection(connID)
                resdict = conn.tick(timestamp = timestamp, babyObj = self.parentBaby, links= links)
                
                if resdict:
                    requests.extend(resdict["requests"])

        
        else:
            for connID in self.connectionsOut:
                conn:Connection = self.parentBaby.getConnection(connID)
                if conn.check_robustness():
                    resdict = conn.tick(timestamp = timestamp, babyObj = self.parentBaby, links= links)
                    
                    if resdict:
                        requests.extend(resdict["requests"])
                else:
                    requests.append({
                        Request.RequestType.value.item.value:Request.RequestType.value.BREAK_CONN.value,
                        Request.INITIATOR_CONN_ID.value:conn.get_conn_id()

                    })
                
        # Chain the events happening in connections when tick a cell (based on request of each connection)
        conn_requests = []
        propagation_requests = []
        break_conn_requests = []
        for request in requests:
            if request[Request.RequestType.value.item.value] == Request.RequestType.value.CHAIN_EVENT.value:
                conn_requests.append(request)

            elif request[Request.RequestType.value.item.value] == Request.RequestType.value.PROPAGATION.value:
                propagation_requests.append(request)

            elif request[Request.RequestType.value.item.value] == Request.RequestType.value.BREAK_CONN.value:
                propagation_requests.append(request)

        self.chain_events(cell = self, conn_requests = conn_requests)
        self.parentBaby.add_propagation_requests(propagation_requests) 
        self.parentBaby.add_break_conn_requests(break_conn_requests)

        # Step 3: Apply cell logic
        pass


        # Step 4: Fill the snapshot
        self.fillSnapshot(timestamp)


        # 如果此细胞存在任何输入或输出conn，有着未结束的Event，那么仍然返回真，并添加conn相关联的Cell
        have_actived_conn = False
        for conn_id in self.connectionsIn + self.connectionsOut:
            connObj = self.parentBaby.connections[conn_id]
            if connObj.records.search_events(progress_min=0) or connObj.records.search_events(end_min=timestamp):
                have_actived_conn = True
                
                cellObj = None
                if connObj.upstream_cell_id:
                    if connObj.upstream_cell_id != self.cellID:
                        cellid = connObj.upstream_cell_id
                        cellObj = self.parentBaby.getCell(cellID=cellid)
                if connObj.downstream_cell_id:
                    if connObj.downstream_cell_id != self.cellID:
                        down_end_signal:dict = connObj.status.get(ConnectionStatus.DOWN_END_SIGNAL.value,{})
                        if down_end_signal.get(connObj.constraints["signal_type"],None):
                            cellid = connObj.downstream_cell_id
                            cellObj = self.parentBaby.getCell(cellID=cellid)

                if cellObj:
                    schedule = [
                        {'obj':cellObj,
                        'tick_method':"tick",
                        'kwargs':{

                        }},
                    ]

                    # 在cellstepper中进行tick，tick过程中对所有outconn进行尝试激活，创建event，并在records中获取最新的link和branch
                    cellStepper.add(key=cellid,schedule=schedule)

        return have_actived_conn 
                
    

                    
            

    # --------------------------------------------------------
    # SNAPSHOT MANAGEMENT
    # --------------------------------------------------------
    def prepareForTick(self, newTimestamp: int):
        """
        Prepares a snapshot slot for newTimestamp by clearing it.
        """
        idx = newTimestamp % 2
        self._snapshots[idx].clear()
        self._validTimestamps.add(newTimestamp)
        self._currentTimestamp = newTimestamp

    def fillSnapshot(self, timestamp: int):
        """
        Copies the cell's live data into the snapshot for 'timestamp'.
        """
        if timestamp not in self._validTimestamps:
            raise ValueError(f"Cell {self.cellID} has no prepared snapshot for timestamp {timestamp}!")
        idx = timestamp % 2
        snap = self._snapshots[idx]

        snap.cellID = self.cellID
        

    def getSnapshot(self, timestamp: int) -> CellSnapshot:
        """Retrieve the snapshot for the given timestamp (barrier-based)."""
        if timestamp not in self._validTimestamps:
            return None
        return self._snapshots[timestamp % 2]
    

    def _calculateApoptosisProbability(self) -> float:
        """Placeholder logic: if q is large or alpha is low => higher chance."""
        if self.q_intg_acc > self.q_acc_max:
            return 1.0
        if self.alpha < 0.1:
            return 0.5
        return 0.0

    def _calculateDivisionProbability(self) -> float:
        """Placeholder: if p is large, alpha is good => more likely."""
        if self.p_intg_saving > 50 and self.alpha > 0.99:
            return 0.1
        return 0.0

    def _calculateConnectionCreationProb(self) -> float:
        """Placeholder for deciding if we want to request new connections."""
        return 0.5

    def _calculateConnectionDisconnectionProb(self) -> float:
        """Placeholder for removing underused or detrimental connections."""
        if self.alpha < 0.5:
            return 0.8
        return 0.2

    # --------------------------------------------------------
    # PULL-PUSH SUBMISSIONS
    # --------------------------------------------------------
    def _requestDivision(self):
        if self.parentBaby:
            newCellParams = {"parentCellID": self.cellID, "portion": 0.3}
            self.parentBaby.requestDivision(self.cellID, newCellParams)
        else:
            print(f"[Cell {self.cellID}] Division requested but no parentBaby reference!")

    def _requestApoptosis(self):
        if self.parentBaby:
            self.parentBaby.requestApoptosis(self.cellID)
        else:
            print(f"[Cell {self.cellID}] Apoptosis requested but no parentBaby reference!")

    def _requestConnectionCreation(self, targetCellID: str):
        if self.parentBaby:
            self.parentBaby.requestConnectionCreation(self.cellID, targetCellID)
        else:
            print(f"[Cell {self.cellID}] Connection creation to {targetCellID} requested, no parentBaby!")

    def _requestConnectionDisconnection(self, connectionObj):
        if self.parentBaby:
            self.parentBaby.requestConnectionDisconnection(connectionObj)
        else:
            print(f"[Cell {self.cellID}] Connection disconnection requested but no parentBaby reference!")


    def inInsolation(self)->bool:
        """Check if this cell has no working connections. And is not receiving external returns."""
        for conn in self.connectionsIn:
            if conn.isWorking() == True:
                return False
        for conn in self.connectionsOut:
            if conn.isWorking() == True:
                return False
        
        
        return True
    # --------------------------------------------------------
    # MISC
    # --------------------------------------------------------
    def __repr__(self):
        return (
            f"Cell(id={self.cellID}, "
            f"status={self.status}, "
            f"constraints={self.constraints}"
        )
        

"""
Baby.py

Refined logic for Cell initialization to avoid saturating the 20-connection domain 
immediately and to exclude cells with a full domain from candidate neighbors.

Key Changes:
1) Add `initDomainFraction` so that, if maxDomain=20, 
   we only create e.g. up to 10 connections for a new cell at creation. 
   This leaves 'slots' for subsequent cells to connect back.
2) Exclude any cell whose domain is already full (>= maxDomain) 
   when picking neighbors for the new cell.
3) Similar improvements in both _assignInitialDomain() and _assignOneCellDomain().
"""
from StepperIterator import *
from BranchActivationRegistry import *
from NeuralNetworkVisualizer import NeuralNetworkVisualizer

class Baby:
    def __init__(
        self,
        maxDomain: int = 20,
        initDomainFraction: float = 0.5,
        allowDomainBreak: bool = True,
        enableInternalLoop: bool = True,
        ablationEarly: bool = False,
        A_cmd_char_index: map = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25}, # map of AudioCharacters (letters)
        A_OrganCellsNum: int = 26,
        A_bufferCellNum: int = 26, # buffer cells are the cells that receive or process specific stimuli from the certain organ before sending it to/after receiving it from the brain (bunch of normal cells)
        A_batchSize: int = int(0.5 * 26),
        
        V_OrganCellsNum: int = 8,
        V_bufferCellNum: int = 8,
        V_batchSize: int = int(0.5 * 8),

        
        LAct_OrganCellsNum: int = 4*2, # the Actuator(s)/signal output interface(s) of the baby
        LAct_bufferCellNum: int = 4*2,
        LAct_batchSize: int = int(0.5 * 4*2),
        
        LState_OrganCellsNum: int = 4*2,
        LState_bufferCellNum: int = 4*2,
        LState_batchSize: int = int(0.5 * 4*2),
        
        LPain_OrganCellsNum: int = 4*2 * 0,
        LPain_bufferCellNum: int = 4*2* 2 * 0,
        LPain_batchSize: int = int(0.5 * 4*2)*0,
        
        Reward_OrganCellsNum: int = 8*0,
        Reward_bufferCellNum: int = 8* 2*0,
        Reward_batchSize: int = int(0.5 * 8)*0,

        centralCellsNum: int = 16,
        centralCell_batchSize: int = int(0.5 * 16),

        loopNum: int = 5,
        
        IOFraction: float = 0.5, 
        trueConnectionFraction: float = 0.0, # all connections initialized as false connections
        
        cellStepper:StepperIterator = None, # 被用于循环调用Cell的tick函数 （Cell在更新链接状态时，先out后in）
        branchLossAllocationStepper: StepperIterator = None, # 被用于循环调用在branch之间分配loss

        branchActivationRegistry:BranchInfoActivationRegistry = None, # 用于
        cellBehaviorController:"CellBehaviorController" = None,
        visualize:bool = True 


    ):
        self.maxDomain = maxDomain
        self.initDomainFraction = initDomainFraction
        self.allowDomainBreak = allowDomainBreak
        self.enableInternalLoop = enableInternalLoop
        self.ablationEarly = ablationEarly

        self.A_OrganCellsNum = A_OrganCellsNum
        self.A_bufferCellNum = A_bufferCellNum
        self.A_batchSize = A_batchSize

        self.V_OrganCellsNum = V_OrganCellsNum
        self.V_bufferCellNum = V_bufferCellNum
        self.V_batchSize = V_batchSize

        self.LAct_OrganCellsNum = LAct_OrganCellsNum
        self.LAct_bufferCellNum = LAct_bufferCellNum
        self.LAct_batchSize = LAct_batchSize

        self.LState_OrganCellsNum = LState_OrganCellsNum
        self.LState_bufferCellNum = LState_bufferCellNum
        self.LState_batchSize = LState_batchSize

        self.LPain_OrganCellsNum = LPain_OrganCellsNum
        self.LPain_bufferCellNum = LPain_bufferCellNum
        self.LPain_batchSize = LPain_batchSize

        self.Reward_OrganCellsNum = Reward_OrganCellsNum
        self.Reward_bufferCellNum = Reward_bufferCellNum
        self.Reward_batchSize = Reward_batchSize

        self.centralCellsNum = centralCellsNum
        self.centralCell_batchSize = centralCell_batchSize

        self.loopNum = loopNum

        self.A_cmd_char_index = A_cmd_char_index

        self.IOFraction = IOFraction
        self.trueConnectionFraction = trueConnectionFraction

        self.cellStepper:"StepperIterator" = cellStepper if cellStepper else StepperIterator()
        self.branchLossAllocationStepper:"LossAllocationStepper" = branchLossAllocationStepper if branchLossAllocationStepper else LossAllocationStepper()
        self.branchActivationRegistry:"BranchInfoActivationRegistry" = branchActivationRegistry if branchActivationRegistry else BranchInfoActivationRegistry()
        self.cellBehaviorController:"CellBehaviorController" = cellBehaviorController if cellBehaviorController else CellBehaviorController(baby=self, registry = self.branchActivationRegistry)





        # Data structures
        self.cells: Dict[int, Cell] = {}
        self.connections: Dict[int, Connection] = {}

        # Organ Cells and their buffers
        self.A_OrganCellsList: List[int] = []  # list[CellID]
        self.A_bufferCellsList: List[int] = []  # list[CellID]
        self.V_OrganCellsList: List[int] = []  # list[CellID]
        self.V_bufferCellsList: List[int] = []  # list[CellID]
        self.LAct_OrganCellsList: List[int] = []  # list[CellID] # FL*2, FR*2, BL*2, BR*2
        self.LAct_bufferCellsList: List[int] = []  # list[CellID]
        self.LState_OrganCellsList: List[int] = []  # list[CellID] # FL*2, FR*2, BL*2, BR*2
        self.LState_bufferCellsList: List[int] = []  # list[CellID]
        self.LPain_OrganCellsList: List[int] = []  # list[CellID]
        self.LPain_bufferCellsList: List[int] = []  # list[CellID]
        self.Reward_OrganCellsList: List[int] = []  # list[CellID]
        self.Reward_bufferCellsList: List[int] = []  # list[CellID]
        

        # Central Cells
        self.centralCellsList: List[int] = []  # list[CellID]

        # Internal Loop
        self.loopIDs: List[int] = []  # list[CellID]


        # Keep track of structural change requests
        self._changeRequests = []

        # Current time
        self.currentTime = 0

        # requests

        self._propagation_requests:list[dict] = []
        self.break_conn_requests:list[dict] = []


        # Cell Behavior List
        self._cell_behavior_list:List[List[dict]] = [] # structure see: return of "CellBehaviorController._trigger_action"

        # Initialize the network
        self.initializeNetwork()

        self.visualize = visualize
        if visualize:
            self.visualizer:"NeuralNetworkVisualizer" = NeuralNetworkVisualizer()

    @property
    def propagation_requests(self):
        if self._propagation_requests:
            # import debugpy
            # debugpy.breakpoint()
            return self._propagation_requests
        return self._propagation_requests
    @propagation_requests.setter
    def propagation_requests(self, value):

        self._propagation_requests = value

    # ----------------------------------------------------------------
    # Network Initialization
    # ----------------------------------------------------------------

    def initializeNetwork(self):
        """
        Create all cells and potential (false) connections or initial true
        connections. Also handle specialized organs, ablation toggles, etc.
        """
        self._createOrganAndBufferCells()

        self._createCentralCells(bufferCellsLists=[self.A_bufferCellsList, self.V_bufferCellsList, self.LAct_bufferCellsList, self.LState_bufferCellsList, self.LPain_bufferCellsList, self.Reward_bufferCellsList], 
                                 batch=self.centralCell_batchSize, 
                                 centralCellNum=self.centralCellsNum, 
                                 averageConnectionNum=self.maxDomain * self.initDomainFraction, 
                                 trueConnectionFraction=self.trueConnectionFraction, 
                                 IOFraction=self.IOFraction)

        # Possibly create certain ring or special structure if config says so
        if self.enableInternalLoop:
            self._createInternalLoopCore()


        # Possibly ablation modifications if config demands it
        if self.ablationEarly:
            self._applyEarlyAblation()

        # # Prepare snapshot 0 for each cell
        # for c in self.cells.values():
        #     c.prepareForTick(0)

        # for conn in self.connections.values():
        #     conn.prepareForTick(0)

        # # 准备步进迭代器
        



    def _createOrganAndBufferCells(self):
        """
        Create specialized organ cells for different organ systems.
        Each organ system has a predefined number of cells.
        """

        
        averageConnectionNum = self.maxDomain * self.initDomainFraction

        # Create auditory organ cells
        for i in range(self.A_OrganCellsNum):
            c = Cell(isOrganCell=True, parentBaby=self,isReceptor=True)
            cid = c.cellID
            self.cells[cid] = c
            c.prepareForTick(0)
            self.A_OrganCellsList.append(cid)
            self._createConnection(upstreamID=None, downstreamID=cid, isTrueConnection=True, isReceptor=True)


        # Create auditory buffer cells
        self.A_bufferCellsList = self._createBufferCells(self.A_OrganCellsList, batch=self.A_batchSize, connectionDirection_organ2buffer = ConnectionDirection.INPUT.value, bufferCellNum=self.A_bufferCellNum, averageConnectionNum = averageConnectionNum, IOFraction=self.IOFraction, trueConnectionFraction=self.trueConnectionFraction) 

        # Create visual organ cells
        for i in range(self.V_OrganCellsNum):
            c = Cell(isOrganCell=True, parentBaby=self, isReceptor=True)
            cid = c.cellID
            self.cells[cid] = c
            c.prepareForTick(0)
            self.V_OrganCellsList.append(cid)
            self._createConnection(upstreamID=None, downstreamID=cid, isTrueConnection=True, isReceptor=True)

        # Create visual buffer cells
        self.V_bufferCellsList = self._createBufferCells(self.V_OrganCellsList, batch=self.V_batchSize, connectionDirection_organ2buffer = ConnectionDirection.INPUT.value, bufferCellNum=self.V_bufferCellNum, averageConnectionNum = averageConnectionNum, IOFraction=self.IOFraction, trueConnectionFraction=self.trueConnectionFraction)

        # Create limbic actuator organ cells
        for i in range(self.LAct_OrganCellsNum):
            c = Cell(isOrganCell=True, parentBaby=self, isActuator= True)
            cid = c.cellID
            self.cells[cid] = c
            c.prepareForTick(0)
            self.LAct_OrganCellsList.append(cid)
            self._createConnection(upstreamID=cid, downstreamID=None, isTrueConnection=True, isActuator=True)

        # Create limbic actuator buffer cells
        self.LAct_bufferCellsList = self._createBufferCells(self.LAct_OrganCellsList, batch=self.LAct_batchSize, connectionDirection_organ2buffer = ConnectionDirection.OUTPUT.value, bufferCellNum=self.LAct_bufferCellNum, averageConnectionNum = averageConnectionNum, IOFraction=self.IOFraction, trueConnectionFraction=self.trueConnectionFraction)

        # Create limbic state organ cells
        for i in range(self.LState_OrganCellsNum):
            c = Cell(isOrganCell=True, parentBaby=self,isReceptor= True)
            cid = c.cellID
            self.cells[cid] = c
            c.prepareForTick(0)
            self.LState_OrganCellsList.append(cid)
            self._createConnection(upstreamID=None, downstreamID=cid, isTrueConnection=True, isReceptor=True)

        # Create limbic state buffer cells
        
        self.LState_bufferCellsList = self._createBufferCells(self.LState_OrganCellsList, batch=self.LState_batchSize, connectionDirection_organ2buffer = ConnectionDirection.INPUT.value, bufferCellNum=self.LState_bufferCellNum, averageConnectionNum = averageConnectionNum, IOFraction=self.IOFraction, trueConnectionFraction=self.trueConnectionFraction)

        # Create limbic pain organ cells
        # (暂时弃用)
        for i in range(self.LPain_OrganCellsNum):
            c = Cell(isOrganCell=True, parentBaby=self, isReceptor=True)
            cid = c.cellID
            self.cells[cid] = c
            c.prepareForTick(0)
            self.LPain_OrganCellsList.append(cid)
            self._createConnection(upstreamID=None, downstreamID=cid, isTrueConnection=True, isReceptor=True)

        # Create limbic pain buffer cells
        
        self.LPain_bufferCellsList = self._createBufferCells(self.LPain_OrganCellsList, batch=self.LPain_batchSize, connectionDirection_organ2buffer = ConnectionDirection.INPUT.value, bufferCellNum=self.LPain_bufferCellNum, averageConnectionNum = averageConnectionNum, IOFraction=self.IOFraction, trueConnectionFraction=self.trueConnectionFraction)

        # Create reward organ cells
        # (暂时弃用)
        for i in range(self.Reward_OrganCellsNum):
            c = Cell(isOrganCell=True, parentBaby=self, isReceptor= True)
            cid = c.cellID
            self.cells[cid] = c
            c.prepareForTick(0)
            self.Reward_OrganCellsList.append(cid)
            self._createConnection(upstreamID=None, downstreamID=cid, isTrueConnection=True, isReceptor=True)

        # Create reward buffer cells
        
        self.Reward_bufferCellsList = self._createBufferCells(self.Reward_OrganCellsList, batch=self.Reward_batchSize, connectionDirection_organ2buffer = ConnectionDirection.INPUT.value, bufferCellNum=self.Reward_bufferCellNum, averageConnectionNum = averageConnectionNum, IOFraction=self.IOFraction, trueConnectionFraction=self.trueConnectionFraction)


    # create 
    def _initializeCellwithConnections(self, cell_groups: List[(BiMultiMap, int, int, float)], isOrganCell=False, connectionDirection_organ2buffer=None, connectionType=None):
        """
        Create a new cell with connections.
        @params
        cell_groups: list of tuples (cellIDBMMap K-V (cellID, connectionnum of the cell) (e.g. cells of different organs), IOFraction (e.g.  0.5), connectionNum (e.g. 0, 1, 2), trueConnectionFraction (e.g. 0.5))

        @ return
        cellID

        @ note: the cell obj is added to self.cells, but please specify the list to add its ID to;
        you needn't tackle connections.
        """
        # 0) Initialize the new Cell
        c = Cell(isOrganCell=isOrganCell, parentBaby=self)
        cid = c.cellID
        self.cells[cid] = c
        c.prepareForTick(0)

        for (cellIDBMM, IOFraction, connectionNum, trueConnectionFraction) in cell_groups:
            
            # 1) Calculate the number of different kinds of connections
            # 1.1) If the connectionNum is 1, regard IOFraction and trueConnectionFraction as probabilities
            if connectionNum == 1:
                IOFraction = random.uniform(0, 1) < IOFraction
                trueConnectionFraction = random.uniform(0, 1) < trueConnectionFraction

            true_connection_num = int(connectionNum * trueConnectionFraction)
            false_connection_num = connectionNum - true_connection_num
            
            input_true_connection_num = int(true_connection_num * IOFraction)
            input_false_connection_num = int(false_connection_num * IOFraction)
            output_true_connection_num = true_connection_num - input_true_connection_num
            output_false_connection_num = false_connection_num - input_false_connection_num
            
            # 2) Create input true connections
            # 2.1) Fetch the input cells by choosing the cells with the least connections. (for organ cells, their connection num should add 1 for receiving outter stumuli)
            cellIDBMM: BiMultiMap = cellIDBMM
            input_cell_pairs = cellIDBMM.get_sorted_pairs(sort_by='value', reverse=False,top_n=input_true_connection_num)
            input_cell_ids = [input_cell for (input_cell,_) in input_cell_pairs]

            # 2.2) Create the connections
            for input_cell_id in input_cell_ids:
                self._createConnection(input_cell_id, cid, isTrueConnection=True, connectionDirection_organ2buffer=connectionDirection_organ2buffer, connectionType=connectionType)
            
            # 3) Create output true connections
            # 3.1) Fetch the output cells by choosing the cells with the least connections. (for organ cells, their connection num should add 1 for receiving outter stumuli)
            output_cell_pairs = cellIDBMM.get_sorted_pairs(sort_by='value', reverse=True,top_n=output_true_connection_num)
            output_cell_ids = [output_cell for (output_cell,_) in output_cell_pairs]

            # 3.2) Create the connections
            for output_cell_id in output_cell_ids:
                self._createConnection(cid, output_cell_id, isTrueConnection=True, connectionDirection_organ2buffer=connectionDirection_organ2buffer, connectionType=connectionType)

            # 4) Create input false connections (False connections are potential connections for the future, so they have a bigger range of choices)
            # 4.1) Fetch the input cells by choosing the cells with the least connections. (for organ cells, their connection num should add 1 for receiving outter stumuli)
            input_cell_pairs = cellIDBMM.get_sorted_pairs(sort_by='value', reverse=False,top_n=input_false_connection_num*2)
            input_cell_ids = random.sample([input_cell for (input_cell,_) in input_cell_pairs], input_false_connection_num)

            # 4.2) Create the connections
            for input_cell_id in input_cell_ids:
                self._createConnection(input_cell_id, cid, isTrueConnection=True, connectionDirection_organ2buffer=connectionDirection_organ2buffer, connectionType=connectionType)

            # 5) Create output false connections
            # 5.1) Fetch the output cells by choosing the cells with the least connections. (for organ cells, their connection num should add 1 for receiving outter stumuli)
            output_cell_pairs = cellIDBMM.get_sorted_pairs(sort_by='value', reverse=True,top_n=output_false_connection_num*2)
            output_cell_ids = random.sample([output_cell for (output_cell,_) in output_cell_pairs], output_false_connection_num)

            # 5.2) Create the connections
            for output_cell_id in output_cell_ids:
                self._createConnection(cid, output_cell_id, isTrueConnection=True, connectionDirection_organ2buffer=connectionDirection_organ2buffer, connectionType=connectionType)


        return cid



    def _createBufferCells(self, organCellsList: List[int], batch:int, connectionDirection_organ2buffer: ConnectionDirection, bufferCellNum: int, averageConnectionNum: int, trueConnectionFraction: float, IOFraction: float, connectionType=None):
        """
        Create buffer cells for each organ system.
        
        @params
        organCellsList: list of cellIDs of organs cells
        batch: int, how many buffer cells to create once
        connectionDirection_organ2buffer: ConnectionDirection
        bufferCellNum: int
        averageConnectionNum: int
        trueConnectionFraction: float
        IOFraction: float
        connectionType: int

        @return
        bufferCellsList: list of buffer cellIDs

        @note: this function trys to make every cell can connect to every other cell in the network, and try to make every cell has the same number of connections (the last bunch of cells have less connections than normal cells as future connection interface).
        
        """

        cellIDBMM: BiMultiMap = BiMultiMap() # K-V (cellID, connectionnum of the cell)
        bufferCellsList = []
        batchCellsList = []
        for organCellID in organCellsList:
            cellIDBMM.add_association(organCellID, self.cells[organCellID].getConnectionNumber())
        for i in range(bufferCellNum):
            connectionNum = max((i%batch)*(batch/(bufferCellNum * averageConnectionNum + batch)), 1)
            cell_groups = [(cellIDBMM, IOFraction, connectionNum, trueConnectionFraction)]
            bufferCellID = self._initializeCellwithConnections(cell_groups=cell_groups, isOrganCell=False, connectionDirection_organ2buffer=connectionDirection_organ2buffer, connectionType=connectionType)

            # when batch is reached, add the batchCellsList to the bufferCellsList and update the cellIDBMM
            batchCellsList.append(bufferCellID)
            if len(batchCellsList) >= batch or len(batchCellsList) >= len(organCellsList) + len(bufferCellsList):
                bufferCellsList.extend(batchCellsList)
                for bufferCellID_batch in batchCellsList:
                    cellIDBMM.add_association(bufferCellID_batch, self.cells[bufferCellID_batch].getConnectionNumber())
                batchCellsList = []


        return bufferCellsList


    def _createCentralCells(self, bufferCellsLists: List[List[int]], batch:int, centralCellNum: int, averageConnectionNum: int, trueConnectionFraction: float, IOFraction: float, connectionType = None):
        """
        Create central cells. Responsible for handling and sending stimuli from/ to different buffer cells. (Main logic processor)
        @params 
        @return cellIDList of created central cells

        """
        cellIDBMM: BiMultiMap = BiMultiMap() # K-V (cellID, connectionnum of the cell)
        centralCellsList = []
        batchCellsList = []
        interfaceCellNum = 0
        for bufferCellsList in bufferCellsLists:
            if len(bufferCellsList) == 0:
                continue
            # 1）calculate the average connection num of the buffer cells
            averageConnectionNum_buffer = 0
            totalConnectionNum_buffer = 0
            for bufferCellID in bufferCellsList:
                totalConnectionNum_buffer += self.cells[bufferCellID].getConnectionNumber()
            averageConnectionNum_buffer = totalConnectionNum_buffer/len(bufferCellsList)

            # 2）fill the cellIDBMM using averageConnectionNum + connectionNum of the buffer cells - averageConnectionNum_buffer for regulation
            for bufferCellID in bufferCellsList:
                bufferCellConnectionNum = self.cells[bufferCellID].getConnectionNumber()
                if bufferCellConnectionNum < averageConnectionNum_buffer:
                    interfaceCellNum += 1
                cellIDBMM.add_association(bufferCellID, bufferCellConnectionNum - averageConnectionNum_buffer + averageConnectionNum)



        for i in range(centralCellNum):
            connectionNum = max((i%batch)*(batch/(centralCellNum * averageConnectionNum + batch)), 1)
            cell_groups = [(cellIDBMM, IOFraction, connectionNum, trueConnectionFraction)]
            centralCellID = self._initializeCellwithConnections(cell_groups=cell_groups, isOrganCell=False, connectionDirection_organ2buffer=None, connectionType=connectionType)

            # when batch is reached, add the batchCellsList to the centralCellsList and update the cellIDBMM
            batchCellsList.append(centralCellID)
            if len(batchCellsList) >= batch or len(batchCellsList) >= interfaceCellNum + len(centralCellsList):
                centralCellsList.extend(batchCellsList)
                for centralCellID_batch in batchCellsList:
                    cellIDBMM.add_association(centralCellID_batch, self.cells[centralCellID_batch].getConnectionNumber())
                batchCellsList = []

        return centralCellsList
    
    def _createInternalLoopCore(self, numCells: int = 5):
        """
        Create a ring or partial ring among a specified number of cells if
        ablation config allows. These might be 'true' connections from the start.
        """
        loopIDs = sorted([cid for cid in self.centralCellsList], reverse=True)[:numCells]
        for i in range(len(loopIDs)):
            upID = loopIDs[i]
            downID = loopIDs[(i + 1) % len(loopIDs)]
            self._createConnection(upID, downID, isTrueConnection=True)
        self.loopIDs = loopIDs


    def _countCellDomain(self, cellObj: "Cell") -> int:
        """
        Count how many neighbors (true + false) this cell has. 
        We'll define a method on Connection to check if it's true or false, 
        but for now we assume all connectionsIn + connectionsOut are domain.
        """
        if not cellObj:
            return 0
        domainSet = set()
        for c in cellObj.connectionsIn:
            domainSet.add(c)
        for c in cellObj.connectionsOut:
            domainSet.add(c)
        return len(domainSet)
    

    def get_latest_command(self):
        """
        The command baby sends to limbs
        """
        baby_control_command = []
        if self.currentTime == 0:
            baby_control_command = [0]*4
        else:
            for limb_cell_id in range(0, 4):
                cellObj1 = self.cells[self.LAct_OrganCellsList[limb_cell_id*2]]
                conn_1:"Connection" = self.connections[cellObj1.connectionsOut[-1]]
                events =  conn_1.records.search_events(start_min=self.currentTime)
                limb_cmd_bit_1 = 0
                if events:
                    event_1 = events[-1]
                    limb_cmd_bit_1 = 1 if event_1[Event.Down_End_Attenuation.value.item.value][Event.Down_End_Attenuation.value.Y0.value]>0 else 0
                    

                    
                cellObj2 = self.cells[self.LAct_OrganCellsList[limb_cell_id*2+1]]
                conn_2:"Connection" = self.connections[cellObj2.connectionsOut[-1]]
                events =  conn_2.records.search_events(start_min=self.currentTime)
                limb_cmd_bit_2 = 0
                if events:
                    event_2 = events[-1]
                    limb_cmd_bit_2 = 1 if event_2[Event.Down_End_Attenuation.value.item.value][Event.Down_End_Attenuation.value.Y0.value]>0 else 0
                
                
                baby_control_command.append ( limb_cmd_bit_1 + 2 * limb_cmd_bit_2)

    
        
        return self.currentTime, baby_control_command

        
    def _connectionExists(self, cidA: str, cidB: str) -> bool:
        """
        Returns True if any connection object already links cidA & cidB.
        """
        check1 = f"{cidA}__to__{cidB}"
        check2 = f"{cidB}__to__{cidA}"
        return (check1 in self.connections) or (check2 in self.connections)

    def _createConnection(
        self,
        upstreamID: int,
        downstreamID: int,
        isTrueConnection: bool = True,
        connectionDirection_organ2buffer=None, 
        trainable:bool = True,
        connectionType=None,
        isReceptor = False,
        isActuator = False
    ):
        """
        Create a connection linking 'upstreamID' to 'downstreamID'. 
        If isTrueConnection=False, mark it as a 'false' (potential) link.
        Exclude or skip if domain is full, unless we allow domain break logic.
        @param connectionDirection_organ2buffer: if not None, this will overwrite the direction of the connection from an organ cell to a buffer cell (as input or output). Two organ cells cannot directly connect
        @param connectionType: if not None, this will apply the connection type
        """
        if upstreamID == downstreamID:
            return

        # If domain limit is enforced, skip if both cells are full 
        # (unless allowDomainBreak is True).
        upObj:Cell = self.cells[upstreamID] if not isReceptor else None
        downObj = self.cells[downstreamID] if not isActuator else None

        if upObj and downObj and upObj.isOrganCell and downObj.isOrganCell:
            return # no connection is established between two organ cells
        
        
        if connectionDirection_organ2buffer is not None and (connectionDirection_organ2buffer == ConnectionDirection.INPUT.value and downObj.isOrganCell or connectionDirection_organ2buffer == ConnectionDirection.OUTPUT.value and upObj.isOrganCell):
            # switch upObj and downObj, upstreamID and downstreamID
            upObj, downObj = downObj, upObj
            upstreamID, downstreamID = downstreamID, upstreamID
        
        upDomain = self._countCellDomain(upObj)
        downDomain = self._countCellDomain(downObj)

        if not self.allowDomainBreak:
            if upDomain >= self.maxDomain and downDomain >= self.maxDomain:
                return



        connObj = Connection(
            is_true_connection=isTrueConnection,
            upstream_cell_id=upstreamID,
            downstream_cell_id=downstreamID,
            trainable=trainable,
            parent_baby=self,
            isReceptor = isReceptor,
            isActuator = isActuator

        )
        connObj.is_true_connection = isTrueConnection
        self.connections[connObj.get_conn_id()] = connObj

        # update references
        if upObj:
            upObj.connectionsOut.append(connObj.get_conn_id())
        if downObj:
            downObj.connectionsIn.append(connObj.get_conn_id())

        # if connObj.is_true_connection:
        #     connObj.prepareForTick(0)
        # 不全局tick了


    def _applyEarlyAblation(self):
        """
        If ablation config says remove some portion or skip certain logic, do it here.
        """
        pass

    def register_branchInfo(self,branch):
        self.branchActivationRegistry.add_BranchInfo(branchInfo=branch)
        

    def add_triggered_actions(self,actions:list):
        self._cell_behavior_list.append(actions)


    def add_propagation_requests(self, propagation_requests:list[dict]):
        for request in propagation_requests:
            if request[Request.RequestType.value.item.value] == Request.RequestType.value.PROPAGATION.value:
                self.propagation_requests.append(request)

    def add_break_conn_requests(self, break_conn_requests:list[dict]):
        for request in break_conn_requests:
            if request[Request.RequestType.value.item.value] == Request.RequestType.value.BREAK_CONN.value:
                self.break_conn_requests.append(request)

    @staticmethod
    def _calculate_upstream_reward_type(downstream_reward_type:"str", upstream_conn_signal_type:"str"):
        """
        return the upstream reward type
        """
        if upstream_conn_signal_type in {Signal_E.E.value}:
            return downstream_reward_type
        if upstream_conn_signal_type in {Signal_F.F.value}:
            if downstream_reward_type == RewardType.POSITIVE.value:
                return RewardType.NEGATIVE.value
            if downstream_reward_type == RewardType.NEGATIVE.value:
                    return RewardType.POSITIVE.value

        raise ValueError("Unknown signal type or reward type")

    def _add_branch_to_stepper_recursively(self,link:"dict",reward_type:str, temporal_contextual_view:int):

        branch:"BranchInfo" = link[Link.Condition.value.item.value][Link.Condition.value.BRANCH.value]
        if branch.reward_type == reward_type:
            return
        if branch.reward_type == RewardType.POSITIVE.value:
            return



        branch.reward_type = reward_type
        if branch.parent == None:
            self.register_branchInfo(branch=branch)
            ParameterGroup.static_differentiation(link=link,conn=self.connections[branch.get_conn_id()],reward_type = reward_type, loss_allocation_stepper= self.branchLossAllocationStepper)
        else:
            branch.parent.differentiation(link=link, reward_type=reward_type, loss_allocation_stepper= self.branchLossAllocationStepper)


        triggers:list[dict] = link[Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value] if link.get(Link.Trigger.value.item.value) else []


        if not triggers:
            conn_id = branch.get_conn_id()
            upstream_cell_id = self.connections[conn_id].upstream_cell_id
            up_cell_obj = self.cells[upstream_cell_id]
            for in_conn_id in up_cell_obj.connectionsIn:
                in_conn = self.connections[in_conn_id]
                contextual_events = in_conn.records.search_events(end_min=branch.get_timestamp() - temporal_contextual_view, finished = True)
                triggers.extend(contextual_events)  

        for event in triggers:
            _link = event[Event.LINK.value]
            _conn_id = event[Event.CONN_ID.value]
            _conn_obj = self.connections[_conn_id]
            _reward_type = self._calculate_upstream_reward_type(downstream_reward_type=reward_type, upstream_conn_signal_type=_conn_obj.constraints["signal_type"])
            self._add_branch_to_stepper_recursively(link=_link,reward_type=_reward_type, temporal_contextual_view=temporal_contextual_view)
    

    def _get_event(self,conn_id, event_id):
        return self.connections[conn_id].records.get_event(event_id=event_id)

    def _back_propagation(self,period:int = 2):
        """
        处理反向传播请求，特定周期调用
        """
        if self.currentTime % period != 0 or self.propagation_requests.__len__() == 0:
            return
        
        positive_requests = []
        negative_requests = []

        for request in self.propagation_requests:
            if request[Request.REWARD_TYPE.value] == RewardType.POSITIVE.value:
                positive_requests.append(request)

            if request[Request.REWARD_TYPE.value] == RewardType.NEGATIVE.value:
                negative_requests.append(request)

        # propagate positive ones with perference
        for request in positive_requests:
            event = self._get_event(conn_id=request[Request.INITIATOR_CONN_ID.value], event_id= request[Request.INITIATOR_EVENT_ID.value])
            link = event[Event.LINK.value]
            self._add_branch_to_stepper_recursively(link = link,reward_type= RewardType.POSITIVE.value, temporal_contextual_view=request[Request.TEMPORAL_CONTEXTUAL_VIEW.value])

        for request in negative_requests:
            event = self._get_event(conn_id=request[Request.INITIATOR_CONN_ID.value], event_id= request[Request.INITIATOR_EVENT_ID.value])
            link = event[Event.LINK.value]
            self._add_branch_to_stepper_recursively(link = link,reward_type= RewardType.NEGATIVE.value, temporal_contextual_view=request[Request.TEMPORAL_CONTEXTUAL_VIEW.value])

        self.propagation_requests.clear()

    def _break_conns(self):
        for request in self.break_conn_requests:
            conn_id = request[Request.INITIATOR_CONN_ID.value]
            conn = self.connections[conn_id]
            conn.status[ConnectionStatus.ENABLED.value] = False
            self.cells[conn.upstream_cell_id].remove_connection(conn=conn)
            self.cells[conn.downstream_cell_id].remove_connection(conn=conn)
            self.cellBehaviorController.update_connection(conn=conn,action="break")

    # 
    def _tick_branchLossAllocationStepper(self,freq = 10)-> list[dict]:
        """
        根据特定频率进行损失传播，并更新参数。最后分析所触发的行为：splitting， connecting。
        
        """
        stable_branches:list[BranchInfo] = []
        for i in range(0,10):
            removed_schedules = self.branchLossAllocationStepper.tick(timestamp=self.currentTime* freq + i)
            stable_branches.extend([removed_schedule[0]['obj'] for removed_schedule in removed_schedules])
        for br in stable_branches:
            self.cellBehaviorController.update_branch(timestamp=self.currentTime, branch=br)

        return stable_branches

    def _apply_behaviours(self,default_Y_p0 = 3, default_X_p1 = 1, default_Y_p1 = 2, min_communication_time = 3, proactivity = 0.02):
        """
        应用行为并更新参数

        # 动作候选列表 (List[Tuple[float, Dict]])
        # 包含所有符合条件的候选动作，按最终评分降序排序
        [
        # 第一个候选动作
        (
            # 最终调整后的动作评分 (float)
            # 用于候选动作的全局优先级排序（分值越高优先级越高）
            8.42,  
            
            # 候选动作详细信息 (Dict)
            {
            # 匹配结果详情 (Dict)
            'pairing': {
                # 匹配的连接ID (int)
                # 表示被匹配的连接标识符
                'conn_id': 123,
                
                # 时间差范围 (Tuple[min_diff, max_diff])
                # 被匹配的分支与目标分支的最小/最大时间差
                'time_diff_range': (20, 45),
                
                # 匹配的分支集合 (Set[BranchInfo])
                # 实际参与匹配的分支对象
                'branches': {branch1, branch2},
                
                # 原始匹配评分 (float)
                # 注册表计算的原始匹配质量（不包含后续调整）
                'score':8.42, 
                
                # 相关目标分支列表 (List[BranchInfo])
                # 触发本次匹配的目标分支
                'related_targets': [target_branch1, target_branch2]
            },
            
            # 上游细胞信息 (Tuple[cell_id, splitting_status])
            # cell_id: 上游细胞唯一标识
            # splitting_status: 该细胞分裂阈值是否满足 (True/False)
            'up_cell': (101, True),
            
            # 下游细胞信息 (Tuple[cell_id, splitting_status])
            # cell_id: 下游细胞唯一标识
            # splitting_status: 该细胞分裂阈值是否满足 (True/False)
            'down_cell': (202, False),
            
            # 时间方向标识 (bool)
            # True: 正向匹配（关注后续事件）
            # False: 反向匹配（关注先前事件）
            'direction': True,

            # 新创建链接的信号类型
            'signal_type': Signal_E.E.value
            }
        ),
        
        # 第二个候选动作
        (
            # 最终调整后的动作评分
            7.15,
            
            # 候选动作详细信息
            {
            'pairing': {
                # 匹配详情结构同上
                ...
            },
            'up_cell': (305, True),
            'down_cell': (408, True),
            'direction': False
            }
        )
        ]
        """
        duplicated_branches = set()
        for promising_actions in self._cell_behavior_list:
            for action in promising_actions:
                score = action[0]
                info = action[1]

                new_conn_signal_type = info["signal_type"]

                up_cell_id = info["up_cell"][0]
                up_cellObj:Cell = self.cells[up_cell_id]
                up_cell_splitting:bool = info["up_cell"][1]

                down_cell_id = info["down_cell"][0]
                down_cellObj:Cell = self.cells[down_cell_id]
                down_cell_splitting:bool = info["down_cell"][1]

                target_branchObj:"BranchInfo" = info["pairing"]['related_targets'][0]
                branchObj:"BranchInfo" = list(info["pairing"]["branches"])[0]
                
                time_diff_range = info['pairing']["time_diff_range"]
                direction = info['direction']

                # No I-I O-O connection
                if up_cellObj.isReceptor and down_cellObj.isReceptor or up_cellObj.isActuator and down_cellObj.isActuator:
                    continue

                if target_branchObj.get_branch_id() in duplicated_branches or branchObj.get_branch_id() in duplicated_branches:
                    break
                if target_branchObj.get_timestamp()>branchObj.get_timestamp():
                    up_br = branchObj
                    down_br = target_branchObj
                    # if direction:
                    #     # target branch should be earlier
                    #     raise ValueError("Direction and timestamp conflict (target should be earlier)")
                else:
                    up_br = target_branchObj
                    down_br = branchObj
                    # if not direction:
                    #     # target branch should be later
                    #     raise ValueError("Direction and timestamp conflict (target should be later)")

                down_conn:"Connection" = self.connections[down_br.get_conn_id()]
                up_conn:"Connection" = self.connections[up_br.get_conn_id()]
                # 1 validating action
                # 不允许down cell为Receptor， up cell 为Actuator
                if up_cellObj.isActuator or down_cellObj.isReceptor:
                    continue
                
                

                # 不允许time diff 跨0
                if np.sign(time_diff_range[0]) != np.sign(time_diff_range[1]):
                    continue
                
                # 验证查询方向不一致
                if (time_diff_range[0] > 0 and direction == False) or  (time_diff_range[0] <= 0 and direction == True):
                    continue

                # 验证时间裕度允许至少创建一个链接
                conn_timespan = min(abs(time_diff_range[0]),abs(time_diff_range[1]))
                if conn_timespan < min_communication_time:
                    continue



                # 2 计算参数
                # 2.1 新链接数目与细胞数
                splitting = (up_cell_splitting or down_cell_splitting)
                splitting = True if conn_timespan/2 >=min_communication_time else False  
                # 不允许两个organ cell直接连接，中间必须插入一个细胞
                if (up_cellObj.isOrganCell and down_cellObj.isOrganCell):
                    splitting = True
                new_conn_num = 2 if splitting else 1


                # 2.2 新约束条件

                """
                constraints = {
                    'signal_type':singal_type,
                    'ts': ts,
                    'k': k,
                    'b':b
                }
                """

                new_conn_signal_types = [Signal_E.E.value] * new_conn_num
                new_conn_signal_types[-1] = new_conn_signal_type
                        

                ts = min(abs(time_diff_range[0]),abs(time_diff_range[1]))
                
                calc_kb = lambda p1, p2: ((p2[1]-p1[1])/(p2[0]-p1[0]), p1[1] - ((p2[1]-p1[1])/(p2[0]-p1[0]))*p1[0])
                p0 = (0,default_Y_p0)
                p1 = (max(abs(time_diff_range[0]- time_diff_range[1]), 1), default_Y_p1)
                
                (k,b) = calc_kb(p0,p1) 
                k = abs(k)

                new_conn_tses = [ts] * new_conn_num
                new_conn_bs = [b]* new_conn_num
                new_conn_ks = [k]* new_conn_num
                new_conn_up_cell_ids = [up_cell_id]* new_conn_num
                new_conn_down_cell_ids = [down_cell_id] * new_conn_num

                if new_conn_num == 2:
                    new_conn_tses[0] = new_conn_tses[0]//2
                    k_0, b_0 =  calc_kb(p0,(default_X_p1,default_Y_p1))
                    new_conn_ks[0] = -k_0
                    new_conn_bs[0] = b_0 
                    
                    new_conn_tses[1] = new_conn_tses[1] - new_conn_tses[0]
   


                # 3 创建细胞和连接
                # 3.1 创建细胞
                if splitting:
                    new_cellObj = Cell(parentBaby=self)
                    self.cells[new_cellObj.cellID] = new_cellObj
                    self.centralCellsList.append(new_cellObj.cellID)

                    new_conn_up_cell_ids[1] = new_cellObj.cellID
                    new_conn_down_cell_ids[0] = new_cellObj.cellID

                # 3.2 创建链接
                new_conns:list[Connection] = []
                for i in range(0, new_conn_num):
                    conn = Connection(parent_baby=self,upstream_cell_id=new_conn_up_cell_ids[i],downstream_cell_id=new_conn_down_cell_ids[i],initial_constraints={
                        "signal_type":new_conn_signal_types[i],
                        "ts":math.floor(new_conn_tses[i] * (1-proactivity)),
                        "k": new_conn_ks[i],
                        "b": new_conn_bs[i]
                        
                    },
                    trainable= True)
                    new_conns.append(conn)
                    self.connections[conn.get_conn_id()] = conn

                    self.cells[new_conn_up_cell_ids[i]].add_connection(conn)
                    self.cells[new_conn_down_cell_ids[i]].add_connection(conn)


                    if i == 0:
                        def replace_int_in_structure(data, old_val, new_val):
                            # 处理字典类型
                            if isinstance(data, dict):
                                return {
                                    replace_int_in_structure(k, old_val, new_val): replace_int_in_structure(v, old_val, new_val)
                                    for k, v in data.items()
                                }
                            
                            # 处理列表、元组、集合等序列类型
                            if isinstance(data, (list, tuple, set, frozenset)):
                                # 根据原类型创建新序列
                                return type(data)(
                                    replace_int_in_structure(item, old_val, new_val)
                                    for item in data
                                )
                            
                            # 处理基础类型
                            if isinstance(data, int) and data == old_val:
                                return new_val
                            
                            # 对于其他不可变类型（字符串、浮点数等）直接返回
                            return data
                        
                        # 适应性复制ParamGroup
                        if up_br.parent:
                            initial_weights = copy.deepcopy(up_br.parent.windows[0][ParamWindow.WEIGHTS.value])
                            initial_weights = replace_int_in_structure(initial_weights, old_val=down_br.get_conn_id(), new_val=conn.get_conn_id())
                        
                            initial_bias = copy.deepcopy(up_br.parent.windows[0][ParamWindow.BIAS.value])
                            initial_bias = replace_int_in_structure(initial_bias, old_val=down_br.get_conn_id(), new_val=conn.get_conn_id())
                        
                            initial_topology = copy.deepcopy(up_br.parent.windows[0][Topology.item.value])
                            initial_topology = replace_int_in_structure(initial_topology, old_val=down_br.get_conn_id(), new_val=conn.get_conn_id())

                            
                            conn.propagation_manager.add_group(initial_weights=initial_weights,initial_bias=initial_bias,initial_topology=initial_topology)
                        # else:
                        #     conn.propagation_manager.add_group(initial_weights=None,initial_bias=None,initial_topology=None)
                    conn.propagation_manager.update_topology()
                
                # 4 收尾工作
                # 4.1 将所涉及的branch 标记为outdated，然后更新这些branch. 
                targets:list["BranchInfo"] = info["pairing"]['related_targets']
                branchObjs:list["BranchInfo"] = list(info["pairing"]["branches"])
                for br in targets + branchObjs:
                    br.outdated = True
                    self.cellBehaviorController.update_branch(branch=br, timestamp = self.currentTime)
                    duplicated_branches.add(br.get_branch_id())
                # 4.2 更新connection
                for conn in new_conns:
                    self.cellBehaviorController.update_connection(conn=conn,timestamp = self.currentTime, action="establish")
                
                # 5 退出 （一组竞争的候选动作最多执行一个）
                break
        self._cell_behavior_list.clear()



    def tick_visualizer(self, step_length = 1):
        if self.visualize and self.currentTime%step_length == 0:

            self.visualizer.update_network(cells=self.cells.values(),conns=self.connections.values(),active_cells={value["schedule"][0]['obj'] for value in self.cellStepper.workspace.values()})
    # ----------------------------------------------------------------
    # The main TICK function
    # ----------------------------------------------------------------

    def tick(
        self,
        time: int,
        baby_limb_state: List[Any] = None,
        mentor_control_command: Dict[str, Any] = None,
        internal_feedback: List[Any] = None,
        external_feedback: List[Any] = None,
    ):

        self.currentTime = time

        if self.currentTime >0:
            # 0) Handle last timestamp requests (no topology modification )
            self._back_propagation()

            # 0) Handle last timestamp requests (with topology modification )
            self._break_conns()
            
            self._apply_behaviours()

        # 1) Apply environment signals
        self._applyEnvironmentSignals(
            timestamp = time, baby_limb_state=baby_limb_state, mentor_control_command=mentor_control_command, internal_feedback=internal_feedback, external_feedback=external_feedback
        )

        # # 2) Possibly update connections
        # self._updateConnections(time)

        # # 3) Update cells
        # self._updateCells(time)

        # 4) Process structural changes
        # self._processChangeRequests()

        # Deliver messages after all cells have processed their tick
        self._deliverMessages()

        self.tick_visualizer()

        self.cellStepper.tick(timestamp=self.currentTime)

        self._tick_branchLossAllocationStepper()


        
        

    def _convert_to_binary(self, original_decimal_list: list[int]):
        # Define the mapping based on the value of each element in the 4-bit input list
        mapping = {
            -1: [None, None],
            0: [0, 0],
            1: [1, 0],
            2: [0, 1],
            3: [1, 1]
        }
        
        # Initialize the result list
        result = []
        
        # For each element in the 4-bit list, map it to its corresponding 8-bit representation
        for bit in original_decimal_list:
            result.extend(mapping[bit])  # Extend the result with the two-bit pair
    
        return result
    def _applyEnvironmentSignals(
        self,
        timestamp,
        baby_limb_state:list, # list of limb states for baby to refer, （Baby knows his limb state） 4 bits, we need to convert to 4*2 bits (8 bits) 
        mentor_control_command:dict, # dict of mentor control commands, including A, V, LStim, LState, LPain, Reward
        internal_feedback, #
        external_feedback
    ):
        """
        Domain-specific routing of signals to organ cells or logic.

        - mentor_control_command :

         {
                CommandAttr.A.value: None,
                CommandAttr.V.value: [-1,-1,-1,-1],
                CommandAttr.LSTIM.value: [-1,-1,-1,-1],
                CommandAttr.CHECK.value: 0,
                CommandAttr.LOOP.value: 0,
                CommandAttr.LIMB_STATE.value: [[-1, -1, -1, -1]], # 此LIMB_STATE为期望的LIMB_STATE，并非baby的实际limb state， 当reward type不为None, or pending时，应当根据此limb state进行反向传播
                CommandAttr.REWARD_SIZE.value: 0, 
                RewardType.item.value: None

            }
        """

        # Apply limb state signals
        baby_limb_state_8bits = self._convert_to_binary(baby_limb_state)
        for i,cellID in enumerate(self.LState_OrganCellsList):
            cellObj:Cell = self.cells[cellID]
            # Apply auditory signals
            if baby_limb_state_8bits[i] is not None and baby_limb_state_8bits[i]>0: 
                cellObj.receptor_input(timestamp=timestamp, value = baby_limb_state_8bits[i])
                schedule = [
                     {'obj':cellObj,
                    'tick_method':"tick",
                    'kwargs':{

                    }},
                ]

                # 在cellstepper中进行tick，tick过程中对所有outconn进行尝试激活，创建event，并在records中获取最新的link和branch
                self.cellStepper.add(key=cellID,schedule=schedule) 

                # 以下不在baby的tick中进行
                # ...获得当前的branch的input，设置其()，但只允许在下个timestamp后开始分配loss
                # self.branchActivationRegistry.add_chain()
                # self.cellBehaviorController.add_chain()# 自动过滤不可训练conn

                
                # cellObj.p_prop_working_internal = baby_limb_state_8bits[i]*1.0 # organ cells provide 100% of their working capital and some reward to downstream cells
            else:
                cellObj.receptor_input_signal = None
            
            # TODO: organ cells recover from debt automatically, promosing that they are activated by input signals

        # Apply limb state feedback

        expected_limb_state_8bits = [None] * 8
        if mentor_control_command.get(RewardType.item.value, RewardType.PENDING.value) not in {RewardType.PENDING.value, None}:
            expected_limb_state = mentor_control_command[CommandAttr.LIMB_STATE.value][0]
            expected_limb_state_8bits = self._convert_to_binary(expected_limb_state)
            
        for i,cellID in enumerate(self.LAct_OrganCellsList):
            cellObj:Cell = self.cells[cellID]
            
            if expected_limb_state_8bits[i] is not None: 
                cellObj.actuator_output(timestamp=timestamp, validation_value= expected_limb_state_8bits[i],reward_type = mentor_control_command.get(RewardType.item.value))
                schedule = [
                     {'obj':cellObj,
                    'tick_method':"tick",
                    'kwargs':{

                    }},
                ]

                # 在cellstepper中进行tick，tick过程中对所有outconn进行尝试激活，创建event，并在records中获取最新的link和branch
                self.cellStepper.add(key=cellID,schedule=schedule) 
            else:
                cellObj.actuator_output_signal = None




        ##### Apply mentor control commands (A, V, LStim, LState, LPain, Reward)
        
        # Apply auditory signals
        # cast alphabet to 26 bits binary list, pos 0 for A, 1 for B, 2 for C, ...., and if any letter is not in the command, set all bits to 0.
        if self.A_cmd_char_index.__len__() != self.A_OrganCellsNum:
            raise ValueError("A_cmd_char_index and A_OrganCellsNum should have the same length")
        
        A_command = [0] * self.A_OrganCellsNum

        # Check if the command exists and is not None
        if mentor_control_command.get(CommandAttr.A.value, None) is not None:
            # Get the letter (ensure it's a valid single alphabetic character)
            letter = mentor_control_command[CommandAttr.A.value]
            
            if letter.isalpha() and len(letter) == 1:  # Ensure it's a valid single letter
                # Set the corresponding bit to 1 
                A_command[self.A_cmd_char_index[letter]] = 1
            else:
                # raise error
                raise ValueError(f"Invalid letter: {letter}")

        for i, cellID in enumerate(self.A_OrganCellsList):
            cellObj = self.cells[cellID]
            # Apply auditory signals
            if A_command[i] is not None and A_command[i] > 0:
                cellObj.receptor_input(timestamp=timestamp,value=A_command[i])
                schedule = [
                     {'obj':cellObj,
                    'tick_method':"tick",
                    'kwargs':{

                    }},
                ]

                # 在cellstepper中进行tick，tick过程中对所有outconn进行尝试激活，创建event，并在records中获取最新的link和branch
                self.cellStepper.add(key=cellID,schedule=schedule) 

                # cellObj.p_prop_working_internal = A_command[i]*1.0 # organ cells provide 100% of their working capital and some reward to downstream cells
            else:
                cellObj.receptor_input_signal = None

        # Apply visual signals
        baby_visual_state_8bits = self._convert_to_binary(mentor_control_command.get(CommandAttr.V.value, [0]*4))
        for i, cellID in enumerate(self.V_OrganCellsList):
            cellObj = self.cells[cellID]
            # Apply visual signals
            if baby_visual_state_8bits[i] is not None and baby_visual_state_8bits[i] > 0:
                cellObj.receptor_input(timestamp=timestamp,value=baby_visual_state_8bits[i])
                schedule = [
                     {'obj':cellObj,
                    'tick_method':"tick",
                    'kwargs':{

                    }},
                ]

                # 在cellstepper中进行tick，tick过程中对所有outconn进行尝试激活，创建event，并在records中获取最新的link和branch
                self.cellStepper.add(key=cellID,schedule=schedule) 
            else:
                cellObj.receptor_input_signal = None

                # cellObj.p_prop_working_internal = baby_visual_state_8bits[i]*1.0 # organ cells provide 100% of their working capital and some reward to downstream cells

        # Apply limbic stimulus signals
       
        # 这里我要考虑是否判负吸引创建链接，但也可能不需要
        # baby_limbic_stimulus_state_8bits = self._convert_to_binary(mentor_control_command.get(CommandAttr.LSTIM.value, [0]*4))
        # for i, cellID in enumerate(self.LStim_OrganCellsList):
        #     cellObj = self.cells[cellID]
        #     # Apply limbic stimulus signals
        #     if baby_limbic_stimulus_state_8bits[i] is not None and baby_limbic_stimulus_state_8bits[i] >= 0:
        #         cellObj.f = baby_limbic_stimulus_state_8bits[i]

        # # Apply reward signals (external feedback here, maybe we need further improvements)
        
        # 这里估计也不需要
        # reward_signal = [mentor_control_command.get(CommandAttr.REWARD_SIZE.value,0)]*8
        # for i, cellID in enumerate(self.Reward_OrganCellsList):
        #     cellObj = self.cells[cellID]
        #     # Apply reward signals
        #     if reward_signal[i] is not None and reward_signal[i] >= 0:
        #         cellObj.e = 1
        #         # cellObj.p_prop_working_internal = reward_signal[i]*1.0


        # 不需要通过此信号来进行强化
        # # Apply internal feedback signals(pain)
        # internal_feedback_8bits = internal_feedback
        # for i, cellID in enumerate(self.LPain_OrganCellsList):
        #     cellObj = self.cells[cellID]
        #     # Apply internal feedback signals
        #     if internal_feedback_8bits[i] is not None and internal_feedback_8bits[i] >= 0:
        #         cellObj.e = internal_feedback_8bits[i]  
        #         # cellObj.q_prop_working_external = internal_feedback_8bits[i]*1.0
        
        
        
        pass

    def _updateConnections(self, time: int):
        """
        This is where you'd do partial gating logic, 
        or converting false connections -> true if conditions are met, etc.
        """
        for connID, connObj in self.connections.items():
            # For advanced usage, call connObj.tick(time)
            connObj.tick(timestamp= time, babyObj = self)
            pass

    def _updateCells(self, time: int):
        """
        Gathers connectionSnapshots for each cell, calls cell.tick(...) 
        with those snapshots.
        """
        connectionSnapshotsPerCell = {}
        for connID, connObj in self.connections.items():
            connSnap = None
            if time in connObj._validTimestamps:
                connSnap = connObj.getSnapshot(time)
            if connSnap is not None:
                upID = connObj.upstream_cell_id
                downID = connObj.downstream_cell_id
                connectionSnapshotsPerCell.setdefault(upID, []).append((connID, connSnap))
                connectionSnapshotsPerCell.setdefault(downID, []).append((connID, connSnap))

        for cid, cellObj in self.cells.items():
            connSnaps = connectionSnapshotsPerCell.get(cid, [])
            cellObj.tick(timestamp=time, dt=1.0, connectionSnapshots=connSnaps)

        

    def _deliverMessages(self):
        """
        Collect and deliver messages from all Cells and Connections' outboxes
        to the appropriate recipients' inboxes.
        """
        # Collect all messages
        all_messages = []
        
        # Gather messages from Cells
        for cell in self.cells.values():
            all_messages.extend(cell.outbox)
            cell.outbox.clear()
            
        # Gather messages from Connections
        for conn in self.connections.values():
            all_messages.extend(conn.outbox)
            conn.outbox.clear()
            
        # Deliver messages to recipients
        for msg in all_messages:
            if msg.recipient_type == RoleType.CELL.value:
                recipient = self.cells.get(msg.recipient)
            elif msg.recipient_type == RoleType.CONNECTION.value:
                recipient = self.connections.get(msg.recipient)
                
            if recipient:
                recipient.inbox.append(msg)

    def _processChangeRequests(self):
        for event in self._changeRequests:
            etype = event[0]
            if etype == "DIVISION":
                _, cellID, newCellParams = event
                self._handleDivision(cellID, newCellParams)
            elif etype == "APOPTOSIS":
                _, cellID = event
                self._handleApoptosis(cellID)
            elif etype == "CONNECTION_CREATE":
                _, sourceCellID, targetCellID = event
                self._createConnection(sourceCellID, targetCellID, isTrueConnection=True)
            elif etype == "CONNECTION_DISCONNECT":
                _, connectionObj = event
                self._handleConnectionDisconnection(connectionObj)
        self._changeRequests.clear()

    # ----------------------------------------------------------------
    # Requests from Cells
    # ----------------------------------------------------------------
    def requestDivision(self, cellID: str, newCellParams: dict):
        self._changeRequests.append(("DIVISION", cellID, newCellParams))

    def requestApoptosis(self, cellID: str):
        self._changeRequests.append(("APOPTOSIS", cellID))

    def requestConnectionCreation(self, sourceCellID: str, targetCellID: str):
        self._changeRequests.append(("CONNECTION_CREATE", sourceCellID, targetCellID))

    def requestConnectionDisconnection(self, connectionObj):
        self._changeRequests.append(("CONNECTION_DISCONNECT", connectionObj))

    # ----------------------------------------------------------------
    # Structural handlers
    # ----------------------------------------------------------------

    def _handleDivision(self, cellID: int, newCellParams: dict, invalidate_path_threshold: float = 0):
        if cellID not in self.cells:
            return
        parentCell = self.cells[cellID]

        if parentCell.divisionCountDown > 2:
            parentCell.force_liquidation()
            parentCell.divisionCountDown -= 1
            return

        newID = f"{cellID}_child_{len(self.cells)}"
        c = Cell(
            cellID=newID,
            canDivide=True,
            canDie=True,
            isOrganCell=False,
            parentBaby=self
        )
        # share some resources
        portion = newCellParams.get("portion", 0.3)
        
        c.q_acc_max = parentCell.q_acc_max * portion
        parentCell.q_acc_max *= (1 - portion)
        
        c.p_saving_max = parentCell.p_saving_max * portion
        parentCell.p_saving_max *= (1 - portion)

        # c.p_intg_reward = parentCell.p_intg_reward * portion
        # parentCell.p_intg_reward *= (1 - portion)

        c.p_intg_saving = parentCell.p_intg_saving * portion
        parentCell.p_intg_saving *= (1 - portion)

        c.alpha = parentCell.alpha * portion
        parentCell.alpha *= (1 - portion)


        # share some connections (Out)
        for connObj in parentCell.connectionsOut[:]:
            if connObj.path.alpha < invalidate_path_threshold:
                # Remove this connection
                connID = connObj.get_conn_id()
                if connID in self.connections:
                    del self.connections[connID]
                upC = self.cells.get(connObj.upstream_cell_id, None)
                if upC:
                    upC.connectionsOut = [co for co in upC.connectionsOut if co is not connObj]
                downC = self.cells.get(connObj.downstream_cell_id, None)
                if downC:
                    downC.connectionsIn = [co for co in downC.connectionsIn if co is not connObj]

            if random.random() < portion:
                parentCell.connectionsOut.remove(connObj) 
                connObj.upstream_cell_id = newID
                c.connectionsOut.append(connObj)
        
        # share some connections (In)
        for connObj in parentCell.connectionsIn[:]:
            # if connObj.path.alpha < invalidate_path_threshold:
                # Remove this connection
            connID = connObj.get_conn_id()
            if connID in self.connections:
                del self.connections[connID]
            upC = self.cells.get(connObj.upstream_cell_id, None)
            if upC:
                upC.connectionsOut = [co for co in upC.connectionsOut if co is not connObj]
            downC = self.cells.get(connObj.downstream_cell_id, None)
            if downC:
                downC.connectionsIn = [co for co in downC.connectionsIn if co is not connObj]
            if random.random() < portion:
                parentCell.connectionsIn.remove(connObj) 
                connObj.downstream_cell_id = newID
                c.connectionsIn.append(connObj)

        # add cell to baby                
        self.cells[newID] = c

        # Create fake connections between parent and child
        self._createConnection(c.cellID, parentCell.cellID, isTrueConnection=True)


        # partial domain assignment
        self._assignOneCellDomain(newID)

    def _assignOneCellDomain(self, newID: int):
        """
        Similar to _assignInitialDomain, but for a single newly created cell.
        We also fill only up to initDomainFraction of maxDomain, 
        and exclude full-domain cells from candidates.
        """
        cellObj = self.cells[newID]
        initTarget = int(self.maxDomain * self.initDomainFraction)
        currentCount = self._countCellDomain(cellObj)
        needed = initTarget - currentCount
        if needed <= 0:
            return

        allCellIDs = list(self.cells.keys())
        random.shuffle(allCellIDs)

        potentialNeighbors = [
            ocid for ocid in allCellIDs
            if ocid != newID and self._countCellDomain(self.cells[ocid]) < self.maxDomain
        ]
        random.shuffle(potentialNeighbors)

        selected = potentialNeighbors[:needed]
        for nbrID in selected:
            if not self._connectionExists(newID, nbrID):
                self._createConnection(newID, nbrID, isTrueConnection=True)

    def _handleApoptosis(self, cellID: str):
        if cellID not in self.cells:
            return
        parentCell = self.cells[cellID]

        if parentCell.apoptosisCountDowm > 2:
            parentCell.force_liquidation()
            parentCell.apoptosisCountDowm -= 1
            return

        # Remove all related connections
        cellObj = self.cells[cellID]

        for connObj in list(cellObj.connectionsIn):
            cID = connObj.get_conn_id()
            if cID in self.connections:
                del self.connections[cID]
            upC = self.cells.get(connObj.upstream_cell_id, None)
            if upC:
                upC.connectionsOut = [co for co in upC.connectionsOut if co is not connObj]

        for connObj in list(cellObj.connectionsOut):
            cID = connObj.get_conn_id()
            if cID in self.connections:
                del self.connections[cID]
            downC = self.cells.get(connObj.downstream_cell_id, None)
            if downC:
                downC.connectionsIn = [co for co in downC.connectionsIn if co is not connObj]

        del self.cells[cellID]

    def _handleConnectionDisconnection(self, connectionObj: "Connection"):
        connID = connectionObj.get_conn_id()
        if connID in self.connections:
            upC = self.cells.get(connectionObj.upstream_cell_id)
            downC = self.cells.get(connectionObj.downstream_cell_id)
            if upC:
                upC.connectionsOut = [co for co in upC.connectionsOut if co is not connectionObj]
            if downC:
                downC.connectionsIn = [ci for ci in downC.connectionsIn if ci is not connectionObj]
            del self.connections[connID]


    def getConnection(self, connID: str) -> "Connection":
        return self.connections.get(connID, None)
    def getCell(self, cellID: str) -> "Cell":
        return self.cells.get(cellID, None)
    
    def getConnectionSnapshot(self, connID: int, timestamp: int) -> Connection:
        connObj = self.connections.get(connID, None)
        if connObj:
            return connObj.getSnapshot(timestamp)
        else:
            raise ValueError(f"Connection {connID} not found.")
        
    def getCellSnapshot(self, cellID: int, timestamp: int) -> Dict[str, Any]:
        cellObj = self.cells.get(cellID, None) # "Cell"
        if cellObj:
            return cellObj.getSnapshot(timestamp)
        else:
            raise ValueError(f"Cell {cellID} not found.")
    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    def __repr__(self):
        return (
            f"Baby(numCells={len(self.cells)}, numConnections={len(self.connections)}, "
            f"currentTime={self.currentTime}, maxDomain={self.maxDomain}, "
            f"initDomainFraction={self.initDomainFraction}, config={self.config})"
        )
