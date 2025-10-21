from __future__ import annotations
from ConstantEnums import *
from EventTable import EventTable



def init_connection_constraints(singal_type,ts,k,b)->dict:
    constraints = {
        'signal_type':singal_type,
        'ts': ts,
        'k': k,
        'b':b
    }

    return constraints


def connection_logic_rules(timestamp: int = None,conn_id:int = None, propagation_manager:object = None, status: dict = None, constraints:dict = None, last_status:dict = None, links:list[dict] = None, eventTable:EventTable = None,temporal_contextural_view = 10, **kwargs)-> dict:
    """
    This function generate update connection status and constraints based on currect connection constraints, last connection status and cell status.
    @param status: current connection status, mutable reference
    @param constraints: current connection constraints, mutable reference
    @param last_status: last connection status, immutable reference
    @param last_upCell_status: last upstream cell status, immutable reference
    @param last_downCell_status: last downstream cell status, immutable reference
    @param events: all events, mutable list reference, must be non-None
    @param events
    @temporal_contextural_view: define a interval that helps a event to search for other temporal related events
    """
    # 0) Check arguments and iniyalize event
    if eventTable == None:
        raise Exception("eventTable must be non-None")
    if timestamp == None:
        raise Exception("timestamp must be non-None")
    
    if propagation_manager == None or not propagation_manager.__getattribute__("cascade_activation"):
        raise Exception("Cannot get activator")
    cascade_activation:function = propagation_manager.__getattribute__("cascade_activation")

    
    
    
    requests = []
    # 1) If the connection is not enabled, do nothing
    if status == None or status == {}:
        return None
    if status[ConnectionStatus.ENABLED.value] == False:
        return
    
    # 1) Tackle with special situation
    event = None
    # Receptor input event (Special case)
    if kwargs.get("receptor_input_signal"):

        receptor_input_signal = kwargs.get("receptor_input_signal")
        event = eventTable.create_event(start_ts = receptor_input_signal["timestamp"], progress = Event.Down_End_Attenuation.value.PROGRESS.value)
        event[Event.IS_RECEPTOR_INPUT_EVENT.value] = True
        event[Event.CONN_ID.value] = conn_id
        event[Event.SIGNAL.value] = receptor_input_signal["signal_type"]
        event[Event.Down_End_Attenuation.value.item.value] = {}
        event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X0.value] = receptor_input_signal["timestamp"]
        event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y0.value] = receptor_input_signal["value"]
        event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = receptor_input_signal["timestamp"]
        event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = receptor_input_signal["value"]
        eventTable.update_event(event)


        # 完善link
        link_tmp = {}
        event[Event.LINK.value] = link_tmp
        link_tmp[Link.EVENT.value] = event
        link_tmp[Link.Condition.value.item.value] = {"y0":receptor_input_signal["value"],"activation_strengths":None, Link.Condition.value.BRANCH.value:None}
        from ParamGroup import BranchInfo
        new_branch = BranchInfo(links=[link_tmp])
        link_tmp[Link.Condition.value.item.value][Link.Condition.value.BRANCH.value] = new_branch
        links = [link_tmp]

        # 无触发事件需要手动向baby注册此输入事件link的branch
        baby = kwargs.get("baby")
        register_branchInfo = baby.__getattribute__("register_branchInfo")
        register_branchInfo(new_branch)



        # 完善status
        status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OTHER.value


        # # send trigger requests
        # request = {}
        # request[Request.RequestType.value.item.value] = Request.RequestType.value.CHAIN_EVENT.value
        # request[Request.INITIATOR_CONN_ID.value] = status[ConnectionStatus.ID.value]
        # request[Request.INITIATOR_EVENT_ID.value] = event[Event.ID.value]
        # request[Request.TRIGGER_SIGNALS.value] = [Signal_E.E.value, Signal_F.F.value]
        # requests.append(request)


        # end the event per timestamp
        event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = receptor_input_signal["timestamp"] + 1
        event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = receptor_input_signal["value"]
        event[Event.END_TIMESTAMP.value] = receptor_input_signal["timestamp"] + 1 
        event[Event.FINISHED.value] = True
        eventTable.update_event(event)

        return {"requests":requests}


    
    # 2) Apply different rules based on connection drive 
    if constraints["signal_type"] == None:
        return None
    
    # 3.1) Signal_E logic
    if constraints["signal_type"] == Signal_E.E.value:

        # 0) Special Situation
        # Actuator Output (Special case)
        if kwargs.get("actuator_output_signal"):
            actuator_output_signal = kwargs.get("actuator_output_signal")
            activation_result = cascade_activation(timestamp, links, **kwargs)
            activated, matched_group, activation_branch = activation_result
            if activated:
                # 激活则E信号不为0，可以反向传播
                event = eventTable.create_event(start_ts = timestamp, progress = Event.Down_End_Attenuation.value.PROGRESS.value)
                event[Event.CONN_ID.value] = conn_id
                event[Event.SIGNAL.value] = Signal_E.E.value
                event[Event.Down_End_Attenuation.value.item.value] = {}
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X0.value] = actuator_output_signal["timestamp"]
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y0.value] = 0.01
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = actuator_output_signal["timestamp"]
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = 0.01

                eventTable.update_event(event)

                # 无需完善link，正常激活的conn会自动创建link

                # 完善status
                status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OTHER.value

                # send trigger requests
                request = {}
                request[Request.RequestType.value.item.value] = Request.RequestType.value.CHAIN_EVENT.value
                request[Request.INITIATOR_CONN_ID.value] = status[ConnectionStatus.ID.value]
                request[Request.INITIATOR_EVENT_ID.value] = event[Event.ID.value]
                request[Request.TRIGGER_SIGNALS.value] = [Signal_E.E.value, Signal_F.F.value]
                requests.append(request)

                # send propagation request
                request_propagation = {}
                request_propagation[Request.RequestType.value.item.value] = Request.RequestType.value.PROPAGATION.value
                request_propagation[Request.INITIATOR_CONN_ID.value] = status[ConnectionStatus.ID.value]
                request_propagation[Request.INITIATOR_EVENT_ID.value] = event[Event.ID.value]
                request_propagation[Request.TEMPORAL_CONTEXTUAL_VIEW.value] = temporal_contextural_view
                if actuator_output_signal[RewardType.item.value] == RewardType.POSITIVE.value:
                    request_propagation[Request.REWARD_TYPE.value] = RewardType.POSITIVE.value if actuator_output_signal["validation_value"]>0 else RewardType.NEGATIVE.value
                elif actuator_output_signal[RewardType.item.value] == RewardType.NEGATIVE.value:
                    request_propagation[Request.REWARD_TYPE.value] = RewardType.NEGATIVE.value if actuator_output_signal["validation_value"]<=0 else RewardType.PENDING.value #TODO 设置为Pending是为了避免反向传播时覆盖路径，但也可能有负面效果，是一个可能的修改点

                requests.append(request_propagation)

                # end the event per timestamp
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = actuator_output_signal["timestamp"] + 1
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y0.value]
                event[Event.END_TIMESTAMP.value] = actuator_output_signal["timestamp"] + 1 
                event[Event.FINISHED.value] = True

                eventTable.update_event(event)

            else:
                # 如果没有激活，但是却期望激活，那么同样创建事件，信号类型是F，虽然无法反向传播，但能吸引创建正向链接。（由于未激活，所以无法反向传播）
                event = eventTable.create_event(start_ts = timestamp, progress = Event.Down_End_Attenuation.value.PROGRESS.value)
                event[Event.CONN_ID.value] = conn_id
                event[Event.SIGNAL.value] = Signal_F.F.value
                event[Event.Down_End_Attenuation.value.item.value] = {}
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X0.value] = actuator_output_signal["timestamp"]
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y0.value] = 1.00
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = actuator_output_signal["timestamp"]
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = 1.00

                eventTable.update_event(event)


                # 完善status
                status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OTHER.value


                # 完善link
                from ParamGroup import BranchInfo
                link_tmp = {}
                event[Event.LINK.value] = link_tmp
                link_tmp[Link.EVENT.value] = event
                link_tmp[Link.Condition.value.item.value] = {"y0":1.0,"activation_strengths":{}, Link.Condition.value.BRANCH.value:None}
                new_branch = BranchInfo(links=[link_tmp])
                link_tmp[Link.Condition.value.item.value][Link.Condition.value.BRANCH.value] = new_branch

                # 无触发事件需要手动向baby注册此输入事件link的branch
                baby = kwargs.get("baby")
                register_branchInfo = baby.__getattribute__("register_branchInfo")
                register_branchInfo(new_branch)

                # send trigger requests
                request = {}
                request[Request.RequestType.value.item.value] = Request.RequestType.value.CHAIN_EVENT.value
                request[Request.INITIATOR_CONN_ID.value] = status[ConnectionStatus.ID.value]
                request[Request.INITIATOR_EVENT_ID.value] = event[Event.ID.value]
                request[Request.TRIGGER_SIGNALS.value] = [Signal_E.E.value]
                requests.append(request)

                # send propagation request
                request_propagation = {}
                request_propagation[Request.RequestType.value.item.value] = Request.RequestType.value.PROPAGATION.value
                request_propagation[Request.INITIATOR_CONN_ID.value] = status[ConnectionStatus.ID.value]
                request_propagation[Request.INITIATOR_EVENT_ID.value] = event[Event.ID.value]
                request_propagation[Request.TEMPORAL_CONTEXTUAL_VIEW.value] = temporal_contextural_view
                if actuator_output_signal[RewardType.item.value] == RewardType.POSITIVE.value:
                    request_propagation[Request.REWARD_TYPE.value] = RewardType.POSITIVE.value if actuator_output_signal["validation_value"]<=0 else RewardType.NEGATIVE.value
                elif actuator_output_signal[RewardType.item.value] == RewardType.NEGATIVE.value:
                    request_propagation[Request.REWARD_TYPE.value] = RewardType.NEGATIVE.value if actuator_output_signal["validation_value"]>0 else RewardType.PENDING.value

                requests.append(request_propagation)

                # end the event per timestamp
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = actuator_output_signal["timestamp"] + 1
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y0.value]
                event[Event.END_TIMESTAMP.value] = actuator_output_signal["timestamp"] + 1 
                event[Event.FINISHED.value] = True

                eventTable.update_event(event)
            return {"requests":requests}

        # 1) Connection Opening Logic (Triggering Phase)
        if not last_status or last_status[ConnectionStatus.GATE_STATUS.value] == GateStatus.CLOSED_READY.value:


            # 1.1) Transfer to OPENING and 
            if not event and cascade_activation(timestamp, links)[0]:


                # Up End Fires       
                if not status.get(ConnectionStatus.UP_END_SIGNAL.value):
                    status[ConnectionStatus.UP_END_SIGNAL.value] = {Signal_E.Trace.value.FATIGUE.value:0}
                    
                up_end_signal = status[ConnectionStatus.UP_END_SIGNAL.value]
                # up_end_signal[Signal_E.Constraint.value.STOCK.value] -= constraints["b"]**2/(constraints["k"]*2)
                up_end_signal[Signal_E.Trace.value.FATIGUE.value] += constraints["b"]**2/(constraints["k"]*2)


                # Transfer to OPENING and launch a new ongoing event
                status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OPENING.value
                event = eventTable.create_event(start_ts = timestamp, progress = Event.Gate_Opening_Transition.value.PROGRESS.value)
                event[Event.CONN_ID.value] = conn_id
                event[Event.SIGNAL.value] = Signal_E.E.value
                event[Event.Gate_Opening_Transition.value.item.value] = {}
                event[Event.Gate_Opening_Transition.value.item.value][LinearEvent.X0.value] = timestamp

                eventTable.update_event(event)

                # 完善link
                link_tmp = links[0]
                event[Event.LINK.value] = link_tmp
                link_tmp[Link.EVENT.value] = event
                

                # send trigger requests
                request = {}
                request[Request.RequestType.value.item.value] = Request.RequestType.value.CHAIN_EVENT.value
                request[Request.INITIATOR_CONN_ID.value] = status[ConnectionStatus.ID.value]
                request[Request.INITIATOR_EVENT_ID.value] = event[Event.ID.value]
                request[Request.TRIGGER_SIGNALS.value] = [Signal_E.E.value, Signal_F.F.value]
                request[Request.TRIGGER_EVENTS.value] = links[0][Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value].copy()
                requests.append(request)

  

        # 2) Connection Opening Logic (Transition Phase)
        if last_status and last_status[ConnectionStatus.GATE_STATUS.value] == GateStatus.OPENING.value:
            if last_status.get(ConnectionStatus.GATE_OPENING_TRANSITION.value, None) == None:
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value] = {}
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.TARGET.value] = constraints['ts']
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.RATE.value] = 1
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.PROGRESS.value] = 0
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.CYCLES.value] = 0
            gate_opening_transition:dict = status[ConnectionStatus.GATE_OPENING_TRANSITION.value]
            gate_opening_transition_threshold = gate_opening_transition[Transition.TARGET.value]
            gate_opening_transition_progress = gate_opening_transition[Transition.PROGRESS.value]
            gate_opening_transition_rate = gate_opening_transition[Transition.RATE.value]
            
            gate_opening_transition_progress += gate_opening_transition_rate
            status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.PROGRESS.value] = gate_opening_transition_progress

            # 2.1) Transfer to OPENED
            if gate_opening_transition_progress >= gate_opening_transition_threshold:
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.PROGRESS.value] = 0
                status[ConnectionStatus.GATE_OPENING_TRANSITION.value][Transition.CYCLES.value] += 1
                status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OPENED.value

                events = eventTable.search_events(progress_min=Event.Gate_Opening_Transition.value.PROGRESS.value,progress_max=Event.Gate_Opening_Transition.value.PROGRESS.value+1)
                if len(events) != 1:
                    raise Exception("Number of events at Gate_Opening_Transition should be 1")
                event = events[0]
                event[Event.Gate_Opening_Transition.value.item.value][LinearEvent.X1.value] = timestamp
                event[Event.Gate_Opened_Transition.value.item.value] = {}
                event[Event.Gate_Opened_Transition.value.item.value][LinearEvent.X0.value] = timestamp
                event[Event.PROGRESS.value] = Event.Gate_Opened_Transition.value.PROGRESS.value
                eventTable.update_event(event)


        # 3) Connection Opened Logic (Transition Phase)
        if last_status and last_status[ConnectionStatus.GATE_STATUS.value] == GateStatus.OPENED.value:
            

            if last_status.get(ConnectionStatus.GATE_OPENED_TRANSITION.value, None) == None:
                status[ConnectionStatus.GATE_OPENED_TRANSITION.value] = {}
                status[ConnectionStatus.GATE_OPENED_TRANSITION.value][Transition.CYCLES.value] = 0
            if last_status.get(ConnectionStatus.DOWN_END_SIGNAL.value,None) == None:
                status[ConnectionStatus.DOWN_END_SIGNAL.value] = {}
            status[ConnectionStatus.GATE_OPENED_TRANSITION.value][Transition.CYCLES.value] += 1
            
            


            # 3.2) Down End Spikes and Advance Event (Linear Model)
            down_end_signal = status[ConnectionStatus.DOWN_END_SIGNAL.value]
            down_end_signal[Signal_E.E.value] = constraints["b"]    # TODO: 如果是抑制，这里应该是F信号
            down_end_signal[LinearM.B.value] = constraints["b"]  
            
            events = eventTable.search_events(progress_min=Event.Gate_Opened_Transition.value.PROGRESS.value,progress_max=Event.Gate_Opened_Transition.value.PROGRESS.value+1)
            if len(events) != 1:
                raise Exception("Number of events at Gate_Opened_Transition should be 1")
            event = events[0]
            event[Event.Gate_Opened_Transition.value.item.value][LinearEvent.X1.value] = timestamp
            
            event[Event.Down_End_Attenuation.value.item.value] = {}
            event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X0.value] = timestamp
            event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y0.value] = down_end_signal[LinearM.B.value]
            event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = timestamp
            event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = down_end_signal[LinearM.B.value] # the Y1 will gradually decreas until Y1 = 0, then we can certify the X1 of this sub event

            event[Event.PROGRESS.value] = Event.Down_End_Attenuation.value.PROGRESS.value
            eventTable.update_event(event)


            # 3.3) Transfer to CLOSED_CHARGING
            status[ConnectionStatus.GATE_STATUS.value] = GateStatus.CLOSED_CHARGING.value


        # 4) Connection Closed Charging Logic (Transition Phase)
        if last_status and last_status[ConnectionStatus.GATE_STATUS.value] == GateStatus.CLOSED_CHARGING.value:

            # 4.1) Connection Closed Charging Logic (Transition Phase)
            if last_status.get(ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value, None) == None:
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value] = {}
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.CYCLES.value] = 0
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.TARGET.value] = int(constraints['ts'] + constraints["b"]/constraints["k"]) * 2
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.RATE.value] = 1
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.PROGRESS.value] = 0


            # 4.2) Up End Charges (Linear Model)
            closed_charging_transition:dict = status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value]
            closed_charging_transition_threshold = closed_charging_transition[Transition.TARGET.value]
            closed_charging_transition_progress = closed_charging_transition[Transition.PROGRESS.value]
            closed_charging_transition_rate = closed_charging_transition[Transition.RATE.value]
            
            closed_charging_transition_progress += closed_charging_transition_rate
            status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.PROGRESS.value] = closed_charging_transition_progress
            

            
            # 4.3) Transfer to CLOSED_RREADY
            if closed_charging_transition_progress >= closed_charging_transition_threshold:
                up_end_signal = status[ConnectionStatus.UP_END_SIGNAL.value]
                status[ConnectionStatus.GATE_STATUS.value] = GateStatus.CLOSED_READY.value
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.PROGRESS.value] = 0
                status[ConnectionStatus.GATE_CLOSED_CHARGING_TRANSITION.value][Transition.CYCLES.value] += 1

        
        # 5) Update Down End Connection Signal Attenuation and Related Values (Linear Model)
        if last_status and last_status[ConnectionStatus.GATE_STATUS.value] not in {GateStatus.OPENED.value, GateStatus.OTHER.value} and last_status.get(ConnectionStatus.DOWN_END_SIGNAL.value) and status.get(ConnectionStatus.DOWN_END_SIGNAL.value):
            down_end_signal = status[ConnectionStatus.DOWN_END_SIGNAL.value]
            last_down_end_signal = last_status[ConnectionStatus.DOWN_END_SIGNAL.value]
            if constraints["signal_type"] == Signal_E.E.value and last_down_end_signal[Signal_E.E.value] >0 :

                down_end_signal[Signal_E.E.value] = max(last_down_end_signal[Signal_E.E.value] - constraints["k"], 0)
                
                event = eventTable.search_events(start_max= timestamp, progress_min=Event.Down_End_Attenuation.value.PROGRESS.value,progress_max=Event.Down_End_Attenuation.value.PROGRESS.value+1)[-1]             
                event[Event.X1] = timestamp
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = timestamp
                event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = down_end_signal[Signal_E.E.value] 
                if down_end_signal[Signal_E.E.value]  == 0:
                    event[Event.END_TIMESTAMP.value] = timestamp
                eventTable.update_event(event)
            # F attenuation
            # if constraints["signal_type"] == Signal_F.F.value and last_down_end_signal[Signal_F.F.value] >0 :

            #     down_end_signal[Signal_F.F.value] = max(last_down_end_signal[Signal_F.F.value] - constraints["k"], 0)
                
            #     event = eventTable.search_events(start_max= timestamp, progress_min=Event.Down_End_Attenuation.value.PROGRESS.value,progress_max=Event.Down_End_Attenuation.value.PROGRESS.value+1)[-1]             
            #     event[Event.X1] = timestamp
            #     event[Event.Down_End_Attenuation.value.item.value][LinearEvent.X1.value] = timestamp
            #     event[Event.Down_End_Attenuation.value.item.value][LinearEvent.Y1.value] = down_end_signal[Signal_F.F.value] 
            #     eventTable.update_event(event)



        # # 6) Update Upstream and Downstream Cell Signals
        # if status.get(ConnectionStatus.UP_END_SIGNAL.value, None) != None or status.get(ConnectionStatus.DOWN_END_SIGNAL.value, None) == None:
        #     upstream_override_signal:dict = status[ConnectionStatus.UPSTREAM_OVERRIDE_SIGNAL.value]
        #     downstream_override_signal:dict = status[ConnectionStatus.DOWNSTREAM_OVERRIDE_SIGNAL.value]        
            
        #     upstream_override_signal[Signal_E.Trace.value.FATIGUE.value] = status[ConnectionStatus.UP_END_SIGNAL.value][Signal_E.Trace.value.FATIGUE.value] # The fatigue is from the consumption of E (linear growth)
        #     downstream_override_signal[Signal_E.E.value] = status[ConnectionStatus.DOWN_END_SIGNAL.value][Signal_E.E.value]
        #     downstream_override_signal[Signal_E.Trace.value.FATIGUE.value] = status[ConnectionStatus.DOWN_END_SIGNAL.value][Signal_E.Trace.value.FATIGUE.value]

        
        return {"requests":requests}





        





    # 3.2) Signal_F Logic
    if last_status and last_status[ConnectionStatus.GATE_STATUS.value] == GateStatus.OPENED.value:
        status[ConnectionStatus.GATE_STATUS.value] = GateStatus.OPENED.value

    


        
        

        


    

    pass


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NeuralModels import Cell, Baby


def single_event_backward_propagation(event:dict, baby:Baby):
    """
    
    """
    

def event_chain_backward_propagation(end_event:dict, baby:Baby):
    '''
    event中应该加入消耗，这样子就可以将消耗认为积分来进行反向传播、正向探索。反向传播采用回报R或者惩罚P信号，正向探索采用生长信号G。
    '''
    


def event_forward_propagation(start_event:dict, baby:Baby):
    ...

def event_convolution(event:dict, baby:Baby):
    '''
    这一步同时从事件链条（图特征）（同时不同连接），和事件时间维度（同链接不同数据）进行卷积
    '''
    
    ...


def cell_logic_rules(timestamp:int = None, status: dict = None, constraints:dict = None, last_status:dict = None, last_upCell_status:dict = None, last_downCell_status:dict = None):
    """
    This function generate update cell status and constraints based on currect cell constraints, last cell status and connection status.
    @param status: current cell status, mutable reference
    @param constraints: current cell constraints, mutable reference
    @param last_status: last cell status, immutable reference
    @param last_upCell_status: last upstream cell status, immutable reference
    @param last_downCell_status: last downstream cell status, immutable reference
    """
    
    






    pass


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NeuralModels import Cell, Baby
def chain_events(cell:"Cell" = None, conn_requests:Request = []):
    """
    Chain the events happening in connections when tick a cell (based on request of each connection)
    """
    for conn_request in conn_requests:
        if conn_request[Request.RequestType.value.item.value] != Request.RequestType.value.CHAIN_EVENT.value:
            continue
        init_conn = cell.parentBaby.getConnection(conn_request[Request.INITIATOR_CONN_ID.value])
        init_event = init_conn.records.get_event(conn_request[Request.INITIATOR_EVENT_ID.value])

        if conn_request.get(Request.TRIGGER_EVENTS.value, None) == None:
            ## 无触发事件（例如actuator端）            
            continue
        for resp_event in conn_request[Request.TRIGGER_EVENTS.value]:
            # Add link
            if not resp_event[Event.LINK.value].get(Link.Sequence.value.item.value):
                resp_event[Event.LINK.value][Link.Sequence.value.item.value] = {}
                resp_event[Event.LINK.value][Link.Sequence.value.item.value][Link.Sequence.value.EVENTS.value] = [init_event]
                resp_event[Event.LINK.value][Link.PathRole.value.item.value] = Link.PathRole.value.MID.value
            else:
                resp_event[Event.LINK.value][Link.Sequence.value.item.value][Link.Sequence.value.EVENTS.value].append(init_event)
                resp_event[Event.LINK.value][Link.PathRole.value.item.value] = Link.PathRole.value.MID.value

            if not init_event[Event.LINK.value].get(Link.Trigger.value.item.value):
                init_event[Event.LINK.value][Link.Trigger.value.item.value] ={}
                init_event[Event.LINK.value][Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value] = [resp_event]
                # init_event[Event.LINK.value][Link.PathRole.value.item.value] = Link.PathRole.value.END.value
            else:
                init_event[Event.LINK.value][Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value].append(resp_event)
                # init_conn[Event.LINK.value][Link.PathRole.value.item.value] = Link.PathRole.value.END.value

                
                            

            

                    

                        
                    

                        


