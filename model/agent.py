from typing import Any, List, Dict

import dispatch as dispatcher
import parse
import reposition as repositioner


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """
    def __init__(self, alpha=0.06, dispatch_gamma=0.8, idle_reward=0, reposition_gamma=0.9997):
        self.dispatcher = dispatcher.Dql(alpha, dispatch_gamma, idle_reward)
        self.repositioner = repositioner.StateValueGreedy(self.dispatcher, reposition_gamma)

    def dispatch(self, dispatch_input: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = parse.parse_dispatch(dispatch_input)
        dispatch = self.dispatcher.dispatch(drivers, requests, candidates)
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input: Dict[str, Any]) -> List[Dict[str, str]]:
        repo_action = []
        for driver in reposition_input['driver_info']:
          # the default reposition is to let drivers stay where they are
          repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
        return repo_action