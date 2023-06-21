from itertools import product
from typing import Any, Dict, List


def generate_all_combination(item, key):
    def flatten(nestedList):
        if not (bool(nestedList)):
            return nestedList
        if isinstance(nestedList[0], tuple):
            return flatten(*nestedList[:1]) + flatten(nestedList[1:])

        return nestedList[:1] + flatten(nestedList[1:])

    current = item[0]
    for i in range(1, len(item)):
        current = list(product(current, item[i]))

    result = []
    for i in current:
        result.append(flatten(i))

    final_result = []
    for item in result:
        tmp = []
        for count, j in enumerate(item):
            tmp.append({key[count]: j})
        final_result.append(tmp)

    return final_result


def get_successors(all_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    possible_state = []
    for key in all_state:
        possible_state.append(all_state[key])

    return generate_all_combination(possible_state, list(all_state.keys()))
