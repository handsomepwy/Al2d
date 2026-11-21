"""Hair physics: builds a physics3.json and previews what it will do.

A physics3.json is a list of pendulums.  Each takes parameters as input (how far the head is
turned moves the pendulum's anchor sideways, how far it is tilted turns gravity) and writes the
angle of its swing into an output parameter, here one of the hair sway parameters.

Simulation is my own small version of the pendulum scheme used by Live2D's published Cubism
Framework (CubismPhysics), written from how that scheme works.  It is for previews and for
choosing numbers; the app has the last word.
"""
import json
import math

# (parameter, what it does to the pendulum, weight in percent): the Editor's usual hair inputs.
HEAD_INPUTS = [('ParamAngleX', 'X', 60), ('ParamAngleZ', 'Angle', 60),
               ('ParamBodyAngleX', 'X', 40), ('ParamBodyAngleZ', 'Angle', 40)]


def pendulum(setting_id, name, output, length, delay, mobility=0.95, acceleration=1.5, scale=1.0,
             inputs=HEAD_INPUTS):
    """One pendulum of the given length that drives one parameter.

    length    longer = slower swing and a smaller angle for the same head movement
    delay     how fast time runs for it (1 = fastest); mobility: how much speed it keeps per step
    scale     output parameter = swing angle in radians x scale
    """
    return {
        'Id': setting_id,
        'Name': name,       # moved into Meta.PhysicsDictionary by physics_file()
        'Input': [{'Source': {'Target': 'Parameter', 'Id': parameter}, 'Weight': weight, 'Type': kind,
                   'Reflect': False} for parameter, kind, weight in inputs],
        'Output': [{'Destination': {'Target': 'Parameter', 'Id': output}, 'VertexIndex': 1, 'Scale': scale,
                    'Weight': 100, 'Type': 'Angle', 'Reflect': False}],
        'Vertices': [
            {'Position': {'X': 0, 'Y': 0}, 'Mobility': 1, 'Delay': 1, 'Acceleration': 1, 'Radius': 0},
            {'Position': {'X': 0, 'Y': length}, 'Mobility': mobility, 'Delay': delay,
             'Acceleration': acceleration, 'Radius': length},
        ],
        'Normalization': {'Position': {'Minimum': -10, 'Default': 0, 'Maximum': 10},
                          'Angle': {'Minimum': -10, 'Default': 0, 'Maximum': 10}},
    }


def physics_file(pendulums, fps=60):
    """The whole physics3.json as a dict."""
    settings = [{key: value for key, value in p.items() if key != 'Name'} for p in pendulums]
    return {
        'Version': 3,
        'Meta': {
            'PhysicsSettingCount': len(settings),
            'TotalInputCount': sum(len(s['Input']) for s in settings),
            'TotalOutputCount': sum(len(s['Output']) for s in settings),
            'VertexCount': sum(len(s['Vertices']) for s in settings),
            'Fps': fps,
            'EffectiveForces': {'Gravity': {'X': 0, 'Y': -1}, 'Wind': {'X': 0, 'Y': 0}},
            'PhysicsDictionary': [{'Id': p['Id'], 'Name': p['Name']} for p in pendulums],
        },
        'PhysicsSettings': settings,
    }


def save(path, data):
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(data, handle, indent='\t')


def _normalized(value, low, high, norm):
    """A parameter value mapped onto the pendulum's own range; sign flipped, as the runtime does."""
    value = min(max(value, low), high)
    middle = low + (high - low) / 2
    offset = value - middle
    if offset > 0:
        result = offset * (norm['Maximum'] - norm['Default']) / (high - middle) + norm['Default']
    elif offset < 0:
        result = offset * (norm['Minimum'] - norm['Default']) / (low - middle) + norm['Default']
    else:
        result = norm['Default']
    return -result


def _angle_between(start, end):
    angle = math.atan2(end[1], end[0]) - math.atan2(start[1], start[0])
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


class Simulation:
    """Steps the pendulums of a physics file.  ranges: {parameter id: (minimum, maximum)}."""

    AIR_RESISTANCE = 5.0
    STILL_BELOW = 0.001         # of the position range: smaller sideways offsets snap to zero

    def __init__(self, data, ranges):
        self.settings = data['PhysicsSettings']
        self.ranges = ranges
        self.strands = []
        for setting in self.settings:
            strand, height = [], 0.0
            for vertex in setting['Vertices']:
                height += vertex['Radius']
                strand.append({'position': [0.0, height], 'velocity': [0.0, 0.0], 'gravity': (0.0, 1.0)})
            self.strands.append(strand)

    def step(self, values, seconds):
        """values: {parameter id: value} for the inputs.  Returns {output parameter id: value}."""
        outputs = {}
        for setting, strand in zip(self.settings, self.strands):
            shift, tilt = 0.0, 0.0
            for source in setting['Input']:
                parameter = source['Source']['Id']
                low, high = self.ranges[parameter]
                weight = source['Weight'] / 100.0
                kind = 'Angle' if source['Type'] == 'Angle' else 'Position'
                amount = _normalized(values.get(parameter, 0.0), low, high, setting['Normalization'][kind]) * weight
                if source['Type'] == 'X':
                    shift += amount
                elif source['Type'] == 'Angle':
                    tilt += amount
            turn = math.radians(-tilt)
            anchor = [shift * math.cos(turn), shift * math.sin(turn)]
            gravity = (math.sin(math.radians(tilt)), math.cos(math.radians(tilt)))
            strand[0]['position'] = anchor
            threshold = self.STILL_BELOW * setting['Normalization']['Position']['Maximum']
            for index in range(1, len(strand)):
                vertex, particle, parent = setting['Vertices'][index], strand[index], strand[index - 1]
                last = list(particle['position'])
                delay = vertex['Delay'] * seconds * 30.0
                direction = [particle['position'][0] - parent['position'][0],
                             particle['position'][1] - parent['position'][1]]
                swing = _angle_between(particle['gravity'], gravity) / self.AIR_RESISTANCE
                direction[0] = math.cos(swing) * direction[0] - direction[1] * math.sin(swing)
                direction[1] = math.sin(swing) * direction[0] + direction[1] * math.cos(swing)
                position = [parent['position'][0] + direction[0] + particle['velocity'][0] * delay
                            + gravity[0] * vertex['Acceleration'] * delay * delay,
                            parent['position'][1] + direction[1] + particle['velocity'][1] * delay
                            + gravity[1] * vertex['Acceleration'] * delay * delay]
                offset = [position[0] - parent['position'][0], position[1] - parent['position'][1]]
                length = math.hypot(*offset) or 1.0
                position = [parent['position'][0] + offset[0] / length * vertex['Radius'],
                            parent['position'][1] + offset[1] / length * vertex['Radius']]
                if abs(position[0]) < threshold:
                    position[0] = 0.0
                if delay:
                    particle['velocity'] = [(position[0] - last[0]) / delay * vertex['Mobility'],
                                            (position[1] - last[1]) / delay * vertex['Mobility']]
                particle['position'] = position
                particle['gravity'] = gravity
            for output in setting['Output']:
                index = output['VertexIndex']
                hanging = [strand[index]['position'][0] - strand[index - 1]['position'][0],
                           strand[index]['position'][1] - strand[index - 1]['position'][1]]
                if index >= 2:
                    straight = [strand[index - 1]['position'][0] - strand[index - 2]['position'][0],
                                strand[index - 1]['position'][1] - strand[index - 2]['position'][1]]
                else:
                    straight = (0.0, 1.0)
                value = _angle_between(straight, hanging) * output['Scale']
                if output['Reflect']:
                    value = -value
                low, high = self.ranges[output['Destination']['Id']]
                outputs[output['Destination']['Id']] = min(max(value, low), high)
        return outputs
