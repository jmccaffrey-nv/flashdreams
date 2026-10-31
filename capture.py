"""Capture real model runs and the GPU-rendered driving interface."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/jmccaffrey/projects/flashdreams-restore-drive-controls')
OUT = Path(__file__).resolve().parent
paths = [ROOT / 'flashdreams', ROOT / 'apps/interactive_drive',
         ROOT / 'integrations_v2/cosmos_interactive',
         ROOT / 'integrations_v2/omnidreams/impl/ludus-renderer',
         ROOT / 'integrations_v2']
sys.path[:0] = list(map(str, paths))
os.environ['PYTHONPATH'] = os.pathsep.join(map(str, paths))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL'] = '1'
os.environ['CUDA_HOME'] = '/tmp/flashdreams-lidar-interactive-cuda/nvidia/cu13'
os.environ['PATH'] = os.environ['CUDA_HOME'] + '/bin:' + str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = os.environ['CUDA_HOME'] + '/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['MAX_JOBS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.pop('COSMOS_INTERACTIVE_CHECKPOINT', None)

mode = sys.argv[1]
if mode == 'help':
    from flashdreams.runtime_v2.cli import entrypoint
    entrypoint(sys.argv[2:])
    raise SystemExit

import numpy as np
import torch
from PIL import Image
from flashdreams.runtime_v2.cli import entrypoint
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.video_encoder import Mp4Encoder
from flashdreams.runtime_v2.user_input_event import KeyboardUserInputEvent, KeyboardInputState, GameWheelUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from interactive_drive.core import InteractiveDriveModelLoop
import interactive_drive.app as drive_app
import cosmos3_nano_sim_auto.apps.interactive_drive.ui as drive_ui

destination = OUT / mode
destination.mkdir(exist_ok=True)
record = {'mode': mode, 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'argv': sys.argv[2:], 'python': sys.executable,
          'source_modules': {m.__name__: m.__file__ for m in (drive_app, drive_ui)},
          'inputs': [], 'model_commands': [], 'captures': [], 'ui_states': []}
assert all(Path(p).is_relative_to(ROOT) for p in record['source_modules'].values())
frame_count = 0
current_phase = None
held_keys = set()
captured_phases = set()
ui_snapshot = None
consumed_commands = []


def phase_for_steps(index):
    return ('idle' if index < 2 else 'throttle' if index < 4 else
            'left-throttle' if index < 6 else 'right-brake' if index < 8 else
            'wheel-throttle' if index < 10 else 'released')


original_encode = Mp4Encoder.write
def encode(self, frames):
    global frame_count
    if mode == 'closed-loop' and (ui_snapshot is None or ui_snapshot['block'] is None):
        return
    original_encode(self, frames)
    for offset, frame in enumerate(frames):
        index = frame_count + offset
        phase = current_phase if mode == 'closed-loop' else 'recorded'
        expected = {'idle': (0, 0, 0), 'throttle': (0, 1, 0), 'left-throttle': (1, 1, 0),
                    'right-brake': (-1, 0, 1), 'wheel-throttle': (.2, .6, 0), 'released': (0, 0, 0)}
        matches = mode != 'closed-loop' or tuple(ui_snapshot[k] for k in ('steer','throttle','brake')) == expected[phase]
        if phase not in captured_phases and index >= 6 and matches:
            name = f'{index:04d}-{phase}.png'
            Image.fromarray(frame).save(destination / name)
            record['captures'].append({'frame': index, 'phase': phase, 'file': name, 'size': list(frame.shape), 'ui_state': ui_snapshot})
            captured_phases.add(phase)
            print('Captured', name, flush=True)
    frame_count += len(frames)


original_inputs = Mp4ClientWindow.get_user_input_events
def inputs(self):
    global held_keys, current_phase
    existing = original_inputs(self).get_events()
    if mode != 'closed-loop':
        return UserInputEvents(existing)
    phase = phase_for_steps(len(record['model_commands']))
    if phase == current_phase:
        return UserInputEvents(existing)
    current_phase = phase
    wanted = {'throttle': {'w'}, 'left-throttle': {'w', 'a'}, 'right-brake': {' ', 'd'}}.get(phase, set())
    timestamp = np.uint64(time.monotonic_ns())
    for key in sorted(held_keys - wanted):
        existing.append(KeyboardUserInputEvent(timestamp=timestamp, key=key, state=KeyboardInputState.RELEASED))
    for key in sorted(wanted - held_keys):
        existing.append(KeyboardUserInputEvent(timestamp=timestamp, key=key, state=KeyboardInputState.PRESSED))
    held_keys = wanted
    if phase == 'wheel-throttle':
        existing.append(GameWheelUserInputEvent(timestamp=timestamp, steering=-0.2, throttle=0.6, brake=0.0))
    elif phase == 'released':
        existing.append(GameWheelUserInputEvent(timestamp=timestamp, action='disconnected'))
    record['inputs'].append({'frame': frame_count, 'phase': phase, 'keys': sorted(wanted)})
    print('Input', record['inputs'][-1], flush=True)
    return UserInputEvents(existing)


original_step = InteractiveDriveModelLoop.step
original_command = InteractiveDriveModelLoop._command
def command(self, *args, **kwargs):
    value = original_command(self, *args, **kwargs)
    consumed_commands.append({'steer': value.steer, 'throttle': value.throttle, 'brake': value.brake, 'reverse': value.reverse})
    return value


def step(self, step_index, events):
    start = len(consumed_commands)
    result = original_step(self, step_index, events)
    used = consumed_commands[start:]
    item = {'step': step_index, 'consumed_commands': used,
            'vehicle': {key: getattr(self.state.vehicle, key) for key in ('x_m', 'y_m', 'yaw_rad', 'speed_mps')} if self.state.vehicle is not None else None}
    record['model_commands'].append(item)
    print('Model command', item, flush=True)
    return result


original_hud = drive_ui.MultiviewDriveUILoop._draw_drive_hud
def hud(self, imgui):
    global ui_snapshot
    original_hud(self, imgui)
    cmd = self.state.drive_input.command()
    ui_snapshot = {'steer': cmd.steer, 'throttle': cmd.throttle, 'brake': cmd.brake,
                   'source': self.state.drive_input.source(), 'wheel_angle': self.state.wheel_cache_angle,
                   'block': self.state.telemetry.blocks_generated if self.state.telemetry is not None else None}
    if not record['ui_states'] or any(ui_snapshot[k] != record['ui_states'][-1][k] for k in ('steer','throttle','brake','source')):
        record['ui_states'].append(dict(ui_snapshot, frame=frame_count))


Mp4Encoder.write = encode
Mp4ClientWindow.get_user_input_events = inputs
InteractiveDriveModelLoop.step = step
InteractiveDriveModelLoop._command = command
drive_ui.MultiviewDriveUILoop._draw_drive_hud = hud
started = time.monotonic()
try:
    entrypoint(sys.argv[2:])
    record['completed'] = True
except BaseException as exc:
    record['completed'] = False
    record['error'] = repr(exc)
    raise
finally:
    record['elapsed_seconds'] = time.monotonic() - started
    record['written_frames'] = frame_count
    record['gpu'] = torch.cuda.get_device_name(0)
    record['torch'] = torch.__version__
    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
    (destination / 'run-result.json').write_text(json.dumps(record, indent=2) + '\n')
