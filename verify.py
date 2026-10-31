"""Check captured GPU runs, command coverage, rendered pixels, and video decoding."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
closed = json.loads((root / 'closed-loop/run-result.json').read_text())
opened = json.loads((root / 'open-loop/run-result.json').read_text())
expected = {'idle': (0, 0, 0), 'throttle': (0, 1, 0), 'left-throttle': (1, 1, 0),
            'right-brake': (-1, 0, 1), 'wheel-throttle': (.2, .6, 0), 'released': (0, 0, 0)}
assert closed['completed'] and opened['completed']
assert closed['commit'] == opened['commit'] == '22b3e8678b6eb8061893109548ccf9e5d134b53d'
captures = {c['phase']: c for c in closed['captures']}
assert set(captures) == set(expected)
for phase, values in expected.items():
    state = captures[phase]['ui_state']
    assert tuple(state[k] for k in ('steer', 'throttle', 'brake')) == values
    assert state['wheel_angle'] == {'left-throttle': 90, 'right-brake': -90, 'wheel-throttle': 36}.get(phase, 0)
commands = [c for step in closed['model_commands'] for c in step['consumed_commands']]
observed = {tuple(c[k] for k in ('steer', 'throttle', 'brake')) for c in commands}
assert set(expected.values()) <= observed
assert len(closed['model_commands']) == 13
assert closed['model_commands'][-1]['consumed_commands'][-1]['throttle'] == 0
assert closed['model_commands'][-1]['consumed_commands'][-1]['brake'] == 0
poses = [s['vehicle'] for s in closed['model_commands']]
assert poses[-1]['x_m'] - poses[0]['x_m'] > 10
assert max(p['yaw_rad'] for p in poses) - min(p['yaw_rad'] for p in poses) > .01

pixels = {phase: np.asarray(Image.open(root / 'closed-loop' / capture['file'])) for phase, capture in captures.items()}
assert all(p.shape == (800, 1280, 3) and p.dtype == np.uint8 for p in pixels.values())
regions = {'wheel': (slice(373, 510), slice(975, 1110)),
           'brake': (slice(373, 510), slice(1118, 1182)),
           'throttle': (slice(373, 510), slice(1190, 1254))}
changed = {}
for name, (ys, xs) in regions.items():
    changed[name] = int(np.count_nonzero(pixels['left-throttle'][ys, xs] != pixels['right-brake'][ys, xs]))
    assert changed[name] > 100

videos = {}
for mode, result in [('closed-loop', closed), ('open-loop', opened)]:
    video = root / mode / 'drive.mp4'
    metadata = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(video)], text=True))['streams'][0]
    assert (metadata['width'], metadata['height']) == (1280, 800)
    assert int(metadata['nb_frames']) == result['written_frames']
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'], check=True, capture_output=True)
    videos[mode] = metadata

summary = {'status':'passed', 'commit':closed['commit'], 'gpu':closed['gpu'],
    'closed_loop':{'model_steps':13,'input_states':list(expected),'consumed_command_states':[list(x) for x in sorted(observed)],
        'captures':closed['captures'],'changed_rgb_values_in_control_regions':changed,'elapsed_seconds':closed['elapsed_seconds']},
    'open_loop':{'requested_generated_frames':49,'captures':opened['captures'],'elapsed_seconds':opened['elapsed_seconds'],
        'controls':'Recorded conditioning; vehicle controls intentionally inactive.'},
    'video_decoding':videos,
    'visual_inspection':'Inspected raw 1280x800 GPU-rendered captures: right panel, left/right rotation, pressed pedals, wheel input, release, and inactive recorded-trajectory UI.',
    'limitations':['One RGB view, residential scene; software-injected keyboard/wheel events, no physical controller.',
        'Interactive validation uses the normal continuous presentation mode. An additional on-demand capture exposed the existing runtime behavior that drops UI input during ticks without a new model frame.',
        'Both MP4 runs returned success and decoded completely, with an existing Bad file descriptor warning during client cleanup.'],
    'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for mode in ('closed-loop','open-loop') for p in sorted((root/mode).glob('*.png'))}}
(root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'closed_loop_states':list(expected),'rendered_pixel_changes':changed,'videos':videos},indent=2))
