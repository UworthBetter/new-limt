"""Explicit manifests only: never discover production arms from a directory."""
import json
from pathlib import Path
from new_gate_b_common import sha256_file


def load_manifest(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value.get('kinds'),list) or not value['kinds'] or len(set(value['kinds']))!=len(value['kinds']):
        raise SystemExit('manifest requires distinct kinds')
    if not isinstance(value.get('label'),str) or not value.get('tags'):
        raise SystemExit('manifest requires label and tags')
    tags={}
    for row in value['tags']:
        if not all(k in row for k in ('tag','kind','seed','path')):
            raise SystemExit('incomplete manifest tag')
        tag=row['tag']
        if tag in tags or row['kind'] not in value['kinds'] or row['seed'] not in (1,2,3):
            raise SystemExit('duplicate/invalid manifest tag')
        for item in (tag,row['kind'],value['label']):
            if Path(item).name!=item or item in ('.','..') or '/' in item or '\\' in item:
                raise SystemExit('unsafe manifest identifier')
        tags[tag]=row
    if any(sorted(r['seed'] for r in tags.values() if r['kind']==kind)!=value.get('seeds',{}).get(kind,[1,2,3]) for kind in value['kinds']):
        raise SystemExit('manifest seeds differ from declared seeds per kind')
    return value,tags


def h5_manifest(selection_path,scores_dir,all_tuning=False):
    from new_h5_common_v3 import selected_tags
    selection=json.loads(Path(selection_path).read_text(encoding='utf-8'))
    groups=selected_tags(selection,all_tuning=all_tuning)
    return dict(kinds=list(groups),label='h5csrank',selection_sha256=sha256_file(selection_path),
                tags=[dict(tag=tag,kind=kind,seed=int(tag.rsplit('_seed',1)[1]),
                           path=str(Path(scores_dir)/(tag+'.npz'))) for kind,tags in groups.items() for tag in tags])
