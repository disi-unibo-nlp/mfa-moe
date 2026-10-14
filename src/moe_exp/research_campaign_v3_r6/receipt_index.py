"""A fixed metadata snapshot avoids per-cell probes of absent Lustre files."""
from pathlib import Path
from functools import lru_cache
from .contracts import Units, file_sha, load, require


class IndexedUnits(Units):
    def __init__(self, root):
        super().__init__(root)
        self.locations={};seen=set()
        def visit(folder,local=False):
            folder=Path(folder);identity=str(folder.resolve())
            if identity in seen:return
            seen.add(identity)
            results={p.name[:64]:p for p in folder.glob('*.result.json')}
            starts={}
            for p in folder.glob('*.start-*.json'):starts.setdefault(p.name[:64],[]).append(p)
            for uid in results.keys() | starts.keys():
                self.locations.setdefault(uid,[]).append((results.get(uid),sorted(starts.get(uid,[])),local))
            manifest=folder.parent/'INHERITED_UNITS.json'
            if manifest.exists():
                directories=load(manifest)['directories'].get(folder.name,[])
                if isinstance(directories,str):directories=[directories]
                for p in directories:visit(p)
        visit(self.root,True)
        self.state=lru_cache(maxsize=128 if self.root.name=='executions' else 8192)(self._state)

    def _state(self,uid):
        require(len(uid)==64 and set(uid)<=set('0123456789abcdef'),'invalid unit ID')
        for result,starts,local in self.locations.get(uid,[]):
            if result:
                value=load(result)
                if local:return value
                if value.get('status') in ('complete','reused','invalid','truncated'):
                    return {**value,'inherited_receipt':str(result),'inherited_receipt_sha256':file_sha(result)}
            if starts:
                value={'status':'started','starts':[load(p) for p in starts]}
                if not local:value['inherited_directory']=str(starts[0].parent)
                return value
        return None
