"""Closed-form channel OLS with one atomic SQLite transaction per condition."""
import json,sqlite3
from pathlib import Path
import numpy as np
from .affine_math import fit

class Moments:
 def __init__(self,path,identity,policy,widths,resume=None):
  self.path=Path(path);self.db=sqlite3.connect(path);self.db.execute('PRAGMA page_size=65536');self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA cache_size=-131072')
  self.db.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT)')
  self.db.execute('CREATE TABLE IF NOT EXISTS cells(step INTEGER,unit INTEGER,mode INTEGER,n INTEGER,moments BLOB,PRIMARY KEY(step,unit))')
  self.db.execute('CREATE TABLE IF NOT EXISTS conditions(i INTEGER PRIMARY KEY,id TEXT UNIQUE,audit TEXT)')
  text=json.dumps(identity,sort_keys=True);old=self.db.execute('SELECT v FROM meta WHERE k="identity"').fetchone()
  if old is None:
   self.db.execute('INSERT INTO meta VALUES(?,?)',('identity',text))
   if resume is not None:self.inherit(resume,identity)
   self.db.commit();self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
  else:assert old[0]==text,'different statistics identity'
  self.provenance=json.loads(self.db.execute('SELECT v FROM meta WHERE k="provenance"').fetchone()[0]) if self.db.execute('SELECT 1 FROM meta WHERE k="provenance"').fetchone() else dict(inherited_conditions=0)
  self.expected={(t,u) for t,row in enumerate(policy) for u,m in enumerate(row) if m};self.policy=policy;self.widths=widths
 def inherit(self,resume,identity):
  import hashlib
  source=Path(resume['path']);h=hashlib.sha256()
  with source.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  assert h.hexdigest()==resume['SHA'],'old committed SQLite snapshot changed'
  src=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True)
  previous=json.loads(src.execute('SELECT v FROM meta WHERE k="identity"').fetchone()[0])
  assert previous['source_manifest_sha']==resume['source_manifest_SHA']
  assert {k:v for k,v in previous.items() if k!='source_manifest_sha'}=={k:v for k,v in identity.items() if k!='source_manifest_sha'}
  conditions=src.execute('SELECT i,id,audit FROM conditions ORDER BY i').fetchall()
  assert [r[0] for r in conditions]==list(range(resume['conditions']))
  assert [r[1] for r in conditions]==identity['calibration_ids'][:len(conditions)]
  digest=hashlib.sha256();count=0
  for t,u,mode,n,blob in src.execute('SELECT step,unit,mode,n,moments FROM cells ORDER BY step,unit'):
   self.db.execute('INSERT INTO cells VALUES(?,?,?,?,?)',(t,u,mode,n,blob));digest.update(json.dumps([t,u,mode,n]).encode());digest.update(blob);count+=1
  self.db.executemany('INSERT INTO conditions VALUES(?,?,?)',conditions)
  check=hashlib.sha256()
  for t,u,mode,n,blob in self.db.execute('SELECT step,unit,mode,n,moments FROM cells ORDER BY step,unit'):
   check.update(json.dumps([t,u,mode,n]).encode());check.update(blob)
  assert check.digest()==digest.digest(),'inherited moments changed'
  provenance=dict(inherited_conditions=len(conditions),previous_identity=previous,previous_SQLite=str(source),previous_SQLite_SHA=resume['SHA'],previous_GPU_exit=resume['exit_path'],positions=count,ordered_moments_SHA=digest.hexdigest(),old_audits_preserved_verbatim=True)
  self.db.execute('INSERT INTO meta VALUES(?,?)',('provenance',json.dumps(provenance,sort_keys=True)))
  src.close()
 @property
 def prefix(self):
  rows=self.db.execute('SELECT i FROM conditions ORDER BY i').fetchall();assert [r[0] for r in rows]==list(range(len(rows)));return len(rows)
 def commit(self,index,cid,delta,audit):
  assert index==self.prefix and set(delta)==self.expected,'whole actualSparse condition required'
  for (t,u),(n,v) in delta.items():assert n>0 and v.shape==(5,self.widths[u]) and v.dtype==np.float64 and np.isfinite(v).all()
  try:
   self.db.execute('BEGIN IMMEDIATE')
   for (t,u),(n,v) in delta.items():
    old=self.db.execute('SELECT mode,n,moments,rowid FROM cells WHERE step=? AND unit=?',(t,u)).fetchone()
    if old is not None:
     assert old[0]==self.policy[t][u];n+=old[1];v=v+np.frombuffer(old[2],dtype=np.float64).reshape(v.shape)
    if old is None:self.db.execute('INSERT INTO cells VALUES(?,?,?,?,?)',(t,u,self.policy[t][u],n,v.tobytes()))
    else:
     with self.db.blobopen('cells','moments',old[3]) as stream:stream.write(v.tobytes())
     self.db.execute('UPDATE cells SET n=? WHERE rowid=?',(n,old[3]))
   self.db.execute('INSERT INTO conditions VALUES(?,?,?)',(index,cid,json.dumps(audit,allow_nan=False)));self.db.commit()
  except BaseException:self.db.rollback();raise
  checkpoint=self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone();assert checkpoint[0]==0,'inactive WAL checkpoint busy'
  return dict(completed_conditions=self.prefix,positions=len(self.expected),db_bytes=self.path.stat().st_size,WAL_checkpoint=list(checkpoint))
 def coefficients(self):
  assert self.prefix==100,'all100 own trajectories required'
  rows=self.db.execute('SELECT step,unit,n,moments FROM cells ORDER BY step,unit');out={}
  for t,u,n,blob in rows:out[f'{t}:{u}']=fit(n,np.frombuffer(blob,dtype=np.float64).reshape(5,self.widths[u]))
  assert len(out)==len(self.expected);return out
 def close(self):self.db.close()
