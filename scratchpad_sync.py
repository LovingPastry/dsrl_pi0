import boto3, botocore, os, sys
from boto3.s3.transfer import TransferConfig
PARTIAL = "/media/fuyx/系统/dsrl_pi0_data/openpi-assets/checkpoints/pi0_libero.partial"
BUCKET="openpi-assets"; PREFIX="checkpoints/pi0_libero/"
s3 = boto3.client('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED, max_pool_connections=32))
cfg = TransferConfig(multipart_threshold=64*1024*1024, max_concurrency=16, multipart_chunksize=64*1024*1024)
p=s3.get_paginator('list_objects_v2')
objs=[]
for page in p.paginate(Bucket=BUCKET, Prefix=PREFIX):
    objs += page.get('Contents',[])
todo=[]
for o in objs:
    rel=o['Key'][len(PREFIX):]
    lp=os.path.join(PARTIAL, rel)
    if os.path.exists(lp) and os.path.getsize(lp)==o['Size']:
        continue
    todo.append((o['Key'], lp, o['Size']))
print(f"[sync] {len(objs)} objects total, {len(todo)} need download:", flush=True)
for k,lp,sz in todo:
    print(f"  -> {k}  ({sz/1e9:.2f} GB)", flush=True)
for k,lp,sz in todo:
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    print(f"[sync] downloading {k} ...", flush=True)
    s3.download_file(BUCKET, k, lp, Config=cfg)
    got=os.path.getsize(lp)
    print(f"[sync] done {k}: {got} bytes (expected {sz}) {'OK' if got==sz else 'MISMATCH!!'}", flush=True)
print("[sync] ALL DONE", flush=True)
