#!/usr/bin/env python3
import glob
from fabric import Connection
from invoke import task

HOST        = 'ec2-34-220-228-32.us-west-2.compute.amazonaws.com'
USER        = 'ubuntu'
ROOT        = 'cash'
REMOTE      = '{user}@{host}:{root}'.format(user=USER, host=HOST, root=ROOT)
VENV        = 'venv'
MODEL       = 'models'
OUTPUT      = 'output'
CHECKPOINT  = 'checkpoints'
LOCAL_FILES = [
    'checkpoints',
    'config',
    'data',
    'detect.py',
    'logs',
    'output',
    'train.py',
    'tests.py',
    'utils',
    'weights',
    'models.py',
    'test.py'
]

@task
def connect(ctx):
    ctx.conn = Connection(host=HOST, user=USER)

@task
def close(ctx):
    ctx.conn.close()

@task(pre=[connect], post=[close])
def setup(ctx):
    ctx.conn.run('mkdir -p {}'.format(ROOT))
    with ctx.conn.cd(ROOT):
        # ctx.conn.run('mkdir -p {}'.format(MODEL))
        # ctx.conn.run('mkdir -p {}'.format(OUTPUT))
        # ctx.conn.run('mkdir -p {}'.format(TESTS))
        ctx.conn.run('sudo apt-get install python3-venv')
        ctx.conn.run('sudo apt-get install dtach -y')
        ctx.conn.run('python3 -m venv {}'.format(VENV))
    # PIP
    ctx.conn.put('requirements.in', remote='{}/requirements.in'.format(ROOT))
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source {}/bin/activate'.format(VENV)):
            ctx.conn.run('pip3 install -U pip')
            ctx.conn.run('pip3 install pip-tools')
            ctx.conn.run('pip-compile --upgrade requirements.in')
            ctx.conn.run('pip-sync')

@task
def push(ctx, model=''):
    ctx.run('rsync -rv {files} {remote}'.format(files=' '.join(LOCAL_FILES), remote=REMOTE))

@task
def pull(ctx):
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=CHECKPOINT))
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=OUTPUT))

@task(pre=[connect], post=[close])
def train(ctx):
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source {}/bin/activate'.format(VENV)):
            ctx.conn.run('dtach -A /tmp/{} python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 100  --batch 4 --pretrained_weights weights/darknet53.conv.74'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def resume(ctx):
    ctx.conn.run('dtach -a /tmp/{}'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def test(ctx, model=''):
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source {}/bin/activate'.format(VENV)):
            ctx.conn.run('python3 detect.py --image_folder data/samples/ --model_def config/yolov3-custom.cfg --weights_path checkpoints/yolov3_ckpt_63.pth --checkpoint_model checkpoints/yolov3_ckpt_63.pth --class_path data/custom/classes.names', pty=True)

@task(pre=[connect], post=[close])
def clean(ctx):
    with ctx.conn.cd(ROOT):
        ctx.conn.run('rm -rf {}/*'.format(ROOT), pty=True)

# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"
# sudo dpkg-reconfigure locales

# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8