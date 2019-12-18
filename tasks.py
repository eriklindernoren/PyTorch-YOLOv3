#!/usr/bin/env python3
import glob
from fabric import Connection
from invoke import task

HOST        = 'ec2-35-155-140-208.us-west-2.compute.amazonaws.com'
USER        = 'ubuntu'
ROOT        = ''
REMOTE      = '{user}@{host}:{root}'.format(user=USER, host=HOST, root=ROOT)
VENV        = 'virtualenv'
MODEL       = 'models'
OUTPUT      = 'output'
CHECKPOINT  = 'checkpoints'
LOCAL_FILES = [
    'fonts',
    'test_inputs',

    'common.py',
    'generate_data.py',
    'generate_image.py',
    'train.py',
    'tests.py',
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
        ctx.conn.run('mkdir -p {}'.format(MODEL))
        ctx.conn.run('mkdir -p {}'.format(OUTPUT))
        ctx.conn.run('mkdir -p {}'.format(TESTS))
        ctx.conn.run('sudo apt install -y dtach')
        ctx.conn.run('python3 -m venv {}'.format(VENV))
    # PIP
    ctx.conn.put('requirements.in', remote='{}/requirements.in'.format(ROOT))
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source {}/bin/activate'.format(VENV)):
            ctx.conn.run('pip install -U pip')
            ctx.conn.run('pip install pip-tools')
            ctx.conn.run('pip-compile --upgrade requirements.in')
            ctx.conn.run('pip-sync')

@task
def push(ctx, model=''):
    ctx.run('rsync -rv {files} {remote}'.format(files=' '.join(LOCAL_FILES), remote=REMOTE))
    model = sorted([fp for fp in glob.glob('models/*') if model and model in fp], reverse=True)
    if model:
        ctx.run('rsync -rv {folder}/ {remote}/{folder}'.format(remote=REMOTE, folder=model[0]))

@task
def pull(ctx):
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=CHECKPOINT))
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=OUTPUT))

@task(pre=[connect], post=[close])
def train(ctx):
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source {}/bin/activate'.format(VENV)):
            ctx.conn.run('dtach -A /tmp/{} python train.py'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def resume(ctx):
    ctx.conn.run('dtach -a /tmp/{}'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def test(ctx, model=''):
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source {}/bin/activate'.format(VENV)):
            ctx.conn.run('python tests.py {}'.format(model), pty=True)

@task(pre=[connect], post=[close])
def clean(ctx):
    with ctx.conn.cd(ROOT):
        ctx.conn.run('rm -rf {}/*'.format(MODEL), pty=True)
