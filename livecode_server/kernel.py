"""livecode kernel to execute code in a sandbox.

Hardened variant: the per-run temp dir is the only place a request can write
(no path traversal onto the Falcon host), and the runner container has no
network, no extra capabilities, a non-root user, a read-only rootfs and a
process cap. A run that ignores the in-container `timeout` is killed from here.
"""
import asyncio
import aiodocker
from pathlib import Path
import os
import tempfile
import json
from .msgtypes import ExecMessage
from . import config

# Hard ceiling enforced from the server side, above the in-container `timeout`.
RUN_TIMEOUT = 15


class Kernel:
    def __init__(self, runtime):
        self.runtime = runtime

    async def execute(self, msg: ExecMessage):
        """Executes the code and yields the messages whenever something is printed by that code.
        """
        kspec = config.get_runtime(self.runtime)
        code_filename = msg.code_filename or kspec['code_filename']
        with tempfile.TemporaryDirectory() as root:
            self.root = root
            if msg.code:
                self.save_file(root, code_filename, msg.code)

            for f in msg.files:
                self.save_file(root, f['filename'], f['contents'])

            container = await self.start_container(
                image=kspec['image'],
                command=msg.command or kspec['command'],
                root=root,
                env=msg.env)

            # TODO: read stdout and stderr seperately
            try:
                async for line in self.read_docker_log_lines(container):
                    if line.startswith("--MSG--"):
                        json_message = line[len("--MSG--"):].strip()
                        msg = json.loads(json_message)
                        # ignore bad cases
                        if "msgtype" not in msg:
                            # TODO: print a warning message
                            continue
                    else:
                        msg = dict(msgtype="write", file="stdout", data=line)
                    yield msg
            finally:
                try:
                    status = await asyncio.wait_for(container.wait(), RUN_TIMEOUT)
                except asyncio.TimeoutError:
                    await container.kill()
                    status = {"StatusCode": 137}
                yield {"msgtype": "exitstatus", "exitstatus": status['StatusCode']}
                await container.delete(force=True)

    async def read_docker_log_lines(self, container, max_line_length=1000000):
        """Reads the docker log line by line.

        When a line is longer, the docker api gives the line in chunks.
        This function combines the chunks into a line and returns one
        line at a time as an generator.

        Also includes a protection against very long lines. if a line
        has more than max_line_length (default 1 million), reading
        is aborted and no further data is read.
        """
        logs = container.log(stdout=True, stderr=True, follow=True)

        remaining = ""

        async for line in logs:
            line = remaining + line
            remaining = ""

            if line.endswith("\n"):
                yield line
            else:
                # protection against very large images
                # stop reading further when the line has more than max_line_length
                if len(line) > max_line_length:
                    return
                remaining = line

        if remaining:
            yield remaining

    def save_file(self, root, filename, contents):
        # A request controls `filename` (files[].filename, code_filename, or a
        # multipart field name). Resolve it and refuse anything that lands
        # outside the per-run dir, so `../x` or `/etc/cron.d/x` cannot write to
        # the Falcon host.
        root_path = Path(root).resolve()
        target = (root_path / filename).resolve()
        if target.parent != root_path:
            raise ValueError(f"invalid filename: {filename!r}")
        target.write_text(contents)
        target.chmod(0o644)

    async def start_container(self, image, command, root, env):
        docker = aiodocker.Docker()
        print('== starting a container ==')
        # The runner must stay sealed off: no network (exercises read their
        # input from the `stdin` file, not the wire), no privilege gain, a
        # read-only rootfs, a non-root user and a process cap. /app stays
        # writable (compiled runtimes build there), so widen the per-run dir
        # for the non-root container user.
        os.chmod(root, 0o777)
        env_entries = [f'{k}={v}' for k, v in env.items()]
        config = {
            'Cmd': command,
            'Image': image,
            'User': '65534:65534',
            'Env': ["PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1"] + env_entries,
            'HostConfig': {
                'Binds': [
                    root + ":/app"
                ],
                'NetworkMode': 'none',
                'CapDrop': ['ALL'],
                'SecurityOpt': ['no-new-privileges'],
                'ReadonlyRootfs': True,
                'Tmpfs': {'/tmp': 'rw,size=16m,noexec,nosuid'},
                'PidsLimit': 64,
                "Memory": 100*1024*1024,
                "MemorySwap": 100*1024*1024,
                "CPUQuota": 50000,
                "CPUPeriod": 100000,
            }
        }
        container = await docker.containers.create(
            config=config
        )
        await container.start()
        return container
