import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Generator

import modal
import requests
import tenacity

from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.runtime.impl.eventstream.eventstream_runtime import (
    EventStreamRuntime,
    LogBuffer,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils.command import get_remote_startup_command
from openhands.runtime.utils.runtime_build import (
    BuildFromImageType,
    prep_build_folder,
)
from openhands.utils.async_utils import call_sync_from_async

# FIXME: this will not work in HA mode. We need a better way to track IDs
MODAL_RUNTIME_IDS: dict[str, str] = {}


# Modal's log generator returns strings, but the upstream LogBuffer expects bytes.
def bytes_shim(string_generator) -> Generator[bytes, None, None]:
    for line in string_generator:
        yield line.encode('utf-8')


class ModalLogBuffer(LogBuffer):
    """Synchronous buffer for Modal sandbox logs.

    This class provides a thread-safe way to collect, store, and retrieve logs
    from a Modal sandbox. It uses a list to store log lines and provides methods
    for appending, retrieving, and clearing logs.
    """

    def __init__(self, sandbox: modal.Sandbox):
        self.client_ready = False
        self.init_msg = 'Runtime client initialized.'

        self.buffer: list[str] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.log_generator = bytes_shim(sandbox.stderr)
        self.log_stream_thread = threading.Thread(target=self.stream_logs)
        self.log_stream_thread.daemon = True
        self.log_stream_thread.start()


class ModalRuntime(EventStreamRuntime):
    """This runtime will subscribe the event stream.

    When receive an event, it will send the event to runtime-client which run inside the Modal sandbox environment.

    Args:
        config (AppConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    container_name_prefix = 'openhands-sandbox-'
    sandbox: modal.Sandbox | None

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_message_callback: Callable | None = None,
        attach_to_existing: bool = False,
    ):
        assert config.modal_api_token_id, 'Modal API token id is required'
        assert config.modal_api_token_secret, 'Modal API token secret is required'

        self.config = config
        self.sandbox = None

        self.modal_client = modal.Client.from_credentials(
            config.modal_api_token_id, config.modal_api_token_secret
        )
        self.app = modal.App.lookup(
            'openhands', create_if_missing=True, client=self.modal_client
        )

        # workspace_base cannot be used because we can't bind mount into a sandbox.
        if self.config.workspace_base is not None:
            logger.warning(
                'Setting workspace_base is not supported in the modal runtime.'
            )

        # This value is arbitrary as it's private to the container
        self.container_port = 3000

        self.session = requests.Session()
        self.status_message_callback = status_message_callback
        self.base_container_image_id = self.config.sandbox.base_container_image
        self.runtime_container_image_id = self.config.sandbox.runtime_container_image
        self.action_semaphore = threading.Semaphore(1)  # Ensure one action at a time

        # Buffer for container logs
        self.log_buffer: LogBuffer | None = None

        if self.config.sandbox.runtime_extra_deps:
            logger.debug(
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}'
            )

        self.init_base_runtime(
            config,
            event_stream,
            sid,
            plugins,
            env_vars,
            status_message_callback,
            attach_to_existing,
        )

    async def connect(self):
        self.send_status_message('STATUS$STARTING_RUNTIME')

        logger.info(f'ModalRuntime `{self.sid}`')

        self.image = self._get_image_definition(
            self.base_container_image_id,
            self.runtime_container_image_id,
            self.config.sandbox.runtime_extra_deps,
        )

        if self.attach_to_existing:
            if self.sid in MODAL_RUNTIME_IDS:
                sandbox_id = MODAL_RUNTIME_IDS[self.sid]
                logger.info(f'Attaching to existing Modal sandbox: {sandbox_id}')
                self.sandbox = modal.Sandbox.from_id(
                    sandbox_id, client=self.modal_client
                )
        else:
            self.send_status_message('STATUS$PREPARING_CONTAINER')
            await call_sync_from_async(
                self._init_sandbox,
                sandbox_workspace_dir=self.config.workspace_mount_path_in_sandbox,
                plugins=self.plugins,
            )

            self.send_status_message('STATUS$CONTAINER_STARTED')

        self.log_buffer = ModalLogBuffer(self.sandbox)
        if self.sandbox is None:
            raise Exception('Sandbox not initialized')
        tunnel = self.sandbox.tunnels()[self.container_port]
        self.api_url = tunnel.url
        logger.info(f'Container started. Server url: {self.api_url}')

        if not self.attach_to_existing:
            logger.info('Waiting for client to become ready...')
            self.send_status_message('STATUS$WAITING_FOR_CLIENT')

        self._wait_until_alive()
        self.setup_initial_env()

        if not self.attach_to_existing:
            self.send_status_message(' ')

    def _get_image_definition(
        self,
        base_container_image_id: str | None,
        runtime_container_image_id: str | None,
        runtime_extra_deps: str | None,
    ) -> modal.Image:
        if runtime_container_image_id:
            base_runtime_image = modal.Image.from_registry(runtime_container_image_id)
        elif base_container_image_id:
            build_folder = tempfile.mkdtemp()
            prep_build_folder(
                build_folder=Path(build_folder),
                base_image=base_container_image_id,
                build_from=BuildFromImageType.SCRATCH,
                extra_deps=runtime_extra_deps,
            )

            base_runtime_image = modal.Image.from_dockerfile(
                path=os.path.join(build_folder, 'Dockerfile'),
                context_mount=modal.Mount.from_local_dir(
                    local_path=build_folder,
                    remote_path='.',  # to current WORKDIR
                ),
            )
        else:
            raise ValueError(
                'Neither runtime container image nor base container image is set'
            )

        return base_runtime_image.run_commands(
            """
# Disable bracketed paste
# https://github.com/pexpect/pexpect/issues/669
echo "set enable-bracketed-paste off" >> /etc/inputrc && \\
echo 'export INPUTRC=/etc/inputrc' >> /etc/bash.bashrc
""".strip()
        )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    )
    def _init_sandbox(
        self,
        sandbox_workspace_dir: str,
        plugins: list[PluginRequirement] | None = None,
    ):
        try:
            logger.info('Preparing to start container...')
            plugin_args = []
            if plugins is not None and len(plugins) > 0:
                plugin_args.append('--plugins')
                plugin_args.extend([plugin.name for plugin in plugins])

            # Combine environment variables
            environment: dict[str, str | None] = {
                'port': str(self.container_port),
                'PYTHONUNBUFFERED': '1',
            }
            if self.config.debug:
                environment['DEBUG'] = 'true'

            browsergym_args = []
            if self.config.sandbox.browsergym_eval_env is not None:
                browsergym_args = [
                    '-browsergym-eval-env',
                    self.config.sandbox.browsergym_eval_env,
                ]

            env_secret = modal.Secret.from_dict(environment)

            logger.debug(f'Sandbox workspace: {sandbox_workspace_dir}')
            sandbox_start_cmd = get_remote_startup_command(
                self.container_port,
                sandbox_workspace_dir,
                'openhands' if self.config.run_as_openhands else 'root',
                self.config.sandbox.user_id,
                plugin_args,
                browsergym_args,
            )
            logger.debug(f'Starting container with command: {sandbox_start_cmd}')
            self.sandbox = modal.Sandbox.create(
                *sandbox_start_cmd,
                secrets=[env_secret],
                workdir='/openhands/code',
                encrypted_ports=[self.container_port],
                image=self.image,
                app=self.app,
                client=self.modal_client,
                timeout=60 * 60,
            )
            MODAL_RUNTIME_IDS[self.sid] = self.sandbox.object_id
            logger.info('Container started')

        except Exception as e:
            logger.error(f'Error: Instance {self.sid} FAILED to start container!\n')
            logger.exception(e)
            self.close()
            raise e

    def close(self):
        """Closes the ModalRuntime and associated objects."""
        # if self.temp_dir_handler:
        # self.temp_dir_handler.__exit__(None, None, None)

        if self.log_buffer:
            self.log_buffer.close()

        if self.session:
            self.session.close()

        if not self.attach_to_existing and self.sandbox:
            self.sandbox.terminate()
