"""Credentials stay at the original location; host keys are never saved."""
import json
import re
import shlex
import threading
import time
from pathlib import Path


def connect(host, username, port=22, password=None, pkey=None, sock=None):
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, port=port, password=password, pkey=pkey,
                   sock=sock, timeout=12, banner_timeout=15, auth_timeout=20,
                   allow_agent=False, look_for_keys=False,compress=True)
    client.get_transport().set_keepalive(30)
    return client


class SiteConnection:
    def __init__(self, site):
        self.site = site
        self.jump = self.bridge = None
        self.lock = threading.Lock()

    def __enter__(self):
        import paramiko
        connection = self.site["connection"]
        source = Path(connection["credentials_directory"])
        note = (source / connection["credentials_note"]).read_text(encoding="utf-8-sig")
        password = re.search(r"passwd:\s*(\S+)", note).group(1)
        jump, bridge = connection["jump"], connection["bridge"]
        try:
            self.jump = connect(jump["host"], jump["user"], jump["port"],
                                pkey=paramiko.RSAKey.from_private_key_file(str(source/connection["key_filename"])))
            channel = self.jump.get_transport().open_channel("direct-tcpip", (bridge["host"], bridge["port"]), ("127.0.0.1", 0), timeout=12)
            self.bridge = connect(bridge["host"], bridge["user"], bridge["port"], password=password, sock=channel)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        for client in (self.bridge, self.jump):
            if client:
                client.close()

    def client(self, host):
        import os
        password = os.environ.get("IDFS_CLIENT_PASSWORD")
        if not password:
            raise ValueError("set IDFS_CLIENT_PASSWORD privately on the experiment computer")
        with self.lock:
            channel = self.bridge.get_transport().open_channel("direct-tcpip", (host, 22), ("127.0.0.1", 0), timeout=12)
        return connect(host, "hust", password=password, sock=channel)

    def storage_command(self, host, command):
        hosts = {s["management_ip"] for s in self.site["storage_nodes"]}
        if host not in hosts:
            raise ValueError("storage host is not in site inventory")
        return shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                           "-o", "UpdateHostKeys=no", "-o", "ControlMaster=no", "-S", "none",
                           "-o", "ForwardAgent=no", "-o", "ConnectTimeout=8", "idfs@"+host, command])

    def run(self, target, command, timeout=30):
        """Caller supplies only fixed query commands from this package."""
        other = None
        try:
            if target["kind"] == "oss":
                client = self.bridge
                command = self.storage_command(target["host"], command)
            else:
                other = self.client(target["host"])
                client = other
            _, out, err = client.exec_command(command, timeout=timeout)
            result = {"stdout": out.read().decode("utf-8", errors="replace"),
                      "stderr": err.read().decode("utf-8", errors="replace"),
                      "exit": out.channel.recv_exit_status()}
            return result
        finally:
            if other:
                other.close()

    def stream(self, target, config, emit, timeout):
        source = Path(__file__).with_name("remote_agent.py").read_text(encoding="utf-8")
        cmd = shlex.join(["python3", "-u", "-c", source, json.dumps(config, separators=(",", ":"))])
        other = None
        try:
            client = self.bridge
            if target["kind"] == "oss":
                cmd = self.storage_command(target["host"], cmd)
            else:
                other = self.client(target["host"])
                client = other
            _, out, err = client.exec_command(cmd, timeout=timeout)
            for line in out:
                if line.strip():
                    record = json.loads(line)
                    record["management_ip"] = target["host"]
                    record["received_local_epoch_ns"] = time.time_ns()
                    record["received_local_mono_ns"] = time.monotonic_ns()
                    emit(record)
            stderr = err.read().decode("utf-8", errors="replace")
            rc = out.channel.recv_exit_status()
            if rc:
                raise RuntimeError("read-only sampler failed: %s rc=%s %s" % (target["host"], rc, stderr[-600:]))
            return {"exit": rc, "stderr": stderr}
        finally:
            if other:
                other.close()

    def clock_probe(self, target, repeats=5):
        """Measure clock RPC RTT after SSH setup; EOF exits our own read-only helper."""
        source = "import sys,time\nfor line in sys.stdin:\n print(int(time.time()*1000000000),flush=True)\n"
        command = shlex.join(["python3","-u","-c",source])
        other=None
        try:
            client=self.bridge
            if target["kind"]=="oss":
                command=self.storage_command(target["host"],command)
            else:
                other=self.client(target["host"])
                client=other
            stdin,stdout,stderr=client.exec_command(command,timeout=15)
            samples=[]
            for _ in range(repeats):
                before=time.time_ns()
                stdin.write("clock\n");stdin.flush()
                remote=int(stdout.readline().strip())
                after=time.time_ns()
                samples.append({"offset_ns":remote-(before+after)//2,"rtt_ns":after-before})
            stdin.channel.shutdown_write()
            rc=stdout.channel.recv_exit_status()
            if rc:raise RuntimeError(stderr.read().decode(errors="replace"))
            best=min(samples,key=lambda s:s["rtt_ns"])
            return dict(host=target["host"],**best,samples=samples)
        finally:
            if other:other.close()
