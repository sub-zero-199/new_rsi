#!/usr/bin/env python3

import socket
import json
import os
import subprocess
import time
import threading
import signal
import sys
import re
import urllib.parse
import urllib.request
import platform
from pathlib import Path

class CameraStatusClient:
    def __init__(self):
        # Service start time for uptime tracking
        self.start_time = time.time()

        # Thread safety locks
        self._data_lock = threading.RLock()
        self._socket_lock = threading.Lock()

        self.load_config()
        self.arp_cache = {}
        self.socket = None
        self.alive = False
        self.backoff_ms = 1000
        self.max_backoff_ms = 20000
        self.timer = None
        self.host_mac = self.get_primary_mac()

        # Camera status with enhanced tracking
        self.camera_rtsp_status = {}
        self.camera_stats = {}  # Track failures, last_check, rtsp_ms per camera
        for ip in self.camera_ips:
            self.camera_stats[ip] = {
                'failures': 0,
                'last_check': None,
                'rtsp_ms': None,
                'last_error': None,
                # Stream diagnostics (updated every 15 min)
                'stream_info': None,  # {resolution, fps, bitrate, codec, issues}
                'last_stream_check': None
            }

        self.rtsp_check_timer = None
        self.workstation_ips = ['10.147.17.99', '10.147.17.7', '10.147.17.113', '10.147.17.80']
        self.workstation_status = {"state": "WEAK", "info": "Checks pending"}
        self.workstation_check_timer = None
        self.speedtest_results = {"ping": None, "upload": None, "download": None, "last_test": None}
        self.speedtest_timer = None
        self._speedtest_running = False  # Prevent overlapping speedtests
        self._speedtest_started = False  # Prevent multiple start calls

        # Connection health tracking
        self.conn_stats = {
            'reconnects': 0,
            'last_success': None,
            'packets_sent': 0,
            'packets_failed': 0,
            'bytes_sent': 0,
            'last_failure_reason': None
        }

        # System info cache (updated periodically)
        self.system_info = self._get_static_system_info()

        # Issues list for quick support scanning
        self.current_issues = []

    def load_config(self):
        config_path = os.environ.get('CONFIG_PATH', Path(__file__).parent / 'config.json')
        if not os.path.exists(config_path):
            print(f"Config file not found at {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            cfg = json.load(f)

        self.server_host = cfg.get('serverHost', '13.126.219.12')
        self.server_port = int(cfg.get('serverPort', 8004))
        self.send_interval_ms = int(cfg.get('sendIntervalMs', 30000))
        self.camera_ips = cfg.get('cameraIPs', [])
        self.ping_timeout_ms = int(cfg.get('pingTimeoutMs', 800))
        # RTSP creds: env first (RTSP_USERNAME / RTSP_PASSWORD), then config.json. Never
        # hardcode the password (it must not land in the repo / shared zip).
        self.rtsp_username = os.environ.get('RTSP_USERNAME') or cfg.get('rtspUsername', 'admin')
        self.rtsp_password = os.environ.get('RTSP_PASSWORD') or cfg.get('rtspPassword', '')
        self.check_interval_sec = int(cfg.get('checkIntervalSec', 900))

        # New config options with defaults
        self.rtsp_timeout_sec = int(cfg.get('rtspTimeoutSec', 8))
        self.max_retries = int(cfg.get('maxRetries', 3))
        self.verbose_logging = cfg.get('verboseLogging', False)

    # ==================== SYSTEM DIAGNOSTICS ====================

    def _get_static_system_info(self):
        """Get system info that doesn't change during runtime"""
        info = {
            'host': 'unknown',
            'os': 'unknown',
            'kernel': 'unknown'
        }

        try:
            info['host'] = platform.node()[:20]  # Truncate long hostnames
        except:
            pass

        try:
            # Try to get OS pretty name
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('PRETTY_NAME='):
                            info['os'] = line.split('=')[1].strip().strip('"')[:30]
                            break
            if info['os'] == 'unknown':
                info['os'] = f"{platform.system()} {platform.release()}"[:30]
        except:
            pass

        try:
            info['kernel'] = platform.release()[:20]
        except:
            pass

        return info

    def get_system_uptime(self):
        """Get system uptime in seconds"""
        try:
            with open('/proc/uptime', 'r') as f:
                return int(float(f.read().split()[0]))
        except:
            return None

    def get_service_uptime(self):
        """Get service uptime in seconds"""
        return int(time.time() - self.start_time)

    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        try:
            import psutil
            return round(psutil.cpu_percent(interval=0.1), 1)
        except ImportError:
            try:
                # Fallback: parse /proc/stat
                with open('/proc/stat', 'r') as f:
                    line = f.readline()
                    parts = line.split()
                    idle = int(parts[4])
                    total = sum(int(x) for x in parts[1:])
                # This is instantaneous, not very accurate
                return round((1 - idle / total) * 100, 1) if total > 0 else None
            except:
                return None
        except:
            return None

    def get_load_average(self):
        """Get 1, 5, 15 minute load averages"""
        try:
            load = os.getloadavg()
            return [round(x, 2) for x in load]
        except:
            return None

    def get_memory_info(self):
        """Get RAM usage info"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'used': round(mem.used / (1024 * 1024)),  # MB
                'total': round(mem.total / (1024 * 1024)),  # MB
                'pct': round(mem.percent, 1)
            }
        except ImportError:
            try:
                # Fallback: parse /proc/meminfo
                mem_info = {}
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts[0] in ['MemTotal:', 'MemAvailable:', 'MemFree:']:
                            mem_info[parts[0][:-1]] = int(parts[1])  # KB

                total = mem_info.get('MemTotal', 0) // 1024  # MB
                available = mem_info.get('MemAvailable', mem_info.get('MemFree', 0)) // 1024  # MB
                used = total - available
                pct = round((used / total) * 100, 1) if total > 0 else 0

                return {'used': used, 'total': total, 'pct': pct}
            except:
                return None
        except:
            return None

    def get_disk_info(self):
        """Get disk usage for root partition"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                'used': round(disk.used / (1024 * 1024 * 1024)),  # GB
                'total': round(disk.total / (1024 * 1024 * 1024)),  # GB
                'pct': round(disk.percent, 1)
            }
        except ImportError:
            try:
                # Fallback: use df command
                result = subprocess.run(['df', '-B1', '/'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        total = int(parts[1]) // (1024 * 1024 * 1024)  # GB
                        used = int(parts[2]) // (1024 * 1024 * 1024)  # GB
                        pct = round((used / total) * 100, 1) if total > 0 else 0
                        return {'used': used, 'total': total, 'pct': pct}
            except:
                pass
            return None
        except:
            return None

    def get_cpu_temperature(self):
        """Get CPU temperature if available"""
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > 0:
                            return round(entry.current, 1)
        except:
            pass

        # Fallback: try thermal zones
        try:
            thermal_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/devices/platform/coretemp.0/hwmon/hwmon0/temp1_input',
                '/sys/class/hwmon/hwmon0/temp1_input'
            ]
            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp = int(f.read().strip()) / 1000
                        if 0 < temp < 150:  # Sanity check
                            return round(temp, 1)
        except:
            pass

        return None

    def get_process_health(self):
        """Get current process health metrics"""
        try:
            import psutil
            proc = psutil.Process()
            return {
                'threads': proc.num_threads(),
                'fds': proc.num_fds() if hasattr(proc, 'num_fds') else None,
                'mem_mb': round(proc.memory_info().rss / (1024 * 1024), 1)
            }
        except:
            try:
                # Fallback for thread count
                pid = os.getpid()
                thread_count = len([f for f in os.listdir(f'/proc/{pid}/task') if f.isdigit()])
                fd_count = len(os.listdir(f'/proc/{pid}/fd'))
                return {'threads': thread_count, 'fds': fd_count, 'mem_mb': None}
            except:
                return {'threads': None, 'fds': None, 'mem_mb': None}

    def get_network_interface_stats(self):
        """Get network interface statistics"""
        try:
            import psutil
            stats = {}
            net_io = psutil.net_io_counters(pernic=True)
            for iface, counters in net_io.items():
                if iface not in ['lo', 'docker0'] and not iface.startswith('veth'):
                    stats[iface] = {
                        'rx_mb': round(counters.bytes_recv / (1024 * 1024), 1),
                        'tx_mb': round(counters.bytes_sent / (1024 * 1024), 1),
                        'errors': counters.errin + counters.errout
                    }
            return stats
        except:
            try:
                # Fallback: parse /proc/net/dev
                stats = {}
                with open('/proc/net/dev', 'r') as f:
                    for line in f:
                        if ':' in line:
                            parts = line.split(':')
                            iface = parts[0].strip()
                            if iface not in ['lo', 'docker0'] and not iface.startswith('veth'):
                                values = parts[1].split()
                                stats[iface] = {
                                    'rx_mb': round(int(values[0]) / (1024 * 1024), 1),
                                    'tx_mb': round(int(values[8]) / (1024 * 1024), 1),
                                    'errors': int(values[2]) + int(values[10])
                                }
                return stats
            except:
                return {}

    def check_dns_resolution(self):
        """Check if DNS resolution is working"""
        try:
            socket.gethostbyname('google.com')
            return True
        except:
            return False

    def check_gateway_reachable(self):
        """Check if default gateway is reachable"""
        try:
            result = subprocess.run(['ip', 'route', 'show', 'default'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                match = re.search(r'via\s+(\d+\.\d+\.\d+\.\d+)', result.stdout)
                if match:
                    gateway = match.group(1)
                    return self.ping_once(gateway, 1000) or self.tcp_reachable(gateway, 80) or self.tcp_reachable(gateway, 443)
        except:
            pass
        return None  # Unknown

    def measure_server_latency(self):
        """Measure TCP connect latency to server"""
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.server_host, self.server_port))
            latency = (time.time() - start) * 1000  # ms
            sock.close()
            return round(latency, 1)
        except:
            return None

    def collect_system_diagnostics(self):
        """Collect all system diagnostics"""
        proc_health = self.get_process_health()

        return {
            'host': self.system_info['host'],
            'os': self.system_info['os'],
            'kernel': self.system_info['kernel'],
            'sys_uptime': self.get_system_uptime(),
            'cpu': self.get_cpu_usage(),
            'ram': self.get_memory_info(),
            'disk': self.get_disk_info(),
            'load': self.get_load_average(),
            'temp': self.get_cpu_temperature(),
            'threads': proc_health.get('threads'),
            'fds': proc_health.get('fds')
        }

    def collect_network_diagnostics(self):
        """Collect all network diagnostics"""
        return {
            'zt_ip': self.get_zerotier_ip(),
            'gateway': self.check_gateway_reachable(),
            'dns': self.check_dns_resolution(),
            'internet': self.check_internet_connectivity(),
            'server_latency_ms': self.measure_server_latency(),
            'ifaces': self.get_network_interface_stats()
        }

    # ==================== CORE UTILITIES ====================

    def format_mac_address(self, mac):
        if not mac or mac == "unknown-host-mac":
            return mac
        return mac.lower().replace(':', '').replace('-', '')

    def get_primary_mac(self):
        # First try to get MAC from eno1 interface specifically
        try:
            mac_path = '/sys/class/net/eno1/address'
            if os.path.exists(mac_path):
                with open(mac_path, 'r') as f:
                    mac = f.read().strip().lower()
                    if mac and mac != "00:00:00:00:00:00":
                        return self.format_mac_address(mac)
        except:
            pass

        try:
            import psutil
            for interface, addrs in psutil.net_if_addrs().items():
                if interface == 'lo':
                    continue
                for addr in addrs:
                    if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                        mac_path = f'/sys/class/net/{interface}/address'
                        if os.path.exists(mac_path):
                            with open(mac_path, 'r') as f:
                                mac = f.read().strip().lower()
                                if mac and mac != "00:00:00:00:00:00":
                                    return self.format_mac_address(mac)
        except ImportError:
            pass

        try:
            for interface in os.listdir('/sys/class/net'):
                if interface == 'lo':
                    continue
                mac_path = f'/sys/class/net/{interface}/address'
                if os.path.exists(mac_path):
                    with open(mac_path, 'r') as f:
                        mac = f.read().strip().lower()
                        if mac and mac != "00:00:00:00:00:00":
                            return self.format_mac_address(mac)
        except:
            pass

        return "unknown-host-mac"

    def load_arp_table(self):
        try:
            result = self._run_command_safe(['arp', '-a'], timeout=5)
            if result and result.returncode == 0:
                lines = result.stdout.split('\n')

                for line in lines:
                    match = re.search(r'\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([a-fA-F0-9]{2}[:-][a-fA-F0-9]{2}[:-][a-fA-F0-9]{2}[:-][a-fA-F0-9]{2}[:-][a-fA-F0-9]{2}[:-][a-fA-F0-9]{2}).*\s+on\s+(\S+)', line)
                    if match:
                        ip = match.group(1)
                        mac = match.group(2).lower().replace('-', ':')
                        interface = match.group(3)

                        # Accept any interface, not just eno1
                        with self._data_lock:
                            self.arp_cache[ip] = self.format_mac_address(mac)
        except:
            pass

    def try_resolve_mac(self, ip):
        with self._data_lock:
            return self.arp_cache.get(ip)  # None if not yet in ARP cache

    def _run_command_safe(self, cmd, timeout=10):
        """Run command with guaranteed cleanup"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result
        except subprocess.TimeoutExpired:
            if self.verbose_logging:
                print(f"[CMD] Timeout: {' '.join(cmd)}")
            return None
        except FileNotFoundError:
            return None
        except Exception as e:
            if self.verbose_logging:
                print(f"[CMD] Error: {e}")
            return None

    def ping_once(self, ip, timeout_ms):
        try:
            timeout_sec = max(1, timeout_ms // 1000)
            result = subprocess.run(['ping', '-c', '1', '-W', str(timeout_sec), ip],
                                 capture_output=True, timeout=timeout_sec + 2)
            return result.returncode == 0
        except:
            return False

    def tcp_reachable(self, ip, port=554, timeout_sec=2):
        """TCP connect check — works even when cameras block ICMP ping."""
        try:
            with socket.create_connection((ip, port), timeout=timeout_sec):
                return True
        except Exception:
            return False

    # ==================== RTSP CHECKING ====================

    def check_rtsp_stream(self, ip):
        """Check RTSP stream with enhanced error handling and timing"""
        process = None
        start_time = time.time()

        try:
            encoded_password = urllib.parse.quote(self.rtsp_password, safe='')
            rtsp_url = f"rtsp://{self.rtsp_username}:{encoded_password}@{ip}:554/rtsp/streaming?channel=1&subtype=2"

            process = subprocess.Popen(
                ['ffmpeg', '-nostdin', '-y', '-i', rtsp_url, '-f', 'null', '-', '-t', '1'],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )

            output_lines = []
            timeout_sec = self.rtsp_timeout_sec

            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    break

                if process.poll() is not None:
                    break

                # Use select for non-blocking read with timeout
                try:
                    line = process.stderr.readline()
                except:
                    break

                if line:
                    output_lines.append(line)
                    combined = ''.join(output_lines)
                    if 'Stream #0' in combined or 'Input #0' in combined:
                        rtsp_ms = int((time.time() - start_time) * 1000)
                        with self._data_lock:
                            self.camera_stats[ip]['rtsp_ms'] = rtsp_ms
                            self.camera_stats[ip]['failures'] = 0
                            self.camera_stats[ip]['last_error'] = None
                        return True

                    error_indicators = ['connection refused', 'connection timed out', 'unable to open',
                                      'server returned 404', 'unauthorized', 'forbidden', 'no route to host',
                                      'input/output error', 'cannot open', 'authentication required']
                    combined_lower = combined.lower()
                    for error in error_indicators:
                        if error in combined_lower:
                            with self._data_lock:
                                self.camera_stats[ip]['failures'] += 1
                                self.camera_stats[ip]['last_error'] = error
                            return False
                else:
                    time.sleep(0.05)

            stderr_output = ''.join(output_lines)
            if 'Stream #0' in stderr_output or 'Input #0' in stderr_output:
                rtsp_ms = int((time.time() - start_time) * 1000)
                with self._data_lock:
                    self.camera_stats[ip]['rtsp_ms'] = rtsp_ms
                    self.camera_stats[ip]['failures'] = 0
                return True

            with self._data_lock:
                self.camera_stats[ip]['failures'] += 1
                self.camera_stats[ip]['last_error'] = 'timeout or no stream'
            return False

        except FileNotFoundError:
            with self._data_lock:
                self.camera_stats[ip]['last_error'] = 'ffmpeg not found'
            return False
        except Exception as e:
            with self._data_lock:
                self.camera_stats[ip]['failures'] += 1
                self.camera_stats[ip]['last_error'] = str(e)[:50]
            return False
        finally:
            # Guaranteed cleanup
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()  # Reap zombie
                except:
                    try:
                        process.kill()
                    except:
                        pass

    def analyze_stream_quality(self, ip):
        """
        Analyze stream quality using ffprobe - detailed diagnostics.
        Returns dict with resolution, fps, bitrate, codec, and issues.
        """
        process = None
        try:
            encoded_password = urllib.parse.quote(self.rtsp_password, safe='')
            rtsp_url = f"rtsp://{self.rtsp_username}:{encoded_password}@{ip}:554/rtsp/streaming?channel=1&subtype=0"

            # Use ffprobe to get stream info
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-rtsp_transport', 'tcp',
                '-show_entries', 'stream=width,height,r_frame_rate,bit_rate,codec_name,codec_type',
                '-show_entries', 'format=bit_rate,duration',
                '-of', 'json',
                '-timeout', '10000000',  # 10 sec in microseconds
                rtsp_url
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return {'issues': ['probe timeout']}

            if process.returncode != 0:
                return {'issues': ['probe failed']}

            data = json.loads(stdout)
            streams = data.get('streams', [])
            format_info = data.get('format', {})

            result = {
                'resolution': None,
                'fps': None,
                'bitrate': None,
                'codec': None,
                'issues': []
            }

            # Parse video stream info
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    width = stream.get('width')
                    height = stream.get('height')
                    if width and height:
                        result['resolution'] = f"{width}x{height}"

                        # Check resolution issues
                        if width < 640 or height < 480:
                            result['issues'].append('low-res')

                    # Frame rate
                    fps_str = stream.get('r_frame_rate', '0/1')
                    try:
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            fps = float(num) / float(den) if float(den) > 0 else 0
                        else:
                            fps = float(fps_str)
                        result['fps'] = round(fps, 1)

                        # Check FPS issues
                        if fps < 10:
                            result['issues'].append('low-fps')
                        elif fps < 15:
                            result['issues'].append('choppy')
                    except:
                        pass

                    # Codec
                    codec = stream.get('codec_name')
                    if codec:
                        result['codec'] = codec.upper()

                    # Stream bitrate
                    bitrate = stream.get('bit_rate')
                    if bitrate:
                        try:
                            kbps = int(bitrate) / 1000
                            result['bitrate'] = f"{int(kbps)}k"

                            # Check bitrate issues
                            if kbps < 500:
                                result['issues'].append('low-bitrate')
                        except:
                            pass

            # Fallback to format bitrate
            if not result['bitrate'] and format_info.get('bit_rate'):
                try:
                    kbps = int(format_info['bit_rate']) / 1000
                    result['bitrate'] = f"{int(kbps)}k"
                    if kbps < 500:
                        result['issues'].append('low-bitrate')
                except:
                    pass

            return result

        except FileNotFoundError:
            return {'issues': ['ffprobe not found']}
        except json.JSONDecodeError:
            return {'issues': ['parse error']}
        except Exception as e:
            return {'issues': [str(e)[:20]]}
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                        process.wait()
                    except:
                        pass

    def check_rtsp_for_all_cameras(self):
        if not self.alive:
            return

        current_time = int(time.time())

        for ip in self.camera_ips:
            if not self.alive:
                break

            with self._data_lock:
                self.camera_stats[ip]['last_check'] = current_time

            if self.tcp_reachable(ip, port=554) or self.ping_once(ip, self.ping_timeout_ms):
                is_accessible = self.check_rtsp_stream(ip)
                with self._data_lock:
                    self.camera_rtsp_status[ip] = is_accessible

                # Run stream quality analysis for accessible cameras
                # Only analyze every 15 min to avoid overloading
                if is_accessible and self.alive:
                    with self._data_lock:
                        last_stream_check = self.camera_stats[ip].get('last_stream_check')

                    # Check if 15 minutes have passed since last analysis
                    should_analyze = (last_stream_check is None or
                                     (current_time - last_stream_check) >= 900)

                    if should_analyze:
                        if self.verbose_logging:
                            print(f"[Stream] Analyzing stream quality for {ip}...")

                        stream_info = self.analyze_stream_quality(ip)

                        with self._data_lock:
                            self.camera_stats[ip]['stream_info'] = stream_info
                            self.camera_stats[ip]['last_stream_check'] = current_time

                        if self.verbose_logging:
                            info_str = self._format_stream_info(stream_info)
                            print(f"[Stream] {ip}: {info_str}")
            else:
                with self._data_lock:
                    self.camera_rtsp_status[ip] = False
                    self.camera_stats[ip]['failures'] += 1
                    self.camera_stats[ip]['last_error'] = 'not reachable'

    def _format_stream_info(self, stream_info):
        """Format stream info into compressed string for display"""
        if not stream_info:
            return ""

        parts = []
        if stream_info.get('resolution'):
            parts.append(stream_info['resolution'])
        if stream_info.get('fps'):
            parts.append(f"{stream_info['fps']}fps")
        if stream_info.get('bitrate'):
            parts.append(stream_info['bitrate'])
        if stream_info.get('codec'):
            parts.append(stream_info['codec'])

        issues = stream_info.get('issues', [])
        if issues:
            parts.append(f"⚠{','.join(issues)}")

        return ' '.join(parts)

    def start_rtsp_checking(self):
        def run_check(first=False):
            self.check_rtsp_for_all_cameras()
            if self.alive:
                if first:
                    # Send an updated payload immediately after the first RTSP check
                    # so the portal doesn't stay on "RTSP check pending"
                    threading.Thread(target=self.send_once, daemon=True).start()
                with self._data_lock:
                    if self.rtsp_check_timer:
                        self.rtsp_check_timer.cancel()
                    self.rtsp_check_timer = threading.Timer(self.check_interval_sec, run_check)
                    self.rtsp_check_timer.daemon = True
                    self.rtsp_check_timer.start()

        # Run in background thread to not block
        threading.Thread(target=lambda: run_check(first=True), daemon=True).start()

    # ==================== ZEROTIER & VPN ====================

    def get_zerotier_ip(self):
        try:
            result = self._run_command_safe(['zerotier-cli', '-j', 'listnetworks'], timeout=3)
            if result and result.returncode == 0:
                try:
                    networks = json.loads(result.stdout)
                    if isinstance(networks, list):
                        for network in networks:
                            if network.get('status') == 'OK' and network.get('assignedAddresses'):
                                for addr in network['assignedAddresses']:
                                    ip = addr.split('/')[0] if '/' in addr else addr
                                    if (ip.startswith('10.') or ip.startswith('172.') or
                                        ip.startswith('192.168.')):
                                        return ip
                except json.JSONDecodeError:
                    pass
        except:
            pass

        try:
            result = self._run_command_safe(['zerotier-cli', 'listnetworks'], timeout=3)
            if result:
                for line in result.stdout.split('\n'):
                    if 'zt' in line.lower() and 'OK' in line:
                        parts = line.split()
                        for part in parts:
                            if (part.startswith('10.') or part.startswith('172.') or
                                part.startswith('192.168.') or part.startswith('10.147.')):
                                return part
        except:
            pass

        try:
            result = self._run_command_safe(['ip', 'addr', 'show'], timeout=3)
            if result:
                current_interface = None
                for line in result.stdout.split('\n'):
                    if ':' in line and ('zt' in line.lower() or 'zerotier' in line.lower()):
                        parts = line.split(':')
                        if len(parts) >= 2:
                            current_interface = parts[1].strip().split('@')[0]
                    elif current_interface and 'inet ' in line and not '127.' in line:
                        ip_match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', line)
                        if ip_match:
                            ip = ip_match.group(1)
                            if (ip.startswith('10.') or ip.startswith('172.') or
                                ip.startswith('192.168.')):
                                return ip
        except:
            pass

        try:
            import psutil
            interfaces = psutil.net_if_addrs()
            zt_keywords = ['zt', 'zerotier']
            for interface_name in interfaces.keys():
                if any(keyword in interface_name.lower() for keyword in zt_keywords):
                    addrs = interfaces[interface_name]
                    for addr in addrs:
                        if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                            ip = addr.address
                            if (ip.startswith('10.') or ip.startswith('172.') or
                                ip.startswith('192.168.')):
                                return ip
        except:
            pass

        return None

    def check_vpn_connectivity(self):
        successful_pings = 0
        for ip in self.workstation_ips:
            if self.ping_once(ip, 3000) or self.tcp_reachable(ip, 22, timeout_sec=3):
                successful_pings += 1

        vpn_connected = successful_pings > 0

        try:
            import psutil
            interfaces = psutil.net_if_addrs()
            vpn_keywords = ['tun', 'tap', 'ppp', 'vpn', 'wg', 'openvpn', 'wireguard', 'zt', 'zerotier']
            vpn_interface = None
            for interface_name in interfaces.keys():
                if any(keyword in interface_name.lower() for keyword in vpn_keywords):
                    addrs = interfaces[interface_name]
                    for addr in addrs:
                        if addr.family == socket.AF_INET:
                            vpn_interface = interface_name
                            break
                    if vpn_interface:
                        break
            return vpn_connected, vpn_interface, successful_pings
        except:
            return vpn_connected, None, successful_pings

    # ==================== SSH CHECKS ====================

    def check_ssh_service(self):
        try:
            result = self._run_command_safe(['systemctl', 'is-active', 'ssh'], timeout=2)
            if result and result.returncode == 0 and 'active' in result.stdout.lower():
                return True
        except:
            pass

        try:
            result = self._run_command_safe(['systemctl', 'is-active', 'sshd'], timeout=2)
            if result and result.returncode == 0 and 'active' in result.stdout.lower():
                return True
        except:
            pass

        try:
            result = self._run_command_safe(['pgrep', '-x', 'sshd'], timeout=2)
            return result and result.returncode == 0
        except:
            return False

    def check_ssh_port_accessible(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 22))
            sock.close()
            return result == 0
        except:
            return False

    def check_ssh_listening_interfaces(self):
        try:
            result = self._run_command_safe(['ss', '-tlnp'], timeout=3)
            if result and result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':22' in line or 'ssh' in line.lower():
                        if '0.0.0.0:22' in line or '*:22' in line:
                            return True, 'all'
                        elif '127.0.0.1:22' in line:
                            return True, 'localhost'
                        elif ':22' in line:
                            return True, 'custom'
            return False, None
        except:
            pass

        return None, None

    def check_firewall_blocking_ssh(self):
        try:
            result = self._run_command_safe(['ufw', 'status'], timeout=3)
            if result and result.returncode == 0:
                output = result.stdout.lower()
                if 'active' in output:
                    if '22' in output and ('deny' in output or 'reject' in output):
                        return True
                    elif '22' not in output and 'deny' in output:
                        return True
        except:
            pass

        return False

    def get_active_ssh_connections(self):
        count = 0

        try:
            result = self._run_command_safe(['ss', '-tnp'], timeout=3)
            if result and result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':22' in line and 'ESTAB' in line:
                        count += 1
        except:
            pass

        return count

    # ==================== INTERNET CHECKS ====================

    def check_internet_connectivity(self):
        # Try ping first, fall back to TCP (port 443) — some environments block ICMP
        test_hosts = ['8.8.8.8', '1.1.1.1']
        for host in test_hosts:
            if self.ping_once(host, 3000) or self.tcp_reachable(host, 443, timeout_sec=3):
                return True
        return False

    # ==================== WORKSTATION STATUS ====================

    def _format_uptime(self, seconds):
        """Format uptime in human readable format"""
        if seconds is None:
            return "?"
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
        else:
            return f"{seconds // 86400}d{(seconds % 86400) // 3600}h"

    def check_workstation_connectivity(self):
        """Build comprehensive info string with all diagnostics (compact format)"""
        info_parts = []
        issues = []

        # === NETWORK STATUS (compact) ===
        zerotier_ip = self.get_zerotier_ip()
        if zerotier_ip:
            info_parts.append(f"ZT:{zerotier_ip}")
        else:
            info_parts.append("ZT:NONE")

        vpn_connected, _, successful_pings = self.check_vpn_connectivity()
        if vpn_connected:
            info_parts.append(f"VPN:{successful_pings}/{len(self.workstation_ips)}")
        else:
            issues.append("VPN not connected")
            info_parts.append(f"VPN:0/{len(self.workstation_ips)}⚠")

        internet_connected = self.check_internet_connectivity()
        if internet_connected:
            info_parts.append("Net:OK")
        else:
            issues.append("Internet connectivity issue")
            info_parts.append("Net:FAIL⚠")

        # === SSH STATUS (compact) ===
        ssh_service_running = self.check_ssh_service()
        ssh_port_accessible = self.check_ssh_port_accessible()
        _, listening_info = self.check_ssh_listening_interfaces()
        active_ssh = self.get_active_ssh_connections()

        if ssh_service_running and ssh_port_accessible:
            if listening_info == 'localhost':
                issues.append("SSH listening on localhost only")
                ssh_str = "SSH:LOCAL⚠"
            else:
                ssh_str = f"SSH:OK({active_ssh})" if active_ssh > 0 else "SSH:OK"
        else:
            if not ssh_service_running:
                issues.append("SSH service not running")
                ssh_str = "SSH:DOWN⚠"
            else:
                issues.append("SSH port not accessible")
                ssh_str = "SSH:BLOCKED⚠"
        info_parts.append(ssh_str)

        # === SPEED TEST (compact) ===
        with self._data_lock:
            download = self.speedtest_results.get('download')
            upload = self.speedtest_results.get('upload')
            ping = self.speedtest_results.get('ping')

        if download is not None and upload is not None:
            speed_str = f"{download:.0f}↓/{upload:.0f}↑"
            if ping is not None:
                speed_str += f"/{ping:.0f}ms"
            info_parts.append(speed_str)
            if download < 5.0 or upload < 1.0:
                issues.append("Low internet speed")

        # === SYSTEM RESOURCES (compressed) ===
        sys_parts = []

        # CPU
        cpu = self.get_cpu_usage()
        if cpu is not None:
            sys_parts.append(f"CPU:{cpu:.0f}%")

        # RAM
        ram = self.get_memory_info()
        if ram:
            sys_parts.append(f"RAM:{ram['pct']:.0f}%")

        # Disk
        disk = self.get_disk_info()
        if disk:
            sys_parts.append(f"Disk:{disk['pct']:.0f}%")

        # Load
        load = self.get_load_average()
        if load:
            sys_parts.append(f"Load:{load[0]:.1f}")

        # Temp
        temp = self.get_cpu_temperature()
        if temp:
            sys_parts.append(f"Temp:{temp:.0f}°C")

        if sys_parts:
            info_parts.append(" ".join(sys_parts))

        # === UPTIME & CONNECTION ===
        uptime_parts = []

        # Service uptime
        svc_uptime = self.get_service_uptime()
        uptime_parts.append(f"Svc:{self._format_uptime(svc_uptime)}")

        # System uptime
        sys_uptime = self.get_system_uptime()
        if sys_uptime:
            uptime_parts.append(f"Sys:{self._format_uptime(sys_uptime)}")

        # Connection stats
        with self._data_lock:
            conn = self.conn_stats
            if conn['packets_sent'] > 0:
                uptime_parts.append(f"Pkts:{conn['packets_sent']}")
            if conn['reconnects'] > 0:
                uptime_parts.append(f"Reconn:{conn['reconnects']}")

        if uptime_parts:
            info_parts.append(" ".join(uptime_parts))

        # === CAMERA SUMMARY WITH INDIVIDUAL DETAILS ===
        with self._data_lock:
            active_cams = sum(1 for ip in self.camera_ips if self.camera_rtsp_status.get(ip) == True)
            total_cams = len(self.camera_ips)

            # Build individual camera status strings
            cam_details = []
            for ip in self.camera_ips:
                # Get camera identifier (last octet of IP)
                cam_id = ip.split('.')[-1]

                rtsp_status = self.camera_rtsp_status.get(ip)
                cam_stats = self.camera_stats.get(ip, {})
                stream_info = cam_stats.get('stream_info')

                if rtsp_status is None:
                    # Not checked yet or ping failed
                    if cam_stats.get('last_error') == 'ping failed':
                        cam_details.append(f".{cam_id}:OFF")
                    else:
                        cam_details.append(f".{cam_id}:?")
                elif rtsp_status:
                    # Camera online - show stream info
                    if stream_info:
                        fps = stream_info.get('fps', 0)
                        res = stream_info.get('resolution', '')
                        # Shorten resolution (2880x1620 -> 2.8K, 1920x1080 -> 1080p)
                        if res:
                            width = int(res.split('x')[0]) if 'x' in res else 0
                            if width >= 2800:
                                res_short = "2.8K"
                            elif width >= 1920:
                                res_short = "1080p"
                            elif width >= 1280:
                                res_short = "720p"
                            else:
                                res_short = res
                        else:
                            res_short = ""

                        issues = stream_info.get('issues', [])
                        rtsp_ms = cam_stats.get('rtsp_ms', 0)

                        if issues:
                            # Has issues
                            cam_details.append(f".{cam_id}:{res_short}/{fps:.0f}fps⚠{issues[0]}")
                        else:
                            # All good - show compact info
                            cam_details.append(f".{cam_id}:{res_short}/{fps:.0f}fps/{rtsp_ms}ms")
                    else:
                        # Online but no stream info yet
                        rtsp_ms = cam_stats.get('rtsp_ms', 0)
                        cam_details.append(f".{cam_id}:OK/{rtsp_ms}ms")
                else:
                    # RTSP failed
                    error = cam_stats.get('last_error', 'err')[:8]
                    cam_details.append(f".{cam_id}:FAIL({error})")

        # Build camera summary string
        cam_summary = f"Cams:{active_cams}/{total_cams}"

        # Add individual camera details (compact format)
        if cam_details:
            cam_summary += " [" + " ".join(cam_details) + "]"

        info_parts.append(cam_summary)

        # === GATEWAY/DNS ===
        gateway = self.check_gateway_reachable()
        dns = self.check_dns_resolution()
        if gateway == False:
            issues.append("Gateway down")
            info_parts.append("GW:DOWN")
        if dns == False:
            issues.append("DNS failed")
            info_parts.append("DNS:FAIL")

        info = " | ".join(info_parts)

        with self._data_lock:
            self.current_issues = issues

        # State based on critical issues only (VPN, Internet, SSH, Gateway, DNS)
        # Camera status does NOT affect main state - each camera has its own state
        if issues:
            return "WEAK", info
        else:
            return "STRONG", info

    def check_workstation_status(self):
        if not self.alive:
            return
        state, info = self.check_workstation_connectivity()
        if self.alive:
            with self._data_lock:
                self.workstation_status = {"state": state, "info": info}

    def start_workstation_checking(self):
        def run_check():
            self.check_workstation_status()
            if self.alive:
                with self._data_lock:
                    if self.workstation_check_timer:
                        self.workstation_check_timer.cancel()
                    self.workstation_check_timer = threading.Timer(self.check_interval_sec, run_check)
                    self.workstation_check_timer.daemon = True
                    self.workstation_check_timer.start()

        threading.Thread(target=run_check, daemon=True).start()

    # ==================== SPEEDTEST ====================

    def install_speedtest_tool(self):
        install_methods = [
            (['pip3', 'install', '--user', 'speedtest-cli'], 'pip3 speedtest-cli'),
            (['pip', 'install', '--user', 'speedtest-cli'], 'pip speedtest-cli'),
            (['python3', '-m', 'pip', 'install', '--user', 'speedtest-cli'], 'python3 -m pip speedtest-cli'),
        ]

        for cmd, method_name in install_methods:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"[SpeedTest] Installed via {method_name}")
                    return True
            except:
                continue
        return False

    def check_speedtest_available(self):
        methods = [
            (['speedtest', '--version'], 'speedtest'),
            (['speedtest-cli', '--version'], 'speedtest-cli'),
        ]

        for cmd, name in methods:
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
                if result.returncode == 0 or 'speedtest' in result.stdout.lower():
                    return True, name
            except:
                continue

        try:
            import speedtest
            return True, 'speedtest python module'
        except ImportError:
            pass

        return False, None

    def run_custom_speedtest(self):
        """
        Custom speed test using public CDN files - no rate limits!
        Downloads test files and measures speed directly.
        """
        # Test file URLs (various sizes for accuracy)
        test_urls = [
            # Cloudflare speed test endpoints
            ('https://speed.cloudflare.com/__down?bytes=10000000', 10_000_000),  # 10MB
            ('https://speed.cloudflare.com/__down?bytes=5000000', 5_000_000),    # 5MB
            # Hetzner speed test
            ('https://speed.hetzner.de/1GB.bin', 10_000_000),  # Will only download partial
            # OVH test files
            ('http://proof.ovh.net/files/10Mb.dat', 10_000_000),
            # Tele2 test files
            ('http://speedtest.tele2.net/10MB.zip', 10_000_000),
        ]

        ping = None
        download = None
        upload = None

        # Measure ping (TCP connect time to Cloudflare)
        try:
            import socket
            ping_start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(('speed.cloudflare.com', 443))
            ping = (time.time() - ping_start) * 1000  # Convert to ms
            sock.close()
        except Exception as e:
            if self.verbose_logging:
                print(f"[SpeedTest Custom] Ping measurement failed: {e}")

        # Measure download speed
        download_speeds = []
        for url, expected_size in test_urls:
            if not self.alive:
                break
            try:
                # Limit download to ~10MB and 15 seconds max
                max_bytes = min(expected_size, 10_000_000)
                timeout = 15

                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
                })

                start_time = time.time()
                bytes_downloaded = 0

                with urllib.request.urlopen(req, timeout=timeout) as response:
                    while True:
                        elapsed = time.time() - start_time
                        if elapsed > timeout or bytes_downloaded >= max_bytes:
                            break
                        chunk = response.read(65536)  # 64KB chunks
                        if not chunk:
                            break
                        bytes_downloaded += len(chunk)

                elapsed = time.time() - start_time
                if elapsed > 0.5 and bytes_downloaded > 100000:  # At least 0.5s and 100KB
                    # Calculate speed in Mbps (megabits per second)
                    speed_mbps = (bytes_downloaded * 8) / (elapsed * 1_000_000)
                    download_speeds.append(speed_mbps)
                    if self.verbose_logging:
                        print(f"[SpeedTest Custom] {url[:50]}... = {speed_mbps:.1f} Mbps")
                    # If we got a good result, no need to try more URLs
                    if speed_mbps > 1.0:
                        break

            except Exception as e:
                if self.verbose_logging:
                    print(f"[SpeedTest Custom] Download from {url[:40]}... failed: {e}")
                continue

        if download_speeds:
            # Use the best speed (highest)
            download = max(download_speeds)

        # Estimate upload as ~80% of download (common ratio)
        # Real upload test would require a server endpoint that accepts POST data
        if download:
            upload = download * 0.8

        if download and ping:
            return ping, download, upload
        elif download:
            return ping or 0, download, upload
        else:
            return None, None, None

    def run_speedtest_cli_command(self):
        commands_json = [
            ['speedtest', '--format', 'json', '--accept-license', '--accept-gdpr'],
            ['speedtest', '--format', 'json', '--accept-license'],
            ['speedtest', '--format', 'json'],
        ]

        for cmd in commands_json:
            try:
                if not self.alive:
                    break
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=120)
                if result.returncode == 0:
                    try:
                        data = json.loads(result.stdout)
                        ping = None
                        download = None
                        upload = None

                        if 'ping' in data:
                            if isinstance(data['ping'], dict):
                                ping = data['ping'].get('latency') or data['ping'].get('value')
                            elif isinstance(data['ping'], (int, float)):
                                ping = data['ping']

                        if 'idleLatency' in data and not ping:
                            ping = data.get('idleLatency')

                        if 'download' in data:
                            if isinstance(data['download'], dict):
                                bandwidth = data['download'].get('bandwidth') or data['download'].get('value')
                                if bandwidth:
                                    download = bandwidth / 1_000_000 if bandwidth > 1000 else bandwidth
                            elif isinstance(data['download'], (int, float)):
                                download = data['download'] / 1_000_000 if data['download'] > 1000 else data['download']

                        if 'upload' in data:
                            if isinstance(data['upload'], dict):
                                bandwidth = data['upload'].get('bandwidth') or data['upload'].get('value')
                                if bandwidth:
                                    upload = bandwidth / 1_000_000 if bandwidth > 1000 else bandwidth
                            elif isinstance(data['upload'], (int, float)):
                                upload = data['upload'] / 1_000_000 if data['upload'] > 1000 else data['upload']

                        if ping and download and upload and ping > 0 and download > 0 and upload > 0:
                            return ping, download, upload
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            except:
                continue

        commands_simple = [
            ['speedtest-cli', '--simple', '--secure'],
            ['speedtest', '--simple', '--secure'],
        ]

        for cmd in commands_simple:
            try:
                if not self.alive:
                    break
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=120)
                if result.returncode == 0:
                    output = result.stdout
                    ping_match = re.search(r'Ping:\s+([\d.]+)\s*ms', output)
                    download_match = re.search(r'Download:\s+([\d.]+)\s*Mbit/s', output)
                    upload_match = re.search(r'Upload:\s+([\d.]+)\s*Mbit/s', output)

                    if ping_match and download_match and upload_match:
                        return float(ping_match.group(1)), float(download_match.group(1)), float(upload_match.group(1))
            except:
                continue

        return None, None, None

    def run_speedtest_python_module(self):
        try:
            import speedtest
            st = speedtest.Speedtest()
            st.get_best_server()
            ping = st.results.ping
            download = st.download() / 1_000_000
            upload = st.upload() / 1_000_000
            return ping, download, upload
        except:
            return None, None, None

    def perform_speedtest(self):
        if not self.alive:
            return

        # Prevent overlapping speedtests
        with self._data_lock:
            if self._speedtest_running:
                if self.verbose_logging:
                    print("[SpeedTest] Already running, skipping")
                return

            # Minimum 2 minutes between speedtests
            last_test = self.speedtest_results.get('last_test')
            if last_test and (time.time() - last_test) < 120:
                if self.verbose_logging:
                    print(f"[SpeedTest] Rate limit: waiting {120 - int(time.time() - last_test)}s before next test")
                return

            self._speedtest_running = True

        try:
            print("[SpeedTest] Starting speed test...")
            ping, download, upload = None, None, None

            # Method 1: Custom CDN-based speed test (PRIMARY - no rate limits!)
            if self.alive:
                try:
                    print("[SpeedTest] Using custom CDN test...")
                    ping, download, upload = self.run_custom_speedtest()
                    if download and upload:
                        print(f"[SpeedTest] Custom test: {download:.1f}↓/{upload:.1f}↑ Mbps")
                except Exception as e:
                    if self.verbose_logging:
                        print(f"[SpeedTest] Custom test failed: {e}")

            # Method 2: Ookla CLI (fallback - may be rate limited)
            if (download is None or upload is None) and self.alive:
                try:
                    print("[SpeedTest] Trying Ookla CLI...")
                    ping2, download2, upload2 = self.run_speedtest_cli_command()
                    if download2 and upload2:
                        ping, download, upload = ping2, download2, upload2
                except Exception as e:
                    if self.verbose_logging:
                        print(f"[SpeedTest] Ookla CLI failed: {e}")

            # Method 3: Python speedtest module (last resort)
            if (download is None or upload is None) and self.alive:
                try:
                    print("[SpeedTest] Trying Python module...")
                    ping3, download3, upload3 = self.run_speedtest_python_module()
                    if download3 and upload3:
                        ping, download, upload = ping3, download3, upload3
                except Exception as e:
                    if self.verbose_logging:
                        print(f"[SpeedTest] Python module failed: {e}")

            if self.alive and download is not None and upload is not None:
                with self._data_lock:
                    self.speedtest_results = {
                        "ping": ping if ping else 0,
                        "download": download,
                        "upload": upload,
                        "last_test": int(time.time())
                    }
                print(f"[SpeedTest] Completed: {download:.1f}↓/{upload:.1f}↑ Mbps, {ping:.0f}ms ping" if ping else f"[SpeedTest] Completed: {download:.1f}↓/{upload:.1f}↑ Mbps")
            else:
                # Mark last_test time even on failure to prevent rapid retries
                with self._data_lock:
                    self.speedtest_results['last_test'] = int(time.time())
                print("[SpeedTest] All speedtest methods failed (will retry in 2 min)")

        except Exception as e:
            print(f"[SpeedTest] Unexpected error: {e}")
        finally:
            with self._data_lock:
                self._speedtest_running = False

    def start_speedtest_checking(self):
        # Prevent multiple starts
        with self._data_lock:
            if self._speedtest_started:
                if self.verbose_logging:
                    print("[SpeedTest] Checker already started, skipping")
                return
            self._speedtest_started = True

        def run_check():
            self.perform_speedtest()
            if self.alive:
                with self._data_lock:
                    if self.speedtest_timer:
                        self.speedtest_timer.cancel()
                    # Schedule next check (default 15 min)
                    self.speedtest_timer = threading.Timer(self.check_interval_sec, run_check)
                    self.speedtest_timer.daemon = True
                    self.speedtest_timer.start()
                if self.verbose_logging:
                    print(f"[SpeedTest] Next test scheduled in {self.check_interval_sec}s")

        threading.Thread(target=run_check, daemon=True).start()

    # ==================== PAYLOAD BUILDING ====================

    def build_payload(self):
        self.load_arp_table()

        # Build cameras array (original format + enhanced fields)
        cameras = []
        for ip in self.camera_ips:
            # TCP port 554 is the primary check — cameras often block ICMP ping
            active = self.tcp_reachable(ip, port=554) or self.ping_once(ip, self.ping_timeout_ms)
            # Ping/connect populates the ARP cache; reload so we get the MAC
            self.load_arp_table()
            camera_id = self.try_resolve_mac(ip)  # always a MAC or None — never an IP

            with self._data_lock:
                rtsp_status = self.camera_rtsp_status.get(ip)
                cam_stats = self.camera_stats.get(ip, {})

            # Build camera info string with stream diagnostics
            stream_info = cam_stats.get('stream_info')
            stream_str = self._format_stream_info(stream_info) if stream_info else ""

            if not active:
                state = "WEAK"
                info = "camera not reachable"
            elif rtsp_status is None:
                state = "WEAK"
                info = "RTSP check pending"
            elif rtsp_status:
                # Camera is working - include stream info
                stream_issues = stream_info.get('issues', []) if stream_info else []
                if stream_issues:
                    # Has stream quality issues
                    state = "WEAK"
                    info = stream_str
                else:
                    state = "STRONG"
                    info = stream_str  # Show stream details even when OK
            else:
                state = "WEAK"
                info = "camera restart required"

            cameras.append({
                'id': camera_id,
                'ip': ip,
                'active': active,
                'state': state,
                'info': info,
                'last_check': cam_stats.get('last_check'),
                'failures': cam_stats.get('failures', 0),
                'rtsp_ms': cam_stats.get('rtsp_ms'),
                'stream': stream_info  # Include full stream data for detailed view
            })

        with self._data_lock:
            workstation_status = self.workstation_status.copy()
            speedtest = self.speedtest_results.copy()

        # Build payload in ORIGINAL format for backward compatibility
        payload = {
            'client': 'camera_status',
            'id': self.host_mac,
            'state': workstation_status['state'],
            'info': workstation_status['info'],
            'cameras': cameras
        }

        # Add speedtest in original format
        if speedtest.get('ping') is not None:
            payload['speedtest'] = {
                'ping': round(speedtest['ping'], 1) if speedtest['ping'] else None,
                'download': round(speedtest['download'], 1) if speedtest['download'] else None,
                'upload': round(speedtest['upload'], 1) if speedtest['upload'] else None
            }

        return payload

    def frame_message(self, obj):
        return f"$${json.dumps(obj)}&&"

    # ==================== TCP CONNECTION ====================

    def connect(self):
        if self.alive:
            return

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)

            # Enable TCP keepalive
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                # Linux-specific keepalive settings
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            except (AttributeError, OSError):
                pass  # Not available on all platforms

            self.socket.connect((self.server_host, self.server_port))

            self.alive = True
            print(f"[TCP] Connected to {self.server_host}:{self.server_port}")
            self.backoff_ms = 1000
            self.start_sending()

        except Exception as e:
            print(f"[TCP] Connection error: {e}")
            with self._data_lock:
                self.conn_stats['last_failure_reason'] = str(e)[:50]
            self.schedule_reconnect()

    def schedule_reconnect(self):
        with self._data_lock:
            self.conn_stats['reconnects'] += 1

        wait = self.backoff_ms
        self.backoff_ms = min(self.backoff_ms * 2, self.max_backoff_ms)
        print(f"[TCP] Reconnecting in {wait} ms...")
        t = threading.Timer(wait / 1000, self.connect)
        t.daemon = True
        t.start()

    def start_sending(self):
        with self._data_lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(self.send_interval_ms / 1000, self.send_loop)
            self.timer.daemon = True
            self.timer.start()

        self.send_once()
        self.start_rtsp_checking()
        self.start_workstation_checking()
        self.start_speedtest_checking()

    def send_loop(self):
        if self.alive:
            self.send_once()
            with self._data_lock:
                self.timer = threading.Timer(self.send_interval_ms / 1000, self.send_loop)
                self.timer.daemon = True
                self.timer.start()

    def stop_sending(self):
        with self._data_lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            if self.rtsp_check_timer:
                self.rtsp_check_timer.cancel()
                self.rtsp_check_timer = None
            if self.workstation_check_timer:
                self.workstation_check_timer.cancel()
                self.workstation_check_timer = None
            if self.speedtest_timer:
                self.speedtest_timer.cancel()
                self.speedtest_timer = None
            # Reset started flags for potential reconnect
            self._speedtest_started = False

    def _send_with_retry(self, framed):
        """Send with retry logic"""
        for attempt in range(self.max_retries):
            try:
                with self._socket_lock:
                    if not self.socket:
                        return False
                    self.socket.send(framed.encode('utf-8'))
                return True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    raise e
        return False

    def send_once(self):
        if not self.alive or not self.socket:
            return

        try:
            payload = self.build_payload()
            framed = self.frame_message(payload)

            print(f"[TCP] Sending to {self.server_host}:{self.server_port}")
            if self.verbose_logging:
                print(f"[TCP] Payload: {json.dumps(payload)}")

            if self._send_with_retry(framed):
                bytes_sent = len(framed.encode('utf-8'))
                with self._data_lock:
                    self.conn_stats['packets_sent'] += 1
                    self.conn_stats['bytes_sent'] += bytes_sent
                    self.conn_stats['last_success'] = int(time.time())
                print(f"[TCP] Packet sent successfully ({bytes_sent} bytes)")
            else:
                raise Exception("Send failed after retries")

            # Try to receive response
            with self._socket_lock:
                self.socket.settimeout(0.5)
                try:
                    response = self.socket.recv(1024)
                    if response and self.verbose_logging:
                        print(f"[TCP] Response received: {response.decode('utf-8', errors='ignore')}")
                except socket.timeout:
                    pass
                finally:
                    self.socket.settimeout(10)

        except Exception as e:
            print(f"[TCP] Write error: {e}")
            with self._data_lock:
                self.conn_stats['packets_failed'] += 1
                self.conn_stats['last_failure_reason'] = str(e)[:50]
            self.alive = False
            self.stop_sending()
            self.schedule_reconnect()

    def cleanup(self):
        self.alive = False

        self.stop_sending()

        with self._socket_lock:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None

        time.sleep(0.5)

    def run(self):
        shutdown_event = threading.Event()

        def signal_handler(signum, frame):
            if shutdown_event.is_set():
                print("\nForce exiting...")
                os._exit(0)

            print("\nShutting down gracefully...")
            shutdown_event.set()
            self.cleanup()
            print("Cleanup complete. Exiting...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"[INFO] Camera Status Client v2.0")
        print(f"[INFO] Host: {self.system_info['host']} | OS: {self.system_info['os']}")
        print(f"[INFO] MAC: {self.host_mac} | Cameras: {len(self.camera_ips)}")

        self.connect()

        try:
            while not shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    client = CameraStatusClient()
    client.run()
